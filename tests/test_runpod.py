from __future__ import annotations

import base64
import time

import pytest

from feedloop.runpod import (
    RunpodAPI,
    RunpodError,
    catalog_quote,
    checked_hourly_rate,
    confirm_absent,
    create_pod,
    direct_endpoint,
    list_pods,
    pin_host_key,
    reconcile_allocation,
    resolve_creation,
)


class Provider:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, method, path, body=None, *, deadline):
        self.calls.append((method, path, body))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def deadline():
    return time.time() + 30


def test_create_journals_before_a_single_post_and_accepts_v2_created(deadline):
    provider = Provider((201, {"id": "pod1", "name": "ingest-example"}))
    entries = []

    def record(status, pod_id, http_status):
        if status == "creating":
            assert provider.calls == []
        entries.append((status, pod_id, http_status))

    assert create_pod(provider, {"name": "ingest-example"}, deadline=deadline, record=record) == "pod1"
    assert provider.calls == [("POST", "/pods", {"name": "ingest-example"})]
    assert entries == [("creating", None, None), ("active", "pod1", 201)]


def test_lost_create_reply_is_recoverable_by_exact_name_over_all_pages(deadline):
    provider = Provider(
        RunpodError("pod_transport_unknown"),
        (200, {
            "pods": [{"id": "unrelated", "name": "ingest-example-other"}],
            "pagination": {"hasNextPage": True, "nextCursor": "page+two"},
        }),
        (200, {
            "pods": [{"id": "owned", "name": "ingest-example"}],
            "pagination": {"hasNextPage": False},
        }),
        (200, {"id": "owned"}),
        RunpodError("pod_transport_unknown"),
        (404, {}),
    )
    journal = []
    record = lambda *entry: journal.append(entry)
    with pytest.raises(RunpodError, match="pod_transport_unknown"):
        create_pod(provider, {"name": "ingest-example"}, deadline=deadline, record=record)
    assert journal == [("creating", None, None), ("creation_unknown", None, None)]
    assert reconcile_allocation(
        provider, name="ingest-example", pod_id=None, deadline=deadline, record=record
    ) == {"confirmed": True, "http_status": 404}
    assert journal[-2:] == [("cleanup_pending", "owned", None), ("gone", "owned", 404)]
    assert [(method, path) for method, path, _ in provider.calls] == [
        ("POST", "/pods"),
        ("GET", "/pods"),
        ("GET", "/pods?cursor=page%2Btwo"),
        ("GET", "/pods/owned"),
        ("DELETE", "/pods/owned"),
        ("GET", "/pods/owned"),
    ]


def test_queue_is_outside_the_immediate_worker_lifecycle(deadline):
    provider = Provider((202, {"id": "queued"}))
    journal = []
    record = lambda *entry: journal.append(entry)
    with pytest.raises(RunpodError, match="immediate_pod_name_required"):
        create_pod(provider, {"name": "job", "queue": {}}, deadline=deadline, record=record)
    assert not provider.calls and not journal
    with pytest.raises(RunpodError, match="pod_creation_unknown"):
        create_pod(provider, {"name": "job"}, deadline=deadline, record=record)
    assert journal[-1] == ("creation_unknown", None, 202)


def test_failed_active_journal_causes_known_pod_cleanup(deadline):
    provider = Provider((201, {"id": "pod1"}), (200, {"id": "pod1"}), (204, {}), (404, {}))
    entries = []

    def record(status, pod_id, http_status):
        if status == "active":
            raise OSError("journal unavailable")
        entries.append((status, pod_id, http_status))

    with pytest.raises(RunpodError, match="pod_allocation_journal_failed"):
        create_pod(provider, {"name": "job"}, deadline=deadline, record=record)
    assert entries == [("creating", None, None), ("gone", "pod1", 404)]
    assert provider.calls[-1] == ("GET", "/pods/pod1", None)


@pytest.mark.parametrize("pods", [[], [{"id": "one", "name": "job"}, {"id": "two", "name": "job"}]])
def test_absent_or_nonunique_name_stays_unresolved(pods, deadline):
    provider = Provider((200, {"pods": pods, "pagination": {"hasNextPage": False}}))
    assert resolve_creation(provider, "job", deadline=deadline) is None


def test_list_fails_closed_on_repeated_cursor(deadline):
    response = (200, {"pods": [], "pagination": {"hasNextPage": True, "nextCursor": "same"}})
    provider = Provider(response, response)
    with pytest.raises(RunpodError, match="pod_list_pagination_invalid"):
        list_pods(provider, deadline=deadline)


@pytest.mark.parametrize(
    "responses, status, calls",
    [
        ([(200, {"id": "different"})], 200, ["GET"]),
        ([(403, {})], 403, ["GET"]),
        ([(200, {"id": "owned"}), (404, {}), (503, {})], 503, ["GET", "DELETE", "GET"]),
    ],
)
def test_cleanup_requires_authoritative_lookup_for_exact_id(responses, status, calls, deadline):
    provider = Provider(*responses)
    assert confirm_absent(provider, "owned", deadline=deadline) == {"confirmed": False, "http_status": status}
    assert [method for method, _, _ in provider.calls] == calls


def test_catalog_and_post_create_price_must_both_fit_the_ceiling(deadline):
    provider = Provider((200, {"gpus": [
        {"id": "gpu-other", "availability": "LOW", "price": {"secure": 0.15}},
        {"id": "gpu", "availability": "LOW", "price": {"secure": 0.28}},
    ]}))
    quote = catalog_quote(provider, gpu="gpu", cloud="SECURE", min_cuda="12.8", deadline=deadline)
    assert quote["available"] is True
    assert checked_hourly_rate(quote["hourly_rate"], 0.30) == 0.28
    with pytest.raises(RunpodError, match="pod_hourly_rate_exceeds_review"):
        checked_hourly_rate(0.31, 0.30)
    with pytest.raises(RunpodError, match="pod_hourly_rate_unavailable"):
        checked_hourly_rate(True, 0.30)


def test_host_key_is_obtained_from_authenticated_exec(deadline):
    blob = b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + bytes(range(32))
    key = base64.b64encode(blob).decode()
    info = {"ssh": {"direct": {"host": "192.0.2.10", "port": 32100, "username": "root", "command": "ignored"}}}
    host, port = direct_endpoint(info)
    provider = Provider((200, {"stdout": f"ssh-ed25519 {key} instance\n", "exitCode": 0}))
    assert pin_host_key(provider, "owned", host=host, port=port, deadline=deadline) == (
        f"[192.0.2.10]:32100 ssh-ed25519 {key}\n"
    )
    assert provider.calls == [(
        "POST", "/pods/owned/exec",
        {"command": ["cat", "/etc/ssh/ssh_host_ed25519_key.pub"], "timeoutSeconds": 5},
    )]


def test_invalid_host_and_absolute_api_path_are_rejected_without_transport(deadline):
    with pytest.raises(RunpodError, match="pod_direct_ssh_invalid"):
        direct_endpoint({"ssh": {"direct": {"host": "foo@bar", "port": 22, "username": "root"}}})
    api = RunpodAPI("key-for-offline-tests")
    with pytest.raises(RunpodError, match="pod_api_path_invalid"):
        api("GET", "//example.com/pods", deadline=deadline)
