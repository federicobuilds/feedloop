"""Runpod REST v2 lifecycle for disposable ingest workers."""

from __future__ import annotations

import base64
import ipaddress
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


API_URL = "https://api.runpod.io/v2"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
RecordAllocation = Callable[[str, str | None, int | None], None]


class RunpodError(RuntimeError):
    """A stable reason code, without provider response text or credentials."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RunpodAPI:
    """An authenticated provider client. Each call makes one HTTP request."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key if api_key is not None else os.environ.get("RUNPOD_API_KEY")
        if not self._api_key or any(character.isspace() for character in self._api_key):
            raise RunpodError("runpod_api_key_required")
        self._opener = build_opener(_NoRedirect())

    def __call__(self, method: str, path: str, body: Mapping | None = None, *, deadline: float):
        target = urlsplit(path)
        if (not path.startswith("/") or target.scheme or target.netloc or target.fragment
                or path.startswith("//") or "\\" in target.path or ".." in target.path.split("/")):
            raise RunpodError("pod_api_path_invalid")
        timeout = min(60, deadline - time.time())
        if timeout <= 0:
            raise RunpodError("pod_deadline")
        request = Request(
            API_URL + path,
            data=None if body is None else json.dumps(body, allow_nan=False).encode(),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "feedloop/1.0",
            },
            method=method,
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                status = response.status
                content = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            error.close()
            return error.code, {}
        except (URLError, OSError):
            raise RunpodError("pod_transport_unknown") from None
        if len(content) > MAX_RESPONSE_BYTES:
            raise RunpodError("pod_response_invalid")
        try:
            data = json.loads(content) if content else {}
        except (ValueError, UnicodeError):
            raise RunpodError("pod_response_invalid") from None
        if not isinstance(data, (dict, list)):
            raise RunpodError("pod_response_invalid")
        return status, data


def _pod_path(pod_id: str) -> str:
    if not isinstance(pod_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", pod_id) is None:
        raise RunpodError("pod_id_invalid")
    return "/pods/" + pod_id


def list_pods(api, *, deadline: float, clock=time) -> list[dict]:
    """Read every page before deciding whether an allocation name is unique."""
    rows, cursor, seen = [], None, set()
    while clock.time() < deadline:
        path = "/pods" + ("?" + urlencode({"cursor": cursor}) if cursor else "")
        status, data = api("GET", path, deadline=deadline)
        if status != 200:
            raise RunpodError("pod_list_unavailable")
        page = data.get("pods") if isinstance(data, dict) else None
        if not isinstance(page, list):
            raise RunpodError("pod_list_response_invalid")
        for row in page:
            if not isinstance(row, dict):
                raise RunpodError("pod_list_response_invalid")
            _pod_path(row.get("id"))
        rows.extend(page)
        pagination = data.get("pagination")
        if not isinstance(pagination, dict) or type(pagination.get("hasNextPage")) is not bool:
            raise RunpodError("pod_list_pagination_invalid")
        if not pagination["hasNextPage"]:
            return rows
        cursor = pagination.get("nextCursor")
        if not isinstance(cursor, str) or not cursor or cursor in seen:
            raise RunpodError("pod_list_pagination_invalid")
        seen.add(cursor)
    raise RunpodError("pod_deadline")


def resolve_creation(api, name: str, *, deadline: float, clock=time) -> str | None:
    """Recover one exact allocation. No match leaves its creation unresolved."""
    if not isinstance(name, str) or not name:
        raise RunpodError("pod_allocation_name_required")
    matches = [row["id"] for row in list_pods(api, deadline=deadline, clock=clock) if row.get("name") == name]
    return matches[0] if len(matches) == 1 else None


def hourly_rate(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise RunpodError("pod_hourly_rate_unavailable")
    return float(value)


def checked_hourly_rate(value, maximum: float | None) -> float:
    rate = hourly_rate(value)
    if maximum is not None and rate > hourly_rate(maximum):
        raise RunpodError("pod_hourly_rate_exceeds_review")
    return rate


def catalog_quote(api, *, gpu: str, cloud: str, min_cuda: str, deadline: float) -> dict:
    quote = {
        "gpu": gpu, "cloud": cloud, "hourly_rate": None, "currency": "USD",
        "compute_only": True, "availability": "UNKNOWN", "available": None,
    }
    if cloud not in ("SECURE", "COMMUNITY"):
        return dict(quote, reason="pod_cloud_invalid")
    query = urlencode({
        "product": "POD", "include": "AVAILABILITY", "cloud": cloud,
        "count": 1, "minCudaVersion": min_cuda,
    })
    try:
        status, data = api("GET", "/catalog/gpus?" + query, deadline=deadline)
        if status != 200 or not isinstance(data, dict) or not isinstance(data.get("gpus"), list):
            return dict(quote, reason="catalog_unavailable")
        matches = [row for row in data["gpus"] if isinstance(row, dict) and row.get("id") == gpu]
        if len(matches) != 1:
            return dict(quote, reason="gpu_not_in_catalog")
        entry = matches[0]
        availability = entry.get("availability")
        if availability in ("NONE", "LOW", "MEDIUM", "HIGH"):
            quote.update(availability=availability, available=availability != "NONE")
        quote["hourly_rate"] = hourly_rate(entry["price"].get(cloud.lower()))
        return quote
    except (RuntimeError, KeyError, TypeError, AttributeError):
        return dict(quote, reason="catalog_quote_unavailable")


def confirm_absent(api, pod_id: str, *, deadline: float, clock=time) -> dict:
    """Request deletion once, then confirm this exact pod is absent via GET."""
    path = _pod_path(pod_id)
    status, requested = None, False
    try:
        while clock.time() < deadline:
            status, data = api("GET", path, deadline=deadline)
            if status == 404:
                return {"confirmed": True, "http_status": 404}
            if status != 200 or not isinstance(data, dict) or data.get("id") != pod_id:
                break
            if not requested:
                requested = True
                try:
                    api("DELETE", path, deadline=deadline)
                except Exception:
                    pass  # The follow-up GET settles an unknown DELETE response.
                continue
            clock.sleep(min(5, max(0, deadline - clock.time())))
    except Exception:
        pass
    return {"confirmed": False, "http_status": status}


def create_pod(api, body: Mapping, *, deadline: float, record: RecordAllocation, clock=time) -> str:
    """Journal intent before POST. Never retry creation after an unknown result.

    The caller owns the durable journal record and its unique provider name.
    Record receives (status, pod_id, http_status).
    """
    name = body.get("name")
    if not isinstance(name, str) or not name or "queue" in body:
        raise RunpodError("immediate_pod_name_required")
    if clock.time() >= deadline:
        raise RunpodError("pod_deadline")
    record("creating", None, None)
    try:
        status, data = api("POST", "/pods", body, deadline=deadline)
    except Exception:
        record("creation_unknown", None, None)
        raise
    try:
        same_allocation = isinstance(data, dict) and data.get("name", name) == name
        pod_id = data.get("id") if same_allocation and status in (200, 201) else None
        _pod_path(pod_id)
    except RunpodError:
        record("creation_unknown", None, status)
        raise RunpodError("pod_creation_unknown") from None
    try:
        record("active", pod_id, status)
    except Exception:
        cleanup = confirm_absent(api, pod_id, deadline=clock.time() + 120, clock=clock)
        try:
            if cleanup["confirmed"]:
                record("gone", pod_id, cleanup["http_status"])
            else:
                record("cleanup_pending", pod_id, cleanup["http_status"])
        except Exception:
            pass
        reason = "pod_allocation_journal_failed" if cleanup["confirmed"] else "pod_cleanup_unconfirmed"
        raise RunpodError(reason) from None
    return pod_id


def reconcile_allocation(api, *, name: str, pod_id: str | None, deadline: float,
                         record: RecordAllocation, clock=time) -> dict:
    """Settle a journal entry, consulting the exact provider name when necessary."""
    if pod_id is None:
        try:
            pod_id = resolve_creation(api, name, deadline=deadline, clock=clock)
        except Exception:
            return {"confirmed": False, "http_status": None}
        if pod_id is None:
            return {"confirmed": False, "http_status": None}
    try:
        record("cleanup_pending", pod_id, None)
    except Exception:
        pass  # A failure to persist state should not block known resource cleanup.
    result = confirm_absent(api, pod_id, deadline=deadline, clock=clock)
    try:
        status = "gone" if result["confirmed"] else "cleanup_pending"
        record(status, pod_id, result["http_status"])
    except Exception:
        result["journaled"] = False
    return result


def direct_endpoint(info: Mapping) -> tuple[str, int] | None:
    """Use direct SSH fields independently of any provider shell command."""
    ssh = info.get("ssh")
    if not isinstance(ssh, dict) or ssh.get("direct") is None:
        return None
    direct = ssh["direct"]
    if not isinstance(direct, dict) or direct.get("username") != "root":
        raise RunpodError("pod_direct_ssh_invalid")
    try:
        return _endpoint(direct.get("host"), direct.get("port"))
    except RunpodError:
        raise RunpodError("pod_direct_ssh_invalid") from None


def _endpoint(host: str, port: int) -> tuple[str, int]:
    if not isinstance(host, str) or type(port) is not int or not 1 <= port <= 65535:
        raise RunpodError("pod_endpoint_invalid")
    try:
        address = ipaddress.IPv4Address(host)
    except ipaddress.AddressValueError:
        raise RunpodError("pod_endpoint_invalid") from None
    return str(address), port


def compatible_cuda_version(value, minimum: str) -> bool:
    def parsed(version):
        if isinstance(version, str) and re.fullmatch(r"[0-9]{1,3}\.[0-9]{1,3}", version):
            return tuple(map(int, version.split(".")))
        return None

    supported, required = parsed(value), parsed(minimum)
    return supported is not None and required is not None and supported >= required


def pin_host_key(api, pod_id: str, *, host: str, port: int, deadline: float, clock=time) -> str:
    """Get the SSH host key through the authenticated provider API."""
    address, port = _endpoint(host, port)
    path = _pod_path(pod_id) + "/exec"
    for _ in range(12):
        if clock.time() >= deadline:
            raise RunpodError("pod_deadline")
        status, data = api(
            "POST", path, {"command": ["cat", "/etc/ssh/ssh_host_ed25519_key.pub"], "timeoutSeconds": 5},
            deadline=deadline,
        )
        if not isinstance(data, dict):
            break
        if status in (409, 503) or (status == 200 and data.get("exitCode") == 1):
            clock.sleep(min(5, max(0, deadline - clock.time())))
            continue
        if status != 200 or data.get("exitCode") != 0 or data.get("timedOut") or data.get("truncated"):
            break
        fields = str(data.get("stdout", "")).strip().split(None, 2)
        if len(fields) < 2 or fields[0] != "ssh-ed25519" or len(fields[1]) != 68:
            break
        try:
            blob = base64.b64decode(fields[1], validate=True)
        except ValueError:
            break
        if len(blob) != 51 or not blob.startswith(b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20"):
            break
        endpoint = address if port == 22 else f"[{address}]:{port}"
        return f"{endpoint} ssh-ed25519 {fields[1]}\n"
    raise RunpodError("pod_provider_host_key_unavailable")
