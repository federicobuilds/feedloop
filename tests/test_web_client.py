"""The packaged no-build client: every asset it loads is served offline by the server, the
hooks the browser contract runner drives are present, and the endpoints the client calls
return the fields its modules read. No browser; the server is driven in process."""
import re

import pytest

from fl3_helpers import feed_request, make_app, request
from feedloop.server.app import web_root

IMPORT = re.compile(r"""^import [^"']+ from ["']\./([\w-]+\.js)["'];""", re.M)
ASSET = re.compile(r"""(?:href|src)="/([\w-]+\.(?:css|js))\"""")


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    return make_app(tmp_path, monkeypatch, segments=True)


def client_files():
    """index.html's assets plus every module reachable through relative imports."""
    root = web_root()
    pending, seen = ASSET.findall((root / "index.html").read_text()), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name.endswith(".js"):
            pending.extend(IMPORT.findall((root / name).read_text()))
    return seen


def test_every_client_asset_is_served_with_its_type(ctx):
    names = client_files()
    assert {"styles.css", "app.js", "cards.js", "feed.js", "home.js", "search.js", "similar.js", "engine.js"} <= names
    for name in sorted(names):
        status, headers, body = ctx.app.handle("GET", "/" + name, {}, b"")
        kind = dict(headers)["Content-Type"]
        assert status == 200 and body, name
        assert ("css" in kind) if name.endswith(".css") else ("javascript" in kind), (name, kind)
    for route in ("/", "/feed", "/home", "/search", "/similar", "/engine"):
        status, headers, body = ctx.app.handle("GET", route, {}, b"")
        assert status == 200 and b'href="/styles.css"' in body and b'src="/app.js"' in body


def test_client_works_offline_and_keeps_its_hooks():
    root = web_root()
    text = {path.name: path.read_text() for path in root.iterdir() if path.suffix in (".html", ".css", ".js")}
    for name, body in text.items():
        assert not re.search(r"https?://|//cdn|@import", body), name
    html = text["index.html"]
    assert all(f'data-view="{view}"' in html for view in ("feed", "home", "search", "engine"))
    assert 'data-view="similar"' not in html
    js = "\n".join(body for name, body in text.items() if name.endswith(".js"))
    for hook in ("data-ai-feed", "data-idx", "data-ai-home-key", "data-ai-home-status", "data-ai-load-more", "data-ai-search-grid",
                 "data-ai-search-status", "ai-search-mode", "data-ai-similar-grid", "data-ai-feedback", "data-ai-undo",
                 "data-ai-ledger", "data-ai-ledger-row", "aid-revert", "aid-tuner-reset", "aid-tick", "data-ai-tuner-error",
                 "data-ai-capture-status", "data-ai-building", "data-ai-retry", "data-prefix", "data-ai-summary", "data-ai-jump"):
        assert hook in js, hook
    # the prefixes the search box offers are the ones the engine parses
    assert 'data-prefix="sound:"' in js and 'data-prefix="both:"' in js


def test_feed_jump_to_moment_uses_the_served_moment_and_the_player_start():
    root = web_root()
    feed, cards, icons = ((root / name).read_text() for name in ("feed.js", "cards.js", "icons.js"))
    assert "moment:" in icons and 'icon("moment")' in feed
    jump = feed[feed.index('if (video && typeof item.best_t === "number")'):feed.index("extras.push(jump)")]
    # only a video with a served moment gets the button, named for the jump it makes
    assert 'typeof item.best_t === "number"' in jump and 'jump.type = "button"' in jump
    assert '"aria-label", "Jump to the matching moment at "' in jump
    # the jump seeks to the player's own start, which openVideo also puts in the source's #t= fragment
    assert "video.__loadSource()" in jump and "video.currentTime = video.__start" in jump
    assert 'item.media_url + "#t=" + video.__start' in cards


def test_grid_cards_use_host_preview_and_open_urls():
    cards = (web_root() / "cards.js").read_text()
    thumb = cards[cards.index("function thumb("):cards.index("const SIDECAR_REASONS")]
    # open_url makes the card media a real link; preview_url feeds only the muted hover preview
    assert 'document.createElement(page ? "a" : "button")' in thumb and "opener.href = page" in thumb
    assert "if (page) return box;" in thumb and "openVideo(video, item, null, preview)" in thumb
    assert 'video.src = item.media_url + "#t=" + video.__start' in thumb and "item.preview_url" not in cards[:cards.index("function thumb(")]


def test_feed_page_carries_the_fields_the_cards_read(ctx):
    ctx.clock.advance(30)
    status, page = request(ctx, "POST", "/api/feed", feed_request())
    assert status == 200 and page["status"] in ("ok", "partial") and page["items"]
    assert set(page["pagination"]) >= {"has_more", "next_offset", "next_cursor"}
    # engine.js reads the ranker's profile summary kept from the latest Feed page
    assert isinstance(page["profile"], dict)
    assert set(page["profile"]) <= {"reason", "profile_tags", "positive_tags", "negative_tags", "corpus_items", "watch_evidence_s", "explicit_only_items"}
    for item in page["items"]:
        assert {"kind", "id", "title", "reason", "score", "media_url", "duration", "best_t", "category", "explore",
                "explanation", "request_id", "served_item_id"} <= set(item), sorted(item)
        assert isinstance(item["id"], int) and isinstance(item["title"], str) and item["media_url"].startswith("/media/" + item["kind"] + "/")
        assert item["score"] is None or isinstance(item["score"], (int, float))


def test_search_similar_and_scorecard_carry_the_fields_the_views_read(ctx):
    ctx.clock.advance(30)
    status, found = request(ctx, "GET", "/api/search?q=amber&limit=24", auth=False)
    assert status == 200 and found["items"]
    for item in found["items"]:
        assert {"kind", "id", "title", "media_url", "duration_s", "score", "search"} <= set(item) and "best_t" in item["search"]
    # a typed prefix is the engine's mode, as the search box shows it
    assert request(ctx, "GET", "/api/search?q=sound:%20amber", auth=False)[1]["status"] == request(ctx, "GET", "/api/search?q=amber&mode=sound", auth=False)[1]["status"]
    status, alike = request(ctx, "GET", "/api/similar?kind=video&id=1&limit=24", auth=False)
    assert status == 200 and alike["items"]
    for item in alike["items"]:
        assert {"kind", "id", "title", "media_url", "duration_s", "score", "similar"} <= set(item)
    status, card = request(ctx, "GET", "/api/scorecard", auth=False)
    assert status == 200 and {"tuner", "automation", "evidence", "capture", "gate"} <= set(card)
    assert isinstance(card["tuner"].get("knobs", []), list) and isinstance(card["tuner"].get("ledger", []), list)
    assert "enabled" in card["automation"] and {"min_trials_per_arm", "ripen_hours"} <= set(card["gate"])
    status, config = request(ctx, "GET", "/api/config", auth=False)
    assert status == 200 and {"primary", "attribution", "encoder", "space_meta"} <= set(config)
