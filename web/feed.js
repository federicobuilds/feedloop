/* Feed: a scroll-snap column that requests one page at a time. A refused continuation
   cursor restarts once at offset 0 with new delivery identities; rendered cells stay,
   duplicates are dropped by qualified identity, and only genuinely new content re-arms
   the automatic restart. Manual Retry discards the dead cursor and re-arms one restart. */
import { deliverFeed, isStaleCursor, itemKey } from "./api.js";
import { building, formatTime, partial, setState } from "./state.js";
import { evidenceStrip, explanation } from "./explain.js";
import { feedbackControls } from "./feedback.js";
import { media, metaLine, similarLink, stopVideos, whyText } from "./cards.js";
import { observeView } from "./watch.js";
import { icon } from "./icons.js";

/* Shortcuts never take keys meant for a control, the player, or a modified chord. */
function ownsKeys(target) {
  return !!(target.closest && target.closest("input, textarea, select, button, video, audio, summary, [contenteditable]:not([contenteditable='false'])"));
}

const MUTE_KEY = "feedloop-muted";

export function mountFeed(host) {
  const view = document.createElement("div");
  view.className = "feed-view";
  const heading = document.createElement("h1"); heading.className = "sr-only"; heading.textContent = "Feed";
  const col = document.createElement("div");
  col.className = "feed-col";
  col.setAttribute("aria-label", "Feed");
  col.setAttribute("data-ai-feed", "1");
  view.append(heading, col);
  host.appendChild(view);
  const state = { items: [], seen: new Set(), offset: 0, cursor: null, hasMore: true, fetching: false, restarted: false, emptyPages: 0, generation: 0, observers: [] };

  function feedState(name, retry) {
    let row = col.querySelector(":scope > [data-ai-state]");
    if (!name) { if (row) { building(row, false); row.remove(); } return; }
    if (!row) { row = document.createElement("div"); row.className = "feed-cell"; row.style.display = "flex"; row.style.flexDirection = "column"; col.appendChild(row); }
    setState(row, name, retry || (() => fetchMore()));
    building(row, name === "loading");
  }

  /* Sound is one setting for every Feed video, muted until the user turns it on, because
     browsers only autoplay muted media before a gesture. */
  let muted = localStorage.getItem(MUTE_KEY) !== "0", active = null, resume = false;
  function paintMute(button) {
    button.replaceChildren(icon(muted ? "muted" : "sound"));
    button.setAttribute("aria-pressed", String(!muted));
  }
  function setMuted(value, remember) {
    muted = value;
    if (remember) localStorage.setItem(MUTE_KEY, value ? "1" : "0");
    col.querySelectorAll(".feed-media video").forEach(video => { video.muted = value; });
    col.querySelectorAll(".feed-mute").forEach(paintMute);
  }
  function start(video) {
    video.muted = muted;
    video.play().catch(error => {
      if (error.name !== "NotAllowedError" || video.muted) return;
      setMuted(true, false);
      video.play().catch(() => {});
    });
  }

  /* The cell at least 60 percent visible (the rule watch.js qualifies views by) is the
     current one: its video plays and the next one preloads; every other video is paused. */
  function activate(cell) {
    if (active === cell) return;
    if (active) active.classList.remove("is-active");
    active = cell;
    cell.classList.add("is-active");
    col.querySelectorAll(".feed-media video").forEach(video => { if (!cell.contains(video)) video.pause(); });
    [cell, cell.nextElementSibling].forEach(near => {
      const video = near && near.querySelector(".feed-media video");
      if (video) { video.preload = "auto"; video.__loadSource(); }
    });
    const video = cell.querySelector(".feed-media video");
    if (video && document.visibilityState === "visible") start(video);
  }
  const stage = new IntersectionObserver(entries => entries.forEach(entry => {
    if (entry.isIntersecting && entry.intersectionRatio >= 0.6) { activate(entry.target); return; }
    entry.target.querySelectorAll("video").forEach(video => video.pause());
    if (active === entry.target) { active.classList.remove("is-active"); active = null; }
  }), { root: col, threshold: [0, 0.6, 1] });
  function onVisibility() {
    const video = active && active.querySelector(".feed-media video");
    if (!video) return;
    if (document.visibilityState !== "visible") { resume = !video.paused; video.pause(); }
    else if (resume) { resume = false; start(video); }
  }
  document.addEventListener("visibilitychange", onVisibility);

  function move(index) {
    const cells = col.querySelectorAll("[data-idx]");
    if (index < 0) return;
    if (index < cells.length) cells[index].focus(); else fetchMore();
  }

  function render(fresh) {
    const row = col.querySelector(":scope > [data-ai-state]");
    fresh.forEach(item => {
      const index = state.items.length;
      state.items.push(item);
      state.seen.add(itemKey(item));
      const cell = document.createElement("section");
      cell.className = "feed-cell";
      cell.setAttribute("data-idx", String(index));
      cell.setAttribute("data-ai-key", itemKey(item));
      cell.setAttribute("aria-label", (item.title || "Item " + item.id) + ", pick " + (index + 1));
      cell.tabIndex = 0;
      const mediaBox = media(item, { root: col, eager: index === 0 });
      mediaBox.className = "feed-media";
      const video = mediaBox.querySelector("video");
      if (video) {
        video.muted = muted;
        // Loops like a short-video feed; replaying after ended keeps each pass its own watch stream.
        video.addEventListener("ended", () => { if (cell === active) video.play().catch(() => {}); });
        video.addEventListener("volumechange", () => { if (cell === active && video.muted !== muted) setMuted(video.muted, true); });
        const mute = document.createElement("button");
        mute.type = "button"; mute.className = "feed-mute"; mute.setAttribute("aria-label", "Sound"); mute.title = "Sound (M)";
        paintMute(mute);
        mute.onclick = () => setMuted(!muted, true);
        mediaBox.appendChild(mute);
      }
      const side = document.createElement("div");
      side.className = "card-side";
      const title = document.createElement("h2"); title.className = "card-title"; title.textContent = item.title || ("Item " + item.id);
      const why = explanation(item);
      const reason = document.createElement("p"); reason.className = "card-reason"; reason.textContent = whyText(item, reason, video);
      const next = document.createElement("button"); next.type = "button"; next.className = "action primary";
      next.append(icon("next"), "Next"); next.onclick = () => move(index + 1);
      const extras = [next];
      // 2026-09-30: the served moment again, through the player's own start (the #t= source fragment when not loaded yet)
      if (video && typeof item.best_t === "number") {
        const jump = document.createElement("button");
        jump.type = "button"; jump.className = "action"; jump.setAttribute("data-ai-jump", "1");
        // 2026-09-30: named for the player's start, which openVideo clamps to 0 once metadata shows the moment is past the end
        const name = () => {
          jump.setAttribute("aria-label", "Jump to the matching moment at " + formatTime(video.__start)); jump.title = jump.getAttribute("aria-label");
          jump.replaceChildren(icon("moment"), formatTime(video.__start));
        };
        name();
        video.addEventListener("loadedmetadata", name);
        jump.onclick = () => {
          if (!video.getAttribute("src")) video.__loadSource();
          video.currentTime = video.__start;
          start(video);
        };
        extras.push(jump);
      }
      if (item.kind === "video") extras.push(similarLink(item, false));
      const hint = document.createElement("p"); hint.className = "feed-hint";
      hint.innerHTML = "<kbd>J</kbd> <kbd>K</kbd> or arrows to move, <kbd>Space</kbd> to pause, <kbd>M</kbd> for sound, <kbd>L</kbd> <kbd>D</kbd> to rate";
      side.append(title, evidenceStrip(item, why, { legend: true }), reason, metaLine(item, video), feedbackControls(item, { extras }), why, hint);
      cell.append(mediaBox, side);
      if (row) col.insertBefore(cell, row); else col.appendChild(cell);
      stage.observe(cell);
      const observer = observeView(cell, item, index, "feed");
      if (observer) state.observers.push(observer);
    });
  }

  function fetchMore(retryDelivery) {
    if (state.fetching || !state.hasMore && !retryDelivery) return;
    state.fetching = true;
    const generation = state.generation, requestCursor = state.cursor;
    feedState("loading");
    (retryDelivery ? retryDelivery() : deliverFeed({ limit: 24, surface: "feed", offset: state.offset, cursor: state.cursor, images: true })).then(d => {
      if (generation !== state.generation) return;
      if (d.pagination && d.pagination.has_more && (!Number.isSafeInteger(d.pagination.next_offset) || d.pagination.next_offset < state.offset + d.items.length || !d.pagination.next_cursor)) {
        const error = new Error("Invalid serving pagination"); error.state = "error"; throw error;
      }
      const fresh = (d.items || []).filter(item => !state.seen.has(itemKey(item)));
      if (fresh.length) feedState(null);
      render(fresh);
      state.offset = Number.isSafeInteger((d.pagination || {}).next_offset) ? d.pagination.next_offset : state.offset;
      state.cursor = d.pagination && d.pagination.next_cursor;
      state.hasMore = !(d.pagination && d.pagination.has_more === false);
      state.fetching = false;
      if (fresh.length) { state.restarted = false; state.emptyPages = 0; }
      partial(col, d, () => fetchMore());
      if (!fresh.length) {
        if ((d.items || []).length && state.hasMore && d.status !== "partial" && ++state.emptyPages <= 3) { fetchMore(); return; }
        state.emptyPages = 0;
        feedState(d.status === "partial" ? "partial" : "empty", state.hasMore ? () => fetchMore() : null);
      }
      col.dispatchEvent(new CustomEvent("feedloop:page", { detail: { fresh: fresh.length } }));
    }).catch(error => {
      if (generation !== state.generation) return;
      state.fetching = false;
      const stale = isStaleCursor(error) && !!requestCursor;
      if (stale && !state.restarted) { state.restarted = true; state.offset = 0; state.cursor = null; fetchMore(); return; }
      feedState(error.state || "unavailable", stale
        ? () => { state.offset = 0; state.cursor = null; state.restarted = false; fetchMore(); }
        : () => fetchMore(error.retryDelivery));
    });
  }

  col.addEventListener("scroll", () => {
    if (col.scrollTop + col.clientHeight >= col.scrollHeight - col.clientHeight / 2) fetchMore();
  });
  view.addEventListener("keydown", event => {
    if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || ownsKeys(event.target)) return;
    const current = event.target.closest && event.target.closest("[data-idx]");
    const index = current ? Number(current.getAttribute("data-idx")) : -1;
    const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
    if (key === "ArrowDown" || key === "j") { event.preventDefault(); move(index + 1); }
    else if ((key === "ArrowUp" || key === "k") && index > 0) { event.preventDefault(); move(index - 1); }
    else if (key === "m") { event.preventDefault(); setMuted(!muted, true); }
    else if ((key === "l" || key === "d") && current) { event.preventDefault(); current.querySelector(key === "l" ? "[data-ai-feedback='like']" : "[data-ai-feedback='dislike']").click(); }
    else if (key === " " && current) {
      const video = current.querySelector("video");
      if (video) { event.preventDefault(); if (video.paused) video.play().catch(() => {}); else video.pause(); }
    }
  });
  fetchMore();
  return { dispose() { state.generation++; feedState(null); stage.disconnect(); stopVideos(host); document.removeEventListener("visibilitychange", onVisibility); if (col.__nearObserver) col.__nearObserver.disconnect(); state.observers.forEach(o => o.disconnect()); }, state, fetchMore };
}
