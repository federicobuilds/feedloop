/* Feed: a scroll-snap column that requests one page at a time. A refused continuation
   cursor restarts once at offset 0 with new delivery identities; rendered cells stay,
   duplicates are dropped by qualified identity, and only genuinely new content re-arms
   the automatic restart. Manual Retry discards the dead cursor and re-arms one restart. */
import { deliverFeed, isStaleCursor, itemKey } from "./api.js";
import { building, partial, setState } from "./state.js";
import { explanation } from "./explain.js";
import { feedbackControls } from "./feedback.js";
import { media, metaLine } from "./cards.js";
import { observeView } from "./watch.js";

export function mountFeed(host) {
  const col = document.createElement("div");
  col.className = "feed-col";
  col.setAttribute("aria-label", "Feed");
  col.setAttribute("data-ai-feed", "1");
  host.appendChild(col);
  const state = { items: [], seen: new Set(), offset: 0, cursor: null, hasMore: true, fetching: false, restarted: false, emptyPages: 0, generation: 0, observers: [] };

  function feedState(name, retry) {
    let row = col.querySelector(":scope > [data-ai-state]");
    if (!name) { if (row) { building(row, false); row.remove(); } return; }
    if (!row) { row = document.createElement("div"); row.className = "feed-cell"; row.style.display = "flex"; row.style.flexDirection = "column"; col.appendChild(row); }
    setState(row, name, retry || (() => fetchMore()));
    building(row, name === "loading");
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
      cell.setAttribute("aria-label", (item.title || "Item " + item.id) + " (" + itemKey(item) + ")");
      cell.tabIndex = 0;
      const mediaBox = media(item, { autoplay: false });
      mediaBox.className = "feed-media";
      const side = document.createElement("div");
      side.className = "card-side";
      const title = document.createElement("h2"); title.className = "card-title"; title.textContent = item.title || ("Item " + item.id);
      const reason = document.createElement("p"); reason.className = "card-reason"; reason.textContent = item.reason || "Serving explanation unavailable";
      side.append(title, reason, metaLine(item), feedbackControls(item), explanation(item));
      cell.append(mediaBox, side);
      if (row) col.insertBefore(cell, row); else col.appendChild(cell);
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
  col.addEventListener("keydown", event => {
    const cells = col.querySelectorAll("[data-idx]");
    const current = document.activeElement.closest && document.activeElement.closest("[data-idx]");
    const index = current ? Number(current.getAttribute("data-idx")) : -1;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      if (index + 1 < cells.length) cells[index + 1].focus(); else fetchMore();
    } else if (event.key === "ArrowUp" && index > 0) { event.preventDefault(); cells[index - 1].focus(); }
  });
  fetchMore();
  return { dispose() { state.generation++; feedState(null); state.observers.forEach(o => o.disconnect()); }, state, fetchMore };
}
