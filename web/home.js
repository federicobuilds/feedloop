/* Home: shelves by category with an explicit Load more. Cards keep their identity across
   appends; a refused cursor restarts once and keeps the shelves; the restart budget
   re-arms only when a new card lands on a shelf. */
import { deliverFeed, isStaleCursor, itemKey } from "./api.js";
import { building, partial, setState } from "./state.js";
import { gridCard } from "./cards.js";
import { observeView } from "./watch.js";

export function mountHome(host) {
  const view = document.createElement("div");
  view.className = "view";
  view.setAttribute("aria-label", "Home");
  view.innerHTML = '<div class="view-head"><h1>Home</h1><p>Shelves grouped by the category the ranker recorded for each pick.</p></div>';
  const status = document.createElement("div"); status.setAttribute("data-ai-home-status", "1");
  const body = document.createElement("div"); body.setAttribute("data-ai-home-body", "1");
  const more = document.createElement("div"); more.className = "load-more";
  const loadMore = document.createElement("button"); loadMore.type = "button"; loadMore.textContent = "Load more"; loadMore.setAttribute("data-ai-load-more", "1");
  more.appendChild(loadMore);
  view.append(status, body, more);
  host.appendChild(view);
  const state = { used: {}, offset: 0, cursor: null, hasMore: true, loading: false, restarted: false, generation: 0, observers: [], count: 0 };

  function homeState(name, retry) {
    setState(status, name, retry);
    building(status, name === "loading");
  }

  function shelf(category) {
    let section = body.querySelector('[data-ai-shelf="' + category + '"]');
    if (!section) {
      section = document.createElement("section"); section.className = "shelf"; section.setAttribute("data-ai-shelf", category);
      const head = document.createElement("h2"); head.textContent = category; const count = document.createElement("small"); head.appendChild(count);
      const grid = document.createElement("div"); grid.className = "grid";
      section.append(head, grid); body.appendChild(section);
    }
    return section;
  }

  function render(items) {
    items.forEach(item => {
      const key = itemKey(item);
      if (state.used[key]) return;
      const section = shelf(item.category || item.kind);
      const card = gridCard(item);
      section.querySelector(".grid").appendChild(card);
      state.used[key] = card;
      const observer = observeView(card, item, state.count++, "home");
      if (observer) state.observers.push(observer);
      section.querySelector("small").textContent = section.querySelectorAll(".card").length + " items";
    });
  }

  function load(retryDelivery) {
    if (state.loading) return;
    const generation = state.generation;
    const requestOffset = state.offset, requestCursor = state.cursor;
    let restart = false;
    state.loading = true; loadMore.setAttribute("aria-disabled", "true");
    homeState("loading");
    (retryDelivery ? retryDelivery() : deliverFeed({ limit: 24, surface: "home", offset: state.offset, cursor: state.cursor, images: true })).then(d => {
      if (generation !== state.generation) return;
      if (d.pagination && d.pagination.has_more && (!Number.isSafeInteger(d.pagination.next_offset) || d.pagination.next_offset < state.offset + d.items.length || !d.pagination.next_cursor)) {
        const error = new Error("Invalid serving pagination"); error.state = "error"; throw error;
      }
      state.hasMore = !(d.pagination && d.pagination.has_more === false);
      loadMore.textContent = state.hasMore ? "Load more" : "End of results";
      // the served continuation is recorded even for an empty page, so the next request follows the server's offset and cursor
      state.offset = Number.isSafeInteger((d.pagination || {}).next_offset) ? d.pagination.next_offset : state.offset;
      state.cursor = d.pagination && d.pagination.next_cursor;
      if (!d.items.length) { homeState(d.status === "partial" ? "partial" : "empty", () => load()); return; }
      const shelved = Object.keys(state.used).length;
      render(d.items);
      if (Object.keys(state.used).length > shelved) state.restarted = false;
      building(status, false); status.textContent = ""; status.removeAttribute("data-ai-state");
      partial(status, d, () => { state.offset = requestOffset; state.cursor = requestCursor; load(); });
    }).catch(error => {
      if (generation !== state.generation) return;
      if (isStaleCursor(error) && requestCursor) {
        if (!state.restarted) { state.restarted = true; state.offset = 0; state.cursor = null; restart = true; return; }
        homeState(error.state || "error", () => { state.offset = 0; state.cursor = null; state.restarted = false; load(); });
        return;
      }
      homeState(error.state || "unavailable", () => load(error.retryDelivery));
    }).finally(() => {
      if (generation === state.generation) {
        state.loading = false; loadMore.setAttribute("aria-disabled", String(!state.hasMore));
        if (restart) load();
      }
    });
  }

  loadMore.onclick = () => { if (loadMore.getAttribute("aria-disabled") !== "true") load(); };
  load();
  return { dispose() { state.generation++; building(status, false); state.observers.forEach(o => o.disconnect()); }, state, load };
}
