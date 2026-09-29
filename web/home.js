/* Home: shelves by the server's insight sentence (else category) with an explicit Load more. Cards keep their identity across
   appends; a refused cursor restarts once and keeps the shelves; the restart budget
   re-arms only when a new card lands on a shelf. A skeleton shelf holds the space until
   the first page lands; completed loads are announced in a persistent polite region. */
import { deliverFeed, isStaleCursor, itemKey } from "./api.js";
import { announce, building, formatNumber, partial, setState } from "./state.js";
import { gridCard, stopVideos } from "./cards.js";
import { observeView } from "./watch.js";
import { icon } from "./icons.js";

function count(n, one, many) { return formatNumber(n, 0) + " " + (n === 1 ? one : many); }

function skeleton() {
  const section = document.createElement("div");
  section.className = "shelf is-skeleton"; section.setAttribute("aria-hidden", "true");
  const head = document.createElement("div"); head.className = "skeleton-line";
  const row = document.createElement("div"); row.className = "shelf-row";
  for (let i = 0; i < 5; i++) { const card = document.createElement("div"); card.className = "skeleton-card"; row.appendChild(card); }
  section.append(head, row);
  return section;
}

export function mountHome(host) {
  const view = document.createElement("div");
  view.className = "view";
  view.setAttribute("aria-label", "Home");
  view.innerHTML = '<div class="view-head"><h1>Home</h1><p>Your picks, grouped by why each one was picked.</p></div>';
  const status = document.createElement("div"); status.setAttribute("data-ai-home-status", "1");
  const body = document.createElement("div"); body.setAttribute("data-ai-home-body", "1");
  const done = document.createElement("p"); done.className = "sr-only"; done.setAttribute("role", "status");
  const guide = document.createElement("div"); guide.className = "empty-guide"; guide.hidden = true;
  guide.innerHTML = "<h2>Your library has nothing to show yet</h2><p>Add videos or images to the media folder the server was started with, then restart it. " +
    "For a title, tags or a duration, put a <code>&lt;file&gt;.json</code> sidecar next to a file, for example <code>clip.mp4.json</code>.</p>" +
    "<p>If the folder already has media, the engine may still be building: retry in a moment.</p>";
  const more = document.createElement("div"); more.className = "load-more";
  const loadMore = document.createElement("button"); loadMore.type = "button"; loadMore.textContent = "Load more"; loadMore.setAttribute("data-ai-load-more", "1");
  more.appendChild(loadMore);
  view.append(status, guide, body, more, done);
  host.appendChild(view);
  const state = { used: {}, offset: 0, cursor: null, hasMore: true, loading: false, restarted: false, generation: 0, observers: [], count: 0 };

  function homeState(name, retry) {
    setState(status, name, retry);
    building(status, name === "loading");
    const first = !body.querySelector(".shelf:not(.is-skeleton)");
    const bone = body.querySelector(".is-skeleton");
    if (name === "loading" && first && !bone) body.appendChild(skeleton());
    else if (name !== "loading" && bone) bone.remove();
    guide.hidden = !(name === "empty" && first);
  }

  /* A shelf is a horizontal row with paging arrows; the first shelf is the lead and
     shows larger cards. */
  function shelf(category) {
    let section = body.querySelector('[data-ai-shelf="' + CSS.escape(category) + '"]');
    if (!section) {
      section = document.createElement("section"); section.className = "shelf"; section.setAttribute("data-ai-shelf", category);
      if (!body.querySelector(".shelf:not(.is-skeleton)")) section.classList.add("is-lead");
      const head = document.createElement("div"); head.className = "shelf-head";
      const title = document.createElement("h2"); title.textContent = category; const tally = document.createElement("small"); tally.className = "num";
      head.append(title, tally);
      const holder = document.createElement("div"); holder.className = "shelf-holder";
      const row = document.createElement("div"); row.className = "shelf-row"; row.setAttribute("role", "list"); row.setAttribute("aria-label", category);
      holder.appendChild(row);
      [["prev", "prev", -1, "Scroll " + category + " left"], ["next", "forward", 1, "Scroll " + category + " right"]].forEach(([cls, glyph, direction, label]) => {
        const arrow = document.createElement("button"); arrow.type = "button"; arrow.className = "shelf-arrow " + cls; arrow.appendChild(icon(glyph)); arrow.setAttribute("aria-label", label);
        arrow.onclick = () => row.scrollBy({ left: direction * row.clientWidth * 0.85, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
        holder.appendChild(arrow);
      });
      section.append(head, holder); body.appendChild(section);
    }
    return section;
  }

  function render(items) {
    items.forEach(item => {
      const key = itemKey(item);
      if (state.used[key]) return;
      const section = shelf(item.shelf || item.category || item.kind);
      const card = gridCard(item, { headingLevel: 3, eager: state.count === 0 });
      card.setAttribute("role", "listitem");
      section.querySelector(".shelf-row").appendChild(card);
      state.used[key] = card;
      const observer = observeView(card, item, state.count++, "home");
      if (observer) state.observers.push(observer);
      section.querySelector("small").textContent = count(section.querySelectorAll(".card").length, "item", "items");
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
      body.querySelectorAll(".is-skeleton").forEach(bone => bone.remove()); guide.hidden = true;
      announce(done, count(Object.keys(state.used).length - shelved, "new pick", "new picks") + " loaded, " + count(body.querySelectorAll(".shelf").length, "shelf", "shelves") + " in total.");
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
  return { dispose() { state.generation++; stopVideos(host); building(status, false); state.observers.forEach(o => o.disconnect()); }, state, load };
}
