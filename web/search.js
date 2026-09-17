/* Search: text over the semantic and sound spaces. Uses the ordinary status region,
   never the Feed building panel; an empty result, a partial result and a missing
   feature stay distinct. */
import { getJSON } from "./api.js";
import { partial, setState } from "./state.js";
import { gridCard } from "./cards.js";

export function mountSearch(host, { onSeed } = {}) {
  const view = document.createElement("div");
  view.className = "view";
  view.setAttribute("aria-label", "Search");
  view.innerHTML = '<div class="view-head"><h1>Search</h1><p>Find items by words. Results need a text encoder and a matching feature space.</p></div>' +
    '<form class="search-form" role="search"><label class="sr-only" for="ai-search-query">Search words</label>' +
    '<input id="ai-search-query" name="q" type="search" autocomplete="off" spellcheck="false" placeholder="sunset, harbor, night\u2026">' +
    '<label class="sr-only" for="ai-search-mode">Mode</label><select id="ai-search-mode" name="mode"><option value="look">Look</option><option value="sound">Sound</option><option value="both">Both</option></select>' +
    '<button type="submit">Search</button></form><p class="results-head" data-ai-search-head hidden></p>' +
    '<div data-ai-search-status></div><div class="grid" data-ai-search-grid></div>';
  host.appendChild(view);
  const form = view.querySelector("form"), field = view.querySelector("#ai-search-query"), mode = view.querySelector("#ai-search-mode");
  const status = view.querySelector("[data-ai-search-status]"), grid = view.querySelector("[data-ai-search-grid]"), head = view.querySelector("[data-ai-search-head]");
  let generation = 0;

  function run(q) {
    const current = ++generation;
    grid.textContent = ""; head.hidden = true;
    if (!q.trim()) { setState(status, "empty"); return; }
    setState(status, "loading");
    getJSON("search?" + new URLSearchParams({ q, mode: mode.value, limit: "24" }), "items").then(d => {
      if (current !== generation) return;
      const items = d.items || [];
      if (!items.length) { setState(status, d.status === "partial" ? "partial" : "empty", () => run(q)); return; }
      status.textContent = ""; status.removeAttribute("data-ai-state");
      head.hidden = false; head.textContent = items.length + " matches for \u201C" + q + "\u201D";
      items.forEach(item => grid.appendChild(gridCard(item, { seedAction: onSeed })));
      partial(status, d, () => run(q));
    }).catch(error => {
      if (current !== generation) return;
      setState(status, error.state || "unavailable", () => run(q));
    });
  }
  /* The hash is the route state (q and mode); a submit that changes it remounts the view,
     which dispatches once from that state. An unchanged hash dispatches directly. */
  form.addEventListener("submit", event => {
    event.preventDefault();
    const next = "#/search?" + new URLSearchParams({ q: field.value, mode: mode.value }).toString();
    if (location.hash === next) run(field.value); else location.hash = next;
  });
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  if (["look", "sound", "both"].includes(params.get("mode"))) mode.value = params.get("mode");
  const initial = params.get("q");
  if (initial) { field.value = initial; run(initial); }
  return { dispose() { generation++; }, run };
}
