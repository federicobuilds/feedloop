/* Search: text over the semantic and sound spaces. The engine reads a leading "sound:"
   or "both:" in the words as the mode; the tips row inserts those prefixes and the mode
   menu follows whatever prefix is typed, so the prefix is visible rather than hidden. Uses the ordinary status region,
   never the Feed building panel; an empty result, a partial result and a missing
   feature stay distinct. */
import { getJSON } from "./api.js";
import { partial, setState } from "./state.js";
import { gridCard } from "./cards.js";

export function mountSearch(host, { onSeed } = {}) {
  const view = document.createElement("div");
  view.className = "view";
  view.setAttribute("aria-label", "Search");
  view.innerHTML = '<div class="view-head"><h1>Search</h1><p>Find items by what they look or sound like. Results need a text encoder and a matching feature space.</p></div>' +
    '<form class="search-form" role="search"><label class="sr-only" for="ai-search-query">Search words</label>' +
    '<input id="ai-search-query" name="q" type="search" autocomplete="off" spellcheck="false" placeholder="Describe a look, or start with sound: for a sound\u2026">' +
    '<label class="sr-only" for="ai-search-mode">Mode</label><select id="ai-search-mode" name="mode"><option value="look">Look</option><option value="sound">Sound</option><option value="both">Both</option></select>' +
    '<button type="submit" class="primary">Search</button></form>' +
    '<div class="search-tips" data-ai-search-tips><span>Prefixes:</span><button type="button" data-prefix="sound:" aria-pressed="false">sound:</button>' +
    '<button type="button" data-prefix="both:" aria-pressed="false">both:</button><span>search the sound space, or look and sound together.</span></div><p class="results-head" data-ai-search-head hidden></p>' +
    '<div data-ai-search-status></div><div class="grid" data-ai-search-grid></div>';
  host.appendChild(view);
  const form = view.querySelector("form"), field = view.querySelector("#ai-search-query"), mode = view.querySelector("#ai-search-mode");
  const status = view.querySelector("[data-ai-search-status]"), grid = view.querySelector("[data-ai-search-grid]"), head = view.querySelector("[data-ai-search-head]");
  let generation = 0;
  const tips = view.querySelectorAll("[data-prefix]");
  function prefixOf(text) { const low = text.trimStart().toLowerCase(); return ["sound:", "both:"].find(prefix => low.startsWith(prefix)) || null; }
  function syncPrefix() {
    const prefix = prefixOf(field.value);
    if (prefix) mode.value = prefix.slice(0, -1);
    tips.forEach(button => button.setAttribute("aria-pressed", String(button.dataset.prefix === prefix)));
  }
  tips.forEach(button => button.addEventListener("click", () => {
    const current = prefixOf(field.value);
    const rest = current ? field.value.trimStart().slice(current.length).trimStart() : field.value;
    field.value = current === button.dataset.prefix ? rest : button.dataset.prefix + " " + rest;
    if (current === button.dataset.prefix) mode.value = "look";
    syncPrefix(); field.focus();
  }));
  field.addEventListener("input", syncPrefix);

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
      head.hidden = false; head.textContent = items.length + " matches for \u201C" + q + "\u201D in " + ({ look: "look", sound: "sound", both: "look and sound" })[mode.value];
      items.forEach(item => grid.appendChild(gridCard(item, { seedAction: onSeed && item.kind === "video" ? onSeed : null })));
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
  if (initial) { field.value = initial; syncPrefix(); run(initial); }
  return { dispose() { generation++; }, run };
}
