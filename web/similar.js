/* Similar: more like one primary item. The seed comes from Search or from an id typed here. */
import { getJSON } from "./api.js";
import { partial, setState } from "./state.js";
import { gridCard } from "./cards.js";

export function mountSimilar(host, { onSeed } = {}) {
  const view = document.createElement("div");
  view.className = "view";
  view.setAttribute("aria-label", "Similar");
  view.innerHTML = '<div class="view-head"><h1>Similar</h1><p>More like one seed video: shared tags, look, voice and sound.</p></div>' +
    '<form class="search-form"><label class="sr-only" for="ai-similar-id">Seed video id</label>' +
    '<input id="ai-similar-id" name="id" type="number" inputmode="numeric" min="1" autocomplete="off" placeholder="video id, for example 1">' +
    '<button type="submit" class="primary">Find similar</button></form><p class="results-head" data-ai-similar-head hidden></p>' +
    '<div data-ai-similar-status></div><div class="grid" data-ai-similar-grid></div>';
  host.appendChild(view);
  const form = view.querySelector("form"), field = view.querySelector("#ai-similar-id");
  const status = view.querySelector("[data-ai-similar-status]"), grid = view.querySelector("[data-ai-similar-grid]"), head = view.querySelector("[data-ai-similar-head]");
  let generation = 0;

  function run(id, kind = "video") {
    const current = ++generation;
    grid.textContent = ""; head.hidden = true;
    if (!id) { setState(status, "empty"); return; }
    setState(status, "loading");
    getJSON("similar?" + new URLSearchParams({ kind, id: String(id), limit: "24" }), "items").then(d => {
      if (current !== generation) return;
      const items = d.items || [];
      if (!items.length) { setState(status, d.status === "partial" ? "partial" : "empty", () => run(id, kind)); return; }
      status.textContent = ""; status.removeAttribute("data-ai-state");
      head.hidden = false; head.textContent = items.length + " items like " + kind + " " + id;
      items.forEach(item => grid.appendChild(gridCard(item, { seedAction: onSeed && item.kind === "video" ? onSeed : null })));
      partial(status, d, () => run(id, kind));
    }).catch(error => {
      if (current !== generation) return;
      setState(status, error.state || "unavailable", () => run(id, kind));
    });
  }
  form.addEventListener("submit", event => { event.preventDefault(); run(field.value); location.hash = "#/similar?id=" + encodeURIComponent(field.value); });
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const initial = params.get("id");
  if (initial) { field.value = initial; run(initial, params.get("kind") || "video"); }
  return { dispose() { generation++; }, run };
}
