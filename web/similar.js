/* Similar: more like one seed video. The seed arrives from "More like this" (kind, id and
   title in the hash); with no seed the page explains where to pick one. Entering an id by
   hand sits under "Advanced". A submit changes the hash, and the remount dispatches once. */
import { getJSON, isAbort } from "./api.js";
import { announce, formatNumber, partial, setState } from "./state.js";
import { gridCard } from "./cards.js";

export function mountSimilar(host) {
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const id = params.get("id"), kind = params.get("kind") || "video", title = params.get("title");
  const seedName = title || (kind === "video" ? "Video " : "Item ") + id;
  const view = document.createElement("div");
  view.className = "view";
  view.setAttribute("aria-label", "Similar");
  view.innerHTML = '<div class="view-head"><h1>Similar</h1><p>More like one seed video: shared tags, look, voice and sound.</p></div>' +
    '<div data-ai-similar-seed></div>' +
    '<details class="advanced" data-ai-similar-advanced><summary>Advanced</summary>' +
    '<form class="search-form"><label for="ai-similar-id">Seed video id</label>' +
    '<input id="ai-similar-id" name="id" type="number" inputmode="numeric" min="1" autocomplete="off" placeholder="For example 1">' +
    '<button id="ai-similar-submit" type="submit" class="primary">Find similar</button></form></details>' +
    '<p class="results-head" data-ai-similar-head role="status" aria-live="polite" aria-atomic="true"></p>' +
    '<div data-ai-similar-status></div><div class="grid" data-ai-similar-grid></div>';
  host.appendChild(view);
  const form = view.querySelector("form"), field = view.querySelector("#ai-similar-id"), advanced = view.querySelector("[data-ai-similar-advanced]");
  const seed = view.querySelector("[data-ai-similar-seed]");
  const status = view.querySelector("[data-ai-similar-status]"), grid = view.querySelector("[data-ai-similar-grid]"), head = view.querySelector("[data-ai-similar-head]");
  let generation = 0, controller = null;

  if (id) {
    seed.className = "seed";
    const picture = document.createElement("video");
    picture.muted = true; picture.preload = "metadata"; picture.tabIndex = -1; picture.setAttribute("aria-hidden", "true");
    picture.src = "/media/" + encodeURIComponent(kind) + "/" + encodeURIComponent(id) + "#t=0.5";
    const words = document.createElement("div");
    const name = document.createElement("h2"); name.textContent = seedName;
    const note = document.createElement("p"); note.className = "card-meta"; note.textContent = "Seed video, id " + id;
    words.append(name, note);
    const change = document.createElement("button"); change.type = "button"; change.textContent = "Change";
    change.onclick = () => { advanced.open = true; field.focus(); };
    seed.append(picture, words, change);
  } else {
    seed.className = "empty-guide";
    seed.innerHTML = '<h2>Pick a seed first</h2><p>Open a video in <a href="#/home">Home</a> or <a href="#/search">Search</a> and choose ' +
      "More like this. The results appear here.</p>";
  }

  function run(seedId) {
    const current = ++generation;
    if (controller) controller.abort();
    controller = new AbortController();
    grid.textContent = ""; head.textContent = "";
    setState(status, "loading");
    getJSON("similar?" + new URLSearchParams({ kind, id: String(seedId), limit: "24" }), "items", controller.signal).then(d => {
      if (current !== generation) return;
      const items = d.items || [];
      if (!items.length) { setState(status, d.status === "partial" ? "partial" : "empty", () => run(seedId)); return; }
      status.textContent = ""; status.removeAttribute("data-ai-state");
      announce(head, formatNumber(items.length, 0) + (items.length === 1 ? " item" : " items") + " like " + seedName);
      items.forEach((item, index) => grid.appendChild(gridCard(item, { eager: index === 0 })));
      partial(status, d, () => run(seedId));
    }).catch(error => {
      if (current !== generation || isAbort(error)) return;
      setState(status, error.state || "unavailable", () => run(seedId));
    });
  }
  form.addEventListener("submit", event => {
    event.preventDefault();
    if (!field.value) { field.focus(); return; }
    const next = "#/similar?" + new URLSearchParams({ kind: "video", id: field.value });
    if (location.hash === next) run(field.value); else location.hash = next;
  });
  if (id) { field.value = id; run(id); }
  return { dispose() { generation++; if (controller) controller.abort(); }, run };
}
