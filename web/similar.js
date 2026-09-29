/* Similar: more like one seed video. It is not a tab; the seed arrives from "More like this"
   (kind, id and title in the hash), and the router sends a seedless #/similar to Feed.
   Change opens an inline id field; a submit changes the hash, and the remount dispatches once. */
import { getJSON, isAbort } from "./api.js";
import { announce, formatNumber, partial, setState } from "./state.js";
import { gridCard } from "./cards.js";

/* Registered before the router's listener, so the navigation that opens Similar already counts:
   Back returns within the app, and a page opened cold falls back to Feed. */
let navigatedInApp = false;
window.addEventListener("hashchange", () => { navigatedInApp = true; });

export function mountSimilar(host) {
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const id = params.get("id"), kind = params.get("kind") || "video", title = params.get("title");
  let seedName = title || (kind === "video" ? "Video " : "Item ") + id;
  const view = document.createElement("div");
  view.className = "view";
  view.setAttribute("aria-label", "Similar");
  view.innerHTML = '<button type="button" class="back" data-ai-similar-back>\u2190 Back</button>' +
    '<div class="view-head"><h1>Similar</h1><p>More like one seed video: shared tags, look, voice and sound.</p></div>' +
    '<div class="seed" data-ai-similar-seed></div>' +
    '<form class="search-form seed-change" hidden><label for="ai-similar-id">Seed video id</label>' +
    '<input id="ai-similar-id" name="id" type="number" inputmode="numeric" min="1" autocomplete="off" placeholder="For example 1">' +
    '<button id="ai-similar-submit" type="submit" class="primary">Find similar</button></form>' +
    '<p class="results-head" data-ai-similar-head role="status" aria-live="polite" aria-atomic="true"></p>' +
    '<div data-ai-similar-status></div><div class="grid" data-ai-similar-grid></div>';
  host.appendChild(view);
  const form = view.querySelector("form"), field = view.querySelector("#ai-similar-id");
  const seed = view.querySelector("[data-ai-similar-seed]");
  const status = view.querySelector("[data-ai-similar-status]"), grid = view.querySelector("[data-ai-similar-grid]"), head = view.querySelector("[data-ai-similar-head]");
  let generation = 0, controller = null;

  view.querySelector("[data-ai-similar-back]").onclick = () => { if (navigatedInApp) history.back(); else location.hash = "#/feed"; };

  const picture = document.createElement("video");
  picture.muted = true; picture.preload = "metadata"; picture.tabIndex = -1; picture.setAttribute("aria-hidden", "true");
  picture.src = "/media/" + encodeURIComponent(kind) + "/" + encodeURIComponent(id) + "#t=0.5";
  const words = document.createElement("div");
  const name = document.createElement("h2"); name.textContent = seedName;
  const note = document.createElement("p"); note.className = "card-meta"; note.textContent = "Seed video, id " + id;
  words.append(name, note);
  const change = document.createElement("button"); change.type = "button"; change.textContent = "Change";
  change.setAttribute("aria-controls", "ai-similar-id");
  change.onclick = () => { form.hidden = false; field.focus(); field.select(); };
  seed.append(picture, words, change);

  /* The response names its seed; that beats the title carried in the hash. */
  function showSeed(info) {
    if (!info || typeof info !== "object") return;
    if (info.title) { seedName = info.title; name.textContent = info.title; }
    if (info.media_url && !picture.src.includes(info.media_url)) picture.src = info.media_url + "#t=0.5";
  }

  function run(seedId) {
    const current = ++generation;
    if (controller) controller.abort();
    controller = new AbortController();
    grid.textContent = ""; head.textContent = "";
    setState(status, "loading");
    getJSON("similar?" + new URLSearchParams({ kind, id: String(seedId), limit: "24" }), "items", controller.signal).then(d => {
      if (current !== generation) return;
      showSeed(d.seed);
      const items = d.items || [];
      if (!items.length) { setState(status, d.status === "partial" ? "partial" : "empty", d.status === "partial" ? () => run(seedId) : null); return; }
      status.textContent = ""; status.removeAttribute("data-ai-state");
      announce(head, formatNumber(items.length, 0) + (items.length === 1 ? " item" : " items") + " like " + seedName);
      items.forEach((item, index) => grid.appendChild(gridCard(item, { eager: index === 0 })));
      partial(status, d, () => run(seedId));
    }).catch(error => {
      if (current !== generation || isAbort(error)) return;
      const state = error.state || "unavailable";
      setState(status, state, ["unavailable", "error"].includes(state) ? () => run(seedId) : null);
    });
  }
  form.addEventListener("submit", event => {
    event.preventDefault();
    if (!field.value) { field.focus(); return; }
    const next = "#/similar?" + new URLSearchParams({ kind: "video", id: field.value });
    if (location.hash === next) run(field.value); else location.hash = next;
  });
  field.value = id;
  run(id);
  return { dispose() { generation++; if (controller) controller.abort(); }, run };
}
