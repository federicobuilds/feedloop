/* Hash router over the five views. The address printed by the server carries the shared
   key once (#key=...); it is kept in session storage and never rendered. */
import { apiKey, getJSON } from "./api.js";
import { mountFeed } from "./feed.js";
import { mountHome } from "./home.js";
import { mountSearch } from "./search.js";
import { mountSimilar } from "./similar.js";
import { mountEngine } from "./engine.js";

apiKey();
const main = document.getElementById("main");
const note = document.getElementById("rail-note");
let current = null;

const VIEWS = {
  feed: host => mountFeed(host),
  home: host => mountHome(host),
  search: host => mountSearch(host, { onSeed: item => { location.hash = "#/similar?id=" + item.id; } }),
  similar: host => mountSimilar(host, { onSeed: item => { location.hash = "#/similar?id=" + item.id; } }),
  engine: host => mountEngine(host),
};

function route() {
  const path = (location.hash.replace(/^#\/?/, "") || "feed").split("?")[0];
  const name = VIEWS[path] ? path : "feed";
  if (current) { current.handle.dispose(); current = null; }
  main.textContent = "";
  document.querySelectorAll(".rail a[data-view]").forEach(a => { if (a.dataset.view === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  current = { name, handle: VIEWS[name](main) };
  main.setAttribute("data-ai-view", name);
}

window.addEventListener("hashchange", route);
route();
getJSON("config").then(c => {
  note.textContent = (c.attribution && c.attribution.policy_revision && c.attribution.policy_revision.startsWith("demo") ? "Demonstration policy: " + c.attribution.window_s + " s attribution window. " : "") +
    (c.encoder ? "Text search encoder: " + Object.values(c.space_meta || {}).map(m => m && m.encoder).filter(Boolean).join(", ") : "No text encoder installed; Search reports no-feature.");
}).catch(() => { note.textContent = "Engine configuration unavailable."; });
window.feedloop = { route, get current() { return current; } };
