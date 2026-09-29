/* Hash router over the five views. The address printed by the server carries the shared
   key once (#key=...); it is kept in session storage and never rendered. Each route sets
   the document title; a navigation to another view moves focus to its heading. */
import { apiKey, config } from "./api.js";
import { icon } from "./icons.js";
import { mountFeed } from "./feed.js";
import { mountHome } from "./home.js";
import { mountSearch } from "./search.js";
import { mountSimilar } from "./similar.js";
import { mountEngine } from "./engine.js";

apiKey();
const main = document.getElementById("main");
let current = null;

const VIEWS = { feed: mountFeed, home: mountHome, search: mountSearch, similar: mountSimilar, engine: mountEngine };
const TITLES = { feed: "Feed", home: "Home", search: "Search", similar: "Similar", engine: "Engine" };

document.querySelectorAll(".tabs a[data-view]").forEach(a => a.prepend(icon(a.dataset.view)));

function route(event) {
  const path = (location.hash.replace(/^#\/?/, "") || "feed").split("?")[0];
  const name = VIEWS[path] ? path : "feed";
  const sameView = current && current.name === name;
  const focusedId = sameView && document.activeElement && main.contains(document.activeElement) ? document.activeElement.id : "";
  if (current) { current.handle.dispose(); current = null; }
  main.textContent = "";
  document.querySelectorAll(".tabs a[data-view]").forEach(a => { if (a.dataset.view === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  current = { name, handle: VIEWS[name](main) };
  main.setAttribute("data-ai-view", name);
  document.title = TITLES[name] + " \u00B7 feedloop";
  // A remount of the same view (new query in the hash) keeps the control that was in use.
  const keep = focusedId && document.getElementById(focusedId);
  if (keep) keep.focus({ preventScroll: true });
  else if (event && !sameView) {
    const heading = main.querySelector("h1");
    if (heading) { heading.tabIndex = -1; heading.focus({ preventScroll: true }); }
  }
}

window.addEventListener("hashchange", route);
route();
config().then(c => {
  const a = c.attribution || {};
  if (!(a.policy_revision && a.policy_revision.startsWith("demo"))) return;
  const badge = document.getElementById("demo-badge");
  const text = "Demonstration policy: " + a.window_s + " s attribution window";
  badge.title = text;
  badge.querySelector(".sr-only").textContent = ". " + text;
  badge.hidden = false;
}).catch(() => {});
window.feedloop = { route, get current() { return current; } };
