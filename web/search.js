/* Search: text over the semantic and sound spaces. The mode is a segmented Look / Sound /
   Both control kept in sync with a leading "sound:" or "both:" in the words, in both
   directions, through one normalizer; the request carries the bare words and the mode.
   A new search cancels the one in flight. Uses the ordinary status region, never the Feed
   building panel; an empty result, a partial result and a missing feature stay distinct. */
import { getJSON, isAbort } from "./api.js";
import { announce, formatNumber, partial, setState } from "./state.js";
import { gridCard, stopVideos } from "./cards.js";

const MODE_WORDS = { look: "look", sound: "sound", both: "look and sound" };

export function mountSearch(host) {
  const view = document.createElement("div");
  view.className = "view";
  view.setAttribute("aria-label", "Search");
  view.innerHTML = '<div class="view-head"><h1>Search</h1><p>Describe a look or a sound in your own words.</p></div>' +
    '<form class="search-form" role="search"><label class="sr-only" for="ai-search-query">Search words</label>' +
    '<input id="ai-search-query" name="q" type="search" autocomplete="off" spellcheck="false" placeholder="Describe a look, or start with sound: for a sound\u2026">' +
    '<fieldset id="ai-search-mode" class="segmented"><legend class="sr-only">Search in</legend>' +
    '<label data-mode="look"><input type="radio" name="mode" value="look" checked><span>Look</span></label>' +
    '<label data-mode="sound"><input type="radio" name="mode" value="sound" data-prefix="sound:"><span>Sound</span></label>' +
    '<label data-mode="both"><input type="radio" name="mode" value="both" data-prefix="both:"><span>Both</span></label></fieldset>' +
    '<button id="ai-search-submit" type="submit" class="primary">Search</button></form>' +
    '<p class="form-hint">Typing <kbd>sound:</kbd> or <kbd>both:</kbd> first switches the mode; deleting it returns to Look.</p>' +
    '<p class="results-head" data-ai-search-head role="status" aria-live="polite" aria-atomic="true"></p>' +
    '<div data-ai-search-status></div><div class="grid" data-ai-search-grid></div>';
  host.appendChild(view);
  const form = view.querySelector("form"), field = view.querySelector("#ai-search-query"), submit = view.querySelector("#ai-search-submit");
  const status = view.querySelector("[data-ai-search-status]"), grid = view.querySelector("[data-ai-search-grid]"), head = view.querySelector("[data-ai-search-head]");
  const radios = Array.from(view.querySelectorAll("#ai-search-mode input"));
  let generation = 0, controller = null;

  const mode = () => form.elements.mode.value;
  function setMode(value) { radios.forEach(radio => { radio.checked = radio.value === value; }); }

  /* The one normalizer: a leading prefix becomes the mode and leaves the words. */
  function normalize(text, current) {
    const trimmed = text.trimStart(), low = trimmed.toLowerCase();
    const hit = radios.find(radio => radio.dataset.prefix && low.startsWith(radio.dataset.prefix));
    return hit ? { mode: hit.value, words: trimmed.slice(hit.dataset.prefix.length).trimStart() } : { mode: current, words: text };
  }
  /* The field and the control mirror each other: a typed prefix selects its mode, deleting
     a prefix returns to Look, and picking a mode rewrites the prefix. `prefixed` tracks
     whether the field last carried a prefix, so a mode picked on an empty field survives
     the first typed words. */
  let prefixed = false;
  function prefixFor(value) { const radio = radios.find(r => r.value === value); return radio && radio.dataset.prefix ? radio.dataset.prefix + " " : ""; }
  field.addEventListener("input", () => {
    const next = normalize(field.value, mode());
    const has = next.words !== field.value;
    if (has) setMode(next.mode); else if (prefixed) setMode("look");
    prefixed = has;
  });
  radios.forEach(radio => radio.addEventListener("change", () => {
    if (!radio.checked) return;
    const before = field.value, words = normalize(before, radio.value).words;
    const after = words ? prefixFor(radio.value) + words : "";
    if (after === before) return;
    const caret = field.selectionStart == null ? before.length : field.selectionStart;
    field.value = after;
    prefixed = after !== words;
    const place = Math.max(prefixFor(radio.value).length, Math.min(after.length, caret + after.length - before.length));
    field.setSelectionRange(place, place);
  }));

  function pending(busy) {
    submit.setAttribute("aria-disabled", String(busy));
    submit.textContent = busy ? "Searching\u2026" : "Search";
    grid.setAttribute("aria-busy", String(busy));
  }

  /* Actions under an empty or unset-up result change the request: edit the words or
     search another mode. A same-query Retry would only repeat the miss. */
  function offer(q, m, text, diagnostic, blocked = []) {
    const box = document.createElement("div");
    box.className = "search-offer"; box.setAttribute("data-ai-search-offer", "1");
    if (diagnostic) box.appendChild(Object.assign(document.createElement("p"), { className: "diagnostic", textContent: diagnostic }));
    if (text) box.appendChild(Object.assign(document.createElement("p"), { textContent: text }));
    const actions = document.createElement("div"); actions.className = "feedback-actions";
    const edit = Object.assign(document.createElement("button"), { type: "button", textContent: "Edit words" });
    edit.onclick = () => { field.focus(); field.select(); };
    actions.appendChild(edit);
    radios.filter(radio => radio.value !== m && !(blocked.length && (radio.value === "both" || blocked.includes(radio.value)))).forEach(radio => {
      const other = Object.assign(document.createElement("button"), { type: "button", textContent: "Search " + MODE_WORDS[radio.value] + " instead" });
      other.onclick = () => { field.value = q; setMode(radio.value); form.requestSubmit(); };
      actions.appendChild(other);
    });
    box.appendChild(actions);
    status.appendChild(box);
  }

  function run(q, m) {
    const current = ++generation;
    if (controller) controller.abort();
    grid.textContent = ""; head.textContent = "";
    const stale = status.querySelector("[data-ai-search-offer]"); if (stale) stale.remove();
    if (!q.trim()) { announce(head, "Type a few words to search."); field.focus(); return; }
    controller = new AbortController();
    pending(true);
    setState(status, "loading");
    getJSON("search?" + new URLSearchParams({ q, mode: m, limit: "24" }), "items", controller.signal).then(d => {
      if (current !== generation) return;
      const items = d.items || [];
      if (!items.length && d.status === "partial") { setState(status, "partial", () => run(q, m)); return; }
      if (!items.length) {
        setState(status, "empty", null, "Nothing matched \u201C" + q + "\u201D in " + MODE_WORDS[m] + ".");
        offer(q, m, "Try fewer or different words, or search another mode.");
        return;
      }
      status.textContent = ""; status.removeAttribute("data-ai-state");
      announce(head, formatNumber(items.length, 0) + (items.length === 1 ? " match" : " matches") + " for \u201C" + q + "\u201D in " + MODE_WORDS[m]);
      items.forEach((item, index) => grid.appendChild(gridCard(item, { eager: index === 0 })));
      partial(status, d, () => run(q, m));
    }).catch(error => {
      if (current !== generation || isAbort(error)) return;
      if (error.state === "no-feature") {
        const missing = Object.entries(error.components || {}).filter(([, c]) => (c && c.status) === "no-feature").map(([name]) => name);
        const names = missing.length ? missing : m === "both" ? ["look", "sound"] : [m];
        const what = names.map(name => MODE_WORDS[name] || name).join(" and ");
        setState(status, "no-feature", null, what.charAt(0).toUpperCase() + what.slice(1) + " search " + (names.length > 1 ? "aren't" : "isn't") + " set up for this library.");
        offer(q, m, null, "Diagnostic: no text encoder or no matching feature space for " + names.join(", ") + " (no-feature).", names);
        return;
      }
      const state = error.state || "unavailable";
      setState(status, state, ["unavailable", "error"].includes(state) ? () => run(q, m) : null);
    }).finally(() => { if (current === generation) pending(false); });
  }
  /* The hash is the route state (q and mode); a submit that changes it remounts the view,
     which dispatches once from that state. An unchanged hash dispatches directly. */
  form.addEventListener("submit", event => {
    event.preventDefault();
    if (submit.getAttribute("aria-disabled") === "true") return;
    const words = normalize(field.value, mode());
    setMode(words.mode);
    const next = "#/search?" + new URLSearchParams({ q: words.words, mode: words.mode }).toString();
    if (location.hash === next) run(words.words, words.mode); else location.hash = next;
  });
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const initial = normalize(params.get("q") || "", MODE_WORDS[params.get("mode")] ? params.get("mode") : "look");
  setMode(initial.mode);
  if (initial.words) {
    field.value = prefixFor(initial.mode) + initial.words; prefixed = initial.mode !== "look";
    run(initial.words, initial.mode);
  }
  return { dispose() { generation++; stopVideos(host); if (controller) controller.abort(); }, run };
}
