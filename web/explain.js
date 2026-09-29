/* "Why this pick" in words and bars; the raw fields stay behind a nested "Technical
   details" disclosure. An absent measurement is "not measured", never a zero bar.
   Contributor names appear only when the item carries them. The body is built on first
   open, so a page of cards does not pay for explanations nobody reads. The evidence strip
   is the glanceable form of the same signals and opens this disclosure. Below 640 px the
   body opens in a full-height sheet instead of inline, so it never lands off screen. */
import { formatNumber } from "./state.js";
import { icon } from "./icons.js";

function number(value) { return typeof value === "number" && Number.isFinite(value) ? value : null; }
function percent(multiplier) { const delta = Math.round((multiplier - 1) * 100); return (delta > 0 ? "+" : "") + delta + "%"; }
const narrow = matchMedia("(max-width: 640px)");

/* The wording follows the surface: Feed and Home compare with what you watch, Similar with
   the seed item, Search with your words. Strip, legend and breakdown all read from here. */
const LABELS = {
  feed: { tags: "Tags you like", look: "Looks like what you watch", sound: "Sounds like what you watch", voice: "Voice like what you watch" },
  similar: { tags: "Shares tags with this item", look: "Looks like this item", sound: "Sounds like this item", voice: "Voice like this item" },
  search: { query: "Matches your words", tags: "Tag match", look: "Look match", sound: "Sound match", voice: "Voice match" },
};
const SHORT = { query: "Words", tags: "Tags", look: "Look", sound: "Sound", voice: "Voice" };

function signalsOf(item) {
  const c = item.explanation || item.similar || item.search || {};
  const surface = item.explanation ? "feed" : item.similar ? "similar" : item.search ? "search" : "feed";
  const tags = surface === "feed" ? c.tag_normalized : c.tag_similarity != null ? c.tag_similarity : c.tag_normalized;
  const values = { query: number(c.query_score), tags: number(tags), look: number(c.visual_similarity), sound: number(c.sound_similarity), voice: number(c.voice_similarity) };
  return Object.keys(LABELS[surface]).map(key => ({ key, short: SHORT[key], label: LABELS[surface][key], value: values[key] }));
}

/* The label of the strongest measured signal, so a card's one-line reason names the same
   signal the strip shows fullest. */
export function strongestSignal(item) {
  const measured = signalsOf(item).filter(s => s.value !== null && s.value > 0);
  return measured.length ? measured.reduce((a, b) => (b.value > a.value ? b : a)).label : null;
}

function valueText(value) { return value === null ? "not measured" : formatNumber(value, 2, true); }
function fillOf(value) { return Math.round(Math.max(0, Math.min(1, value)) * 100) + "%"; }

/* One equal slot per signal, filled to its own magnitude: zero is an empty slot, not
   measured is hatched. With nothing measured at all the strip is a quiet line instead. */
export function evidenceStrip(item, details, { legend = false } = {}) {
  const signals = signalsOf(item);
  if (signals.every(s => s.value === null)) {
    const none = document.createElement("p");
    none.className = "evidence-none";
    none.textContent = "No measurements yet";
    return none;
  }
  const words = signals.map(s => s.label + " " + valueText(s.value)).join(", ");
  const strip = document.createElement("button");
  strip.type = "button";
  strip.className = "evidence";
  strip.setAttribute("aria-label", "Evidence: " + words + ". Show why this pick");
  strip.title = words;
  signals.forEach(s => {
    const slot = document.createElement("span");
    slot.className = "evidence-slot is-" + s.key + (s.value === null ? " is-missing" : "");
    const name = document.createElement("span");
    name.className = "evidence-name";
    name.textContent = legend ? s.short + " " + (s.value === null ? "\u2013" : formatNumber(s.value, 2, true)) : s.short;
    const track = document.createElement("span");
    track.className = "evidence-track";
    if (s.value !== null) { const fill = document.createElement("span"); fill.style.width = fillOf(s.value); track.appendChild(fill); }
    slot.append(track, name);
    strip.appendChild(slot);
  });
  strip.addEventListener("click", event => {
    event.stopPropagation();
    details.__opener = strip;
    details.open = true;
    if (narrow.matches) return;
    details.querySelector("summary").focus({ preventScroll: true });
    details.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  });
  return strip;
}

/* One shared modal sheet for narrow screens. It lives on <body>, outside the card's
   containment, so it can cover the viewport; Esc closes it through the dialog. */
let sheet = null, sheetOwner = null;
function openSheet(details) {
  if (!sheet) {
    sheet = document.createElement("dialog");
    sheet.className = "sheet";
    sheet.setAttribute("aria-labelledby", "ai-sheet-title");
    const head = document.createElement("div"); head.className = "sheet-head";
    const title = document.createElement("h2"); title.id = "ai-sheet-title"; title.textContent = "Why this pick";
    const close = document.createElement("button"); close.type = "button"; close.className = "sheet-close";
    close.append(icon("close"), "Close");
    close.onclick = () => sheet.close();
    head.append(title, close);
    const scroll = document.createElement("div"); scroll.className = "sheet-body";
    sheet.append(head, scroll);
    sheet.addEventListener("close", () => {
      const owner = sheetOwner; sheetOwner = null;
      if (!owner) return;
      owner.details.open = false;
      const back = owner.opener && owner.opener.isConnected ? owner.opener : owner.details.querySelector("summary");
      if (back && back.isConnected) back.focus({ preventScroll: true });
    });
    document.body.appendChild(sheet);
  }
  sheetOwner = { details, opener: details.__opener || details.querySelector("summary") };
  details.__opener = null;
  sheet.querySelector(".sheet-body").replaceChildren(details.__body);
  sheet.querySelector(".sheet-body").scrollTop = 0;
  sheet.showModal();
  sheet.querySelector(".sheet-close").focus();
}

export function explanation(item) {
  const details = document.createElement("details");
  details.className = "ai-explanation";
  const summary = document.createElement("summary");
  summary.textContent = "Why this pick";
  details.appendChild(summary);
  details.addEventListener("toggle", () => {
    if (!details.open) return;
    if (!details.__body) details.__body = explanationBody(item);
    if (narrow.matches) openSheet(details);
    else details.appendChild(details.__body);
  });
  details.addEventListener("click", e => e.stopPropagation());
  return details;
}

function explanationBody(item) {
  const c = item.explanation || item.similar || item.search || {};
  const selection = c.selection && typeof c.selection === "object" ? c.selection : {};
  const body = document.createElement("div");
  body.className = "ai-explain-body";
  function section(title) {
    const block = document.createElement("div"); block.className = "ai-explain-section";
    const head = document.createElement("strong"); head.textContent = title; block.appendChild(head);
    body.appendChild(block); return block;
  }
  if (item.reason) {
    const lead = document.createElement("p"); lead.className = "ai-explain-lead"; lead.textContent = item.reason; body.appendChild(lead);
  }
  const bars = section("Signals");
  signalsOf(item).forEach(({ label, value }) => {
    const line = document.createElement("div"); line.className = "ai-explain-bar";
    const name = document.createElement("span"); name.textContent = label;
    const track = document.createElement("span"); track.className = "ai-explain-track"; track.setAttribute("role", "img");
    const shown = document.createElement("span"); shown.className = "ai-explain-value";
    if (value === null) {
      track.setAttribute("aria-label", label + ": not measured"); shown.textContent = "not measured"; line.classList.add("is-missing");
    } else {
      const fill = document.createElement("span"); fill.style.width = fillOf(value); track.appendChild(fill);
      track.setAttribute("aria-label", label + ": " + valueText(value)); shown.textContent = valueText(value);
    }
    line.append(name, track, shown); bars.appendChild(line);
  });
  const adjustments = [];
  [["Contributor boost", c.affinity_multiplier], ["Watch history", c.history_multiplier], ["Watched recently", c.cooldown_multiplier]].forEach(([label, raw]) => {
    const value = number(raw);
    if (value !== null && value !== 1) adjustments.push(label + " " + percent(value));
  });
  if (number(selection.diversity_penalty) !== null && selection.diversity_penalty > 0) adjustments.push("Variety on this page \u2212" + formatNumber(selection.diversity_penalty, 2, true));
  if (adjustments.length) {
    const list = document.createElement("ul"); list.className = "ai-explain-list";
    adjustments.forEach(text => { const entry = document.createElement("li"); entry.textContent = text; list.appendChild(entry); });
    section("Adjustments").appendChild(list);
  }
  const contributions = Array.isArray(c.tag_contributions) ? c.tag_contributions.slice() : Array.isArray(c.contributors) ? c.contributors.slice() : [];
  if (contributions.length) {
    contributions.sort((a, b) => Math.abs(number(b.contribution) || 0) - Math.abs(number(a.contribution) || 0));
    const chips = section("Because of these tags");
    const names = item.tag_names || {};
    contributions.slice(0, 6).forEach(row => {
      const chip = document.createElement("span");
      const contribution = number(row.contribution) || 0;
      chip.className = "ai-explain-chip " + (contribution < 0 ? "is-negative" : "is-positive");
      chip.textContent = (contribution < 0 ? "\u2212 " : "+ ") + (names[String(row.tag_id)] || row.name || ("tag #" + row.tag_id)).replace(/_/g, " ").toLowerCase();
      chips.appendChild(chip);
    });
  }
  const people = Array.isArray(item.contributor_ids) ? item.contributor_ids.filter(Boolean) : [];
  if (people.length) {
    const who = section("Contributors"); const line = document.createElement("span"); line.textContent = people.slice(0, 3).join(", "); who.appendChild(line);
  }
  const sourceLabels = { tags: "Tag match", visual: "Visual match", voice: "Voice match", sound: "Sound match", images: "Secondary lane",
    explore: "Exploration pick", control: "Control pick", fallback: "Fallback pick" };
  const sources = Array.isArray(c.sources) ? c.sources : [];
  if (sources.length) {
    const routes = section("Found through");
    sources.forEach(name => { const chip = document.createElement("span"); chip.className = "ai-explain-chip"; chip.textContent = sourceLabels[name] || String(name); routes.appendChild(chip); });
  }
  const placement = [];
  if (c.explore === true || item.explore === true) placement.push("Exploration pick: chosen from outside your usual to test your taste");
  if (c.control === true || item.control === true) placement.push("Control pick: shown without personalization to measure the engine");
  if (number(c.position) !== null) placement.push("Ranked #" + (c.position + 1) + " in this generation");
  if (c.arm) placement.push("Experiment arm " + c.arm);
  if (c.fallback) placement.push("Fallback: " + String(c.fallback).replace(/_/g, " "));
  if (placement.length) {
    const how = section("Placement"); const lines = document.createElement("ul"); lines.className = "ai-explain-list";
    placement.forEach(text => { const entry = document.createElement("li"); entry.textContent = text; lines.appendChild(entry); });
    how.appendChild(lines);
  }
  const generation = item.provenance || {};
  const footer = document.createElement("p"); footer.className = "ai-explain-footer";
  footer.textContent = "Ranker " + (generation.ranking_revision || item.ranking_revision || "unavailable") + " \u00B7 features " +
    (generation.revisions && generation.revisions.features ? String(generation.revisions.features).slice(0, 12) : "unavailable") +
    (number(generation.captured_at) !== null ? " \u00B7 ranked " + new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "medium" }).format(new Date(generation.captured_at * 1000)) : "");
  body.appendChild(footer);
  const raw = document.createElement("details");
  raw.className = "ai-explanation-raw";
  const rawSummary = document.createElement("summary"); rawSummary.textContent = "Technical details"; raw.appendChild(rawSummary);
  const fields = item.explanation ? ["explanation", "provenance", "category", "score"] : item.similar ? ["similar", "provenance", "score"] : ["search", "score"];
  fields.forEach(field => {
    const row = document.createElement("div"), label = document.createElement("strong"), value = document.createElement("pre");
    label.textContent = field.replace(/_/g, " ") + ": ";
    const data = item[field];
    value.textContent = data == null ? "Unavailable" : typeof data === "string" ? data : JSON.stringify(data, null, 2);
    row.append(label, value); raw.appendChild(row);
  });
  body.appendChild(raw);
  return body;
}
