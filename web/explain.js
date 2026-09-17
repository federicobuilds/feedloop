/* "Why this item" in words and bars; the raw fields stay behind a nested "Technical
   details" disclosure. An absent measurement is "not measured", never a zero bar.
   Contributor names appear only when the item carries them. */

function number(value) { return typeof value === "number" && Number.isFinite(value) ? value : null; }
function percent(multiplier) { const delta = Math.round((multiplier - 1) * 100); return (delta > 0 ? "+" : "") + delta + "%"; }

export function explanation(item) {
  const details = document.createElement("details");
  details.className = "ai-explanation";
  const summary = document.createElement("summary");
  summary.textContent = "Why this item";
  details.appendChild(summary);
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
  const signals = [["Tags you like", number(c.tag_normalized)], ["Looks like what you watch", number(c.visual_similarity)],
    ["Sounds like what you watch", number(c.sound_similarity)], ["Voice like what you watch", number(c.voice_similarity)]];
  if (c.tag_similarity != null) signals.unshift(["Shares tags with the seed", number(c.tag_similarity)]);
  if (c.query_score != null) signals.unshift(["Matches your words", number(c.query_score)]);
  const bars = section("Signals");
  signals.forEach(([label, value]) => {
    const line = document.createElement("div"); line.className = "ai-explain-bar";
    const name = document.createElement("span"); name.textContent = label;
    const track = document.createElement("span"); track.className = "ai-explain-track"; track.setAttribute("role", "img");
    const shown = document.createElement("span"); shown.className = "ai-explain-value";
    if (value === null) {
      track.setAttribute("aria-label", label + ": not measured"); shown.textContent = "not measured"; line.classList.add("is-missing");
    } else {
      const fill = document.createElement("span"); fill.style.width = Math.round(Math.max(0, Math.min(1, value)) * 100) + "%"; track.appendChild(fill);
      track.setAttribute("aria-label", label + ": " + value.toFixed(2)); shown.textContent = value.toFixed(2);
    }
    line.append(name, track, shown); bars.appendChild(line);
  });
  const adjustments = [];
  [["Contributor boost", c.affinity_multiplier], ["Watch history", c.history_multiplier], ["Watched recently", c.cooldown_multiplier]].forEach(([label, raw]) => {
    const value = number(raw);
    if (value !== null && value !== 1) adjustments.push(label + " " + percent(value));
  });
  if (number(selection.diversity_penalty) !== null && selection.diversity_penalty > 0) adjustments.push("Variety on this page \u2212" + selection.diversity_penalty.toFixed(2));
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
  details.appendChild(body);
  details.addEventListener("click", e => e.stopPropagation());
  return details;
}
