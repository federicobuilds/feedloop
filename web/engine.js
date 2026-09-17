/* Engine panel: the real scorecard, evidence gates, the current experiment, and the
   reversible tuner ledger. Missing counts stay missing; demonstration evidence is
   labelled as such; every control is confirmed by the server before the panel refreshes. */
import { classify, getJSON, post } from "./api.js";
import { announce, setState, whenTime } from "./state.js";

function esc(value) { return value == null ? "Unavailable" : String(value); }
function num(value, digits = 3) { return typeof value === "number" && Number.isFinite(value) ? new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value) : "Unavailable"; }

function armWords(label, arm) {
  if (!arm || typeof arm !== "object") return null;
  const bits = [label];
  if (arm.trials != null) bits.push(arm.trials + " trials");
  if (arm.successes != null) bits.push(arm.successes + " liked");
  if (typeof arm.mean_reward === "number") bits.push("mean reward " + num(arm.mean_reward, 4));
  if (typeof arm.like_rate === "number") bits.push(Math.round(arm.like_rate * 100) + "% like rate");
  return bits.join(", ");
}

function evidenceWords(ev) {
  if (!ev || typeof ev !== "object") return "Unavailable";
  const parts = [armWords("base", ev.base), armWords("candidate", ev.cand || ev.candidate)].filter(Boolean);
  if (typeof ev.diff === "number") parts.push("difference " + (ev.diff > 0 ? "+" : "") + num(ev.diff, 4) + (typeof ev.ci_halfwidth === "number" ? " \u00B1 " + num(ev.ci_halfwidth, 4) : ""));
  if (ev.reason) parts.push(String(ev.reason).replace(/_/g, " ") + (ev.next ? ", next knob " + ev.next : ""));
  if (ev.reverted_ledger_id != null) parts.push("undo of move " + ev.reverted_ledger_id);
  if (ev.status && !parts.length) parts.push(String(ev.status).replace(/_/g, " "));
  return parts.length ? parts.join(" \u00B7 ") : JSON.stringify(ev);
}

function cell(text, cls) { const td = document.createElement("td"); if (cls) td.className = cls; td.textContent = text; return td; }

export function renderScorecard(sc, config) {
  const t = sc.tuner || {}, auto = sc.automation || {}, evidence = sc.evidence || {}, capture = sc.capture || {}, gate = sc.gate || {};
  const root = document.createElement("div");
  const captureVerified = capture.verified === true;
  const captureBox = document.createElement("div");
  captureBox.className = "gate" + (captureVerified ? " ok" : "");
  captureBox.setAttribute("data-ai-capture-status", captureVerified ? "enabled" : "unavailable");
  captureBox.textContent = captureVerified
    ? "Watch capture verified: " + num(capture.captured_outcomes, 0) + " committed watch outcomes from " + num((capture.receipts || {}).imported || 0, 0) + " imported batches" +
      (capture.last_imported_at ? ", last at " + whenTime(capture.last_imported_at) : "") + "; " + num(capture.attributed_outcomes || 0, 0) + " outcomes attributed" +
      (capture.attribution_through_ts ? " through " + whenTime(capture.attribution_through_ts) : "") + "."
    : "Watch capture readiness unavailable: no committed watch batch has produced an outcome yet. Player configuration alone is not proof.";
  const promotion = document.createElement("div");
  promotion.className = "gate";
  promotion.setAttribute("data-ai-promotion", auto.enabled === true ? "automatic" : "disabled");
  promotion.textContent = auto.enabled === true
    ? "Automatic tuning is on: the engine re-evaluates every " + esc(auto.interval_hours) + " h and promotes a knob by itself once each arm clears " + esc(gate.min_trials_per_arm) +
      " ripened trials across " + esc(auto.min_sessions_per_arm) + " sessions with a significant, diversity-safe win. Trials ripen " + esc(gate.ripen_hours) + " h after attribution."
    : "Automatic promotion is off. Decisions require an explicit tick.";
  const reasons = document.createElement("div"); reasons.className = "aid-note";
  reasons.setAttribute("data-ai-evidence", evidence.valid === true ? "valid" : "invalid");
  reasons.textContent = "Evidence " + (evidence.valid === true ? "valid" : "not yet valid") + ": " + ((evidence.validity_reasons || []).concat(evidence.promotion_reasons || []).join(", ") || "no reasons reported");
  if (config && config.attribution && config.attribution.policy_revision && config.attribution.policy_revision.startsWith("demo")) {
    const demo = document.createElement("span"); demo.className = "aid-chip demo"; demo.setAttribute("data-ai-demo-label", "1");
    demo.textContent = "Demonstration policy: " + config.attribution.window_s + " s attribution window, " + config.attribution.policy_revision;
    reasons.appendChild(document.createTextNode(" ")); reasons.appendChild(demo);
  }
  root.append(captureBox, promotion, reasons);

  const experiment = document.createElement("div");
  experiment.setAttribute("data-ai-experiment", t.knob || "none");
  experiment.className = "aid-note";
  experiment.textContent = t.knob ? "Current experiment: " + t.knob + " base " + esc(t.base) + " versus candidate " + esc(t.candidate) +
    (t.experiment_started ? ", since " + whenTime(t.experiment_started) : "") + "; stalls " + esc(t.stalls) + "; promotion eligible now: " + (t.promotion_eligible === true ? "yes" : "no") + "."
    : "No active experiment: arms are created by an explicit tuner initialization or reset.";
  root.appendChild(experiment);

  const knobs = document.createElement("div"); knobs.className = "aid-scroll";
  const table = document.createElement("table"); table.className = "aid-tbl";
  table.innerHTML = "<thead><tr><th>Knob</th><th class=num>Default</th><th class=num>Settled</th><th class=num>Base</th><th class=num>Candidate</th></tr></thead>";
  const tbody = document.createElement("tbody");
  (t.knobs || []).forEach(knob => {
    const tr = document.createElement("tr");
    tr.append(cell(esc(knob.knob)), cell(num(knob.default), "num"), cell(num(knob.settled), "num"), cell(num(knob.base), "num"), cell(num(knob.candidate), "num"));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody); knobs.appendChild(table); root.appendChild(knobs);
  if (t.arms && Object.keys(t.arms).length) { const arms = document.createElement("div"); arms.className = "aid-note"; arms.textContent = "Attributed trials: " + evidenceWords(t.arms); root.appendChild(arms); }

  const ledgerHead = document.createElement("h3"); ledgerHead.textContent = "Tuner ledger"; root.appendChild(ledgerHead);
  if ((t.ledger || []).length) {
    const scroll = document.createElement("div"); scroll.className = "aid-scroll";
    const ledger = document.createElement("table"); ledger.className = "aid-tbl"; ledger.setAttribute("data-ai-ledger", "1");
    ledger.innerHTML = "<thead><tr><th>When</th><th>Knob</th><th>Move</th><th>Status</th><th>Evidence</th><th></th></tr></thead>";
    const rows = document.createElement("tbody");
    t.ledger.forEach(l => {
      const tr = document.createElement("tr"); tr.setAttribute("data-ai-ledger-row", String(l.id)); tr.setAttribute("data-ai-ledger-status", esc(l.status));
      tr.append(cell(whenTime(l.ts)), cell(esc(l.knob)), cell(num(l.old) + " \u2192 " + num(l.new), "num"));
      const status = document.createElement("td"); const chip = document.createElement("span"); chip.className = "aid-chip" + (l.status === "applied" ? " warn" : ""); chip.textContent = esc(l.status); status.appendChild(chip); tr.appendChild(status);
      const ev = l.evidence || {};
      const evidenceCell = cell(evidenceWords(ev));
      if (ev.demonstration === true) { const label = document.createElement("span"); label.className = "aid-chip demo"; label.textContent = " demonstration evidence"; evidenceCell.appendChild(label); }
      tr.appendChild(evidenceCell);
      const action = document.createElement("td");
      if (l.status === "applied") { const undo = document.createElement("button"); undo.type = "button"; undo.className = "aid-revert"; undo.setAttribute("data-id", String(l.id)); undo.textContent = "Undo move"; action.appendChild(undo); }
      tr.appendChild(action); rows.appendChild(tr);
    });
    ledger.appendChild(rows); scroll.appendChild(ledger); root.appendChild(scroll);
  } else {
    const none = document.createElement("div"); none.className = "aid-note"; none.setAttribute("data-ai-ledger-empty", "1");
    none.textContent = "No moves yet. Every change the tuner makes will be listed here with its evidence, undoable."; root.appendChild(none);
  }
  const controls = document.createElement("div"); controls.style.marginTop = "12px";
  const reset = document.createElement("button"); reset.type = "button"; reset.id = "aid-tuner-reset"; reset.textContent = "Reset knobs to standard"; controls.appendChild(reset);
  const tick = document.createElement("button"); tick.type = "button"; tick.id = "aid-tick"; tick.style.marginLeft = "8px"; tick.textContent = "Run attribution and evaluation now"; controls.appendChild(tick);
  root.appendChild(controls);
  const note = document.createElement("div"); note.className = "aid-note"; note.textContent = "Knob values are not enjoyment probabilities. Ledger rows are retained, never rewritten."; root.appendChild(note);
  return root;
}

export function mountEngine(host) {
  const view = document.createElement("div");
  view.className = "view";
  view.setAttribute("aria-label", "Engine");
  view.innerHTML = '<div class="view-head"><h1>Engine</h1><p>What the engine has measured, what it is testing, and every knob move it has made.</p></div>' +
    '<div id="aid-tuning-body" class="engine-section" aria-live="polite"></div>';
  host.appendChild(view);
  const body = view.querySelector("#aid-tuning-body");
  let config = null, disposed = false;

  function change(route) {
    if (body.getAttribute("aria-busy") === "true") return;
    const previous = body.querySelector("[data-ai-tuner-error]");
    if (previous) { clearTimeout(previous.__announcement); previous.textContent = ""; }
    body.setAttribute("aria-busy", "true");
    post(route).then(d => {
      if (d.ok !== true && !("attributed" in d)) throw new Error("Tuner change rejected");
      load();
    }).catch(error => {
      body.setAttribute("aria-busy", "false");
      let message = body.querySelector("[data-ai-tuner-error]");
      if (!message) { message = document.createElement("div"); message.setAttribute("data-ai-tuner-error", "1"); message.setAttribute("role", "status"); message.setAttribute("aria-atomic", "true"); body.appendChild(message); }
      announce(message, error.state === "configuration-required"
        ? "Tuner controls need the shared key for this exact origin. No change was accepted."
        : "Tuner change failed or was rejected. Refresh before retrying.");
    }).finally(() => body.setAttribute("aria-busy", "false"));
  }

  function load() {
    const original = body;
    const configPromise = config ? Promise.resolve(config) : getJSON("config").then(c => (config = c));
    Promise.all([getJSON("scorecard"), configPromise]).then(([sc]) => {
      if (disposed || original !== view.querySelector("#aid-tuning-body")) return;
      body.textContent = "";
      body.appendChild(renderScorecard(sc, config));
      body.querySelectorAll(".aid-revert").forEach(a => a.addEventListener("click", e => { e.preventDefault(); change("tuner/revert?ledger_id=" + a.getAttribute("data-id")); }));
      body.querySelector("#aid-tuner-reset").addEventListener("click", () => change("tuner/reset"));
      body.querySelector("#aid-tick").addEventListener("click", () => change("tick"));
    }).catch(error => { if (!disposed) setState(body, error.state || "unavailable", load); });
  }
  load();
  return { dispose() { disposed = true; }, load };
}
