/* Engine dashboard: the taste profile behind the latest Feed page, verdict outcomes, the
   real scorecard, evidence gates in plain sentences, the current experiment, and the
   reversible tuner ledger. Missing counts stay missing; demonstration evidence is labelled
   as such; every control is confirmed by the server before the panel refreshes. The view
   refreshes itself every 15 s while the tab is visible, re-renders only when the data
   changed, and keeps focus on the control that was in use. */
import { config, getJSON, post, readProfile } from "./api.js";
import { announce, formatTime, setState, whenTime } from "./state.js";
import { icon } from "./icons.js";

const REFRESH_S = 15;

/* Each reason code the engine reports, as a sentence; the raw code stays in the tooltip. */
const REASONS = {
  no_qualified_views: "No qualified views yet: a pick counts once it stays mostly on screen for a moment.",
  attribution_pending: "Some outcomes are still inside the attribution window.",
  unknown_source_provenance: "Some deliveries have no recorded source, so they cannot be attributed.",
  session_mapping_changed: "A session was re-mapped after delivery, so its trials are held back.",
  session_mapping_unresolved: "Some sessions are not mapped to a canonical session yet.",
  feedback_unresolved: "A feedback action is still unconfirmed.",
  sync_capture_unresolved: "A watch-capture sync has not finished.",
  sync_capture_gap: "Watch capture has a gap, so outcomes inside it are not trusted.",
  watch_capture_quarantined: "Watch capture is quarantined after an inconsistency.",
  watch_capture_unavailable: "Watch capture has not produced a committed outcome yet.",
  verdict_config_missing: "The verdict settings are missing.",
  correction_pending: "A correction to an outcome is still pending.",
  cumulative_verdict_unavailable: "The cumulative verdict for a trial is unavailable.",
  cumulative_cutoff_mismatch: "A cumulative verdict used a different cutoff.",
  no_eligible_trials: "No trials are eligible for evaluation yet.",
  attributed_evidence_unavailable: "No attributed evidence is available yet.",
  automatic_promotion_disabled: "Automatic promotion is off.",
  session_evidence_policy_required: "Promotion needs a session evidence policy.",
  evidence_invalid: "The evidence is not valid yet.",
  outside_attribution_window: "Some outcomes fell outside the attribution window.",
  pre_exposure_watch: "Some watching happened before the pick was shown.",
  invalid_window: "An attribution window is invalid.",
  eligibility_snapshot_future: "An eligibility snapshot is dated in the future.",
  invalid_shared_reward: "A shared reward was invalid.",
  no_preference_history: "No ratings or watch history yet.",
  no_positive_tag_profile: "No liked tags yet.",
  no_embedding_profile: "No look profile yet.",
};
const LEDGER_STATUS = { applied: "Applied", reverted: "Undone", revert: "Undo", reset_to_standard: "Reset to standard", superseded_by_reset: "Superseded by reset" };

function humanize(code) { const text = String(code).replace(/_/g, " "); return text.charAt(0).toUpperCase() + text.slice(1); }
function reasonText(code) { return REASONS[code] || humanize(code) + "."; }
function esc(value) { return value == null ? "Unavailable" : String(value); }
function num(value, digits = 3) { return typeof value === "number" && Number.isFinite(value) ? new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value) : "Unavailable"; }

function armWords(label, arm) {
  if (!arm || typeof arm !== "object") return null;
  const bits = [label];
  if (arm.trials != null) bits.push(num(arm.trials, 0) + " trials");
  if (arm.successes != null) bits.push(num(arm.successes, 0) + " liked");
  if (typeof arm.mean_reward === "number") bits.push("mean reward " + num(arm.mean_reward, 4));
  if (typeof arm.like_rate === "number") bits.push(Math.round(arm.like_rate * 100) + "% like rate");
  return bits.join(", ");
}

function evidenceWords(ev) {
  if (!ev || typeof ev !== "object") return "Unavailable";
  const parts = [armWords("Base", ev.base), armWords("candidate", ev.cand || ev.candidate)].filter(Boolean);
  if (typeof ev.diff === "number") parts.push("difference " + (ev.diff > 0 ? "+" : "") + num(ev.diff, 4) + (typeof ev.ci_halfwidth === "number" ? " \u00B1 " + num(ev.ci_halfwidth, 4) : ""));
  if (ev.reason) parts.push(reasonText(ev.reason).replace(/\.$/, "") + (ev.next ? ", next knob " + ev.next : ""));
  if (ev.reverted_ledger_id != null) parts.push("undo of move " + ev.reverted_ledger_id);
  if (ev.status && !parts.length) parts.push(humanize(ev.status));
  return parts.length ? parts.join(" \u00B7 ") : "No evidence recorded";
}

function panel(title, ...children) {
  const box = document.createElement("section"); box.className = "panel";
  const head = document.createElement("h2"); head.textContent = title;
  box.append(head, ...children);
  return box;
}

function element(tag, cls, text) { const node = document.createElement(tag); if (cls) node.className = cls; if (text != null) node.textContent = text; return node; }

/* The profile is the one the ranker reported with the most recent Feed page in this tab
   (kept by api.js); the verdict counts are the scorecard's attributed trial outcomes. */
function summary(sc, record) {
  const t = sc.tuner || {}, arms = t.arms || {}, capture = sc.capture || {}, auto = sc.automation || {};
  const p = record && record.profile && typeof record.profile === "object" && Object.keys(record.profile).length ? record.profile : null;
  const box = element("dl", "summary"); box.setAttribute("data-ai-summary", "1");
  function stat(label, value, tone, hint) {
    const cell = element("div", tone || null); cell.append(element("dt", null, label), element("dd", "num", value));
    if (hint) cell.title = hint;
    box.appendChild(cell);
  }
  const from = record && record.at ? "From the Feed page served " + whenTime(record.at) : null;
  if (p && p.reason) stat("Taste profile", "None yet", null, reasonText(p.reason) + (from ? " " + from : ""));
  else if (p) {
    stat("Profile tags", num(p.profile_tags, 0), null, from);
    stat("Tags pulling up", num(p.positive_tags, 0), "is-positive");
    stat("Tags pushing down", num(p.negative_tags, 0), "is-negative");
    stat("Watch evidence", typeof p.watch_evidence_s === "number" ? formatTime(p.watch_evidence_s) : "Unavailable");
  } else stat("Taste profile", "Unavailable", null, "The latest Feed page in this tab reported none. Open Feed to load one.");
  const armRows = ["base", "cand"].map(name => arms[name] || (name === "cand" ? arms.candidate : null)).filter(a => a && typeof a === "object");
  const trials = armRows.reduce((sum, a) => sum + (a.trials || 0), 0), liked = armRows.reduce((sum, a) => sum + (a.successes || 0), 0);
  stat("Attributed verdicts", armRows.length ? num(trials, 0) : "Unavailable");
  stat("Liked outcomes", armRows.length ? num(liked, 0) : "Unavailable", "is-positive");
  stat("Watch outcomes", num(capture.captured_outcomes, 0));
  stat("Automatic tuning", auto.enabled === true ? "On" : "Off");
  return box;
}

function armBars(arms) {
  const box = document.createElement("div");
  [["Base", arms.base], ["Candidate", arms.cand || arms.candidate]].forEach(([label, arm]) => {
    if (!arm || typeof arm !== "object") return;
    const row = element("div", "arm");
    const track = element("span", "arm-track"); track.setAttribute("role", "img");
    const rate = typeof arm.like_rate === "number" ? arm.like_rate : null;
    track.setAttribute("aria-label", label + " like rate: " + (rate === null ? "not measured" : Math.round(rate * 100) + "%"));
    if (rate !== null) { const fill = document.createElement("span"); fill.style.width = Math.round(Math.max(0, Math.min(1, rate)) * 100) + "%"; track.appendChild(fill); }
    row.append(element("span", null, label), track, element("span", "num", (rate === null ? "Not measured" : Math.round(rate * 100) + "% liked") + ", " + num(arm.trials, 0) + (arm.trials === 1 ? " trial" : " trials")));
    box.appendChild(row);
  });
  return box;
}

function cell(text, cls) { return element("td", cls, text); }

function gate(ok, text) {
  const box = element("div", "gate" + (ok ? " ok" : ""));
  box.append(icon(ok ? "check" : "more"), element("span", null, text));
  return box;
}

export function renderScorecard(sc, cfg, profileRecord) {
  const t = sc.tuner || {}, auto = sc.automation || {}, evidence = sc.evidence || {}, capture = sc.capture || {}, g = sc.gate || {};
  const root = document.createElement("div");
  const captureVerified = capture.verified === true;
  const captureBox = gate(captureVerified, captureVerified
    ? "Watch capture verified: " + num(capture.captured_outcomes, 0) + " committed watch outcomes from " + num((capture.receipts || {}).imported || 0, 0) + " imported batches" +
      (capture.last_imported_at ? ", last at " + whenTime(capture.last_imported_at) : "") + "; " + num(capture.attributed_outcomes || 0, 0) + " outcomes attributed" +
      (capture.attribution_through_ts ? " through " + whenTime(capture.attribution_through_ts) : "") + "."
    : "Watch capture is not verified yet: no committed watch batch has produced an outcome. Player configuration alone is not proof.");
  captureBox.setAttribute("data-ai-capture-status", captureVerified ? "enabled" : "unavailable");
  const promotion = gate(auto.enabled === true, auto.enabled === true
    ? "Automatic tuning is on: every " + esc(auto.interval_hours) + " h the engine re-evaluates, and it promotes a knob by itself once each arm clears " + esc(g.min_trials_per_arm) +
      " ripened trials across " + esc(auto.min_sessions_per_arm) + " sessions with a significant, diversity-safe win. Trials ripen " + esc(g.ripen_hours) + " h after attribution."
    : "Automatic promotion is off. Decisions need an explicit run.");
  promotion.setAttribute("data-ai-promotion", auto.enabled === true ? "automatic" : "disabled");
  const codes = Array.from(new Set((evidence.validity_reasons || []).concat(evidence.promotion_reasons || [])));
  const reasons = element("div", "aid-note");
  reasons.setAttribute("data-ai-evidence", evidence.valid === true ? "valid" : "invalid");
  reasons.appendChild(element("p", null, evidence.valid === true ? "The evidence is valid." : codes.length ? "The evidence is not valid yet:" : "The evidence is not valid yet; no reason was reported."));
  if (codes.length) {
    const list = element("ul", "reasons");
    codes.forEach(code => { const entry = element("li", null, reasonText(code)); entry.title = code; list.appendChild(entry); });
    reasons.appendChild(list);
  }
  const setup = element("p", "aid-note");
  if (!cfg) setup.textContent = "Engine configuration is unavailable, so the attribution policy and search encoder are not shown.";
  else {
    const encoders = Object.values(cfg.space_meta || {}).map(m => m && m.encoder).filter(Boolean);
    setup.textContent = cfg.encoder ? "Text search uses " + (encoders.join(", ") || "the installed encoder") + "." : "No text encoder is installed, so Search reports no usable feature.";
    const a = cfg.attribution || {};
    if (a.policy_revision && a.policy_revision.startsWith("demo")) {
      const demo = element("span", "aid-chip demo", "Demonstration policy: " + a.window_s + " s attribution window, " + a.policy_revision);
      demo.setAttribute("data-ai-demo-label", "1");
      setup.append(" ", demo);
    }
  }
  root.append(summary(sc, profileRecord), panel("Evidence gates", captureBox, promotion, reasons, setup));

  const experiment = element("p", "aid-note", t.knob ? "Testing " + t.knob + ": base " + esc(t.base) + " against candidate " + esc(t.candidate) +
    (t.experiment_started ? ", since " + whenTime(t.experiment_started) : "") + ". Stalls " + esc(t.stalls) + "; eligible for promotion now: " + (t.promotion_eligible === true ? "yes" : "no") + "."
    : "No active experiment. Arms are created when the tuner is initialized or reset.");
  experiment.setAttribute("data-ai-experiment", t.knob || "none");
  const experimentPanel = panel("Current experiment", experiment);
  if (t.arms && Object.keys(t.arms).length) experimentPanel.append(armBars(t.arms), element("p", "aid-note", "Attributed trials: " + evidenceWords(t.arms)));
  root.appendChild(experimentPanel);

  const knobs = element("div", "aid-scroll");
  const table = element("table", "aid-tbl");
  table.innerHTML = "<thead><tr><th scope=col>Knob</th><th scope=col class=num>Default</th><th scope=col class=num>Settled</th><th scope=col class=num>Base</th><th scope=col class=num>Candidate</th></tr></thead>";
  const tbody = document.createElement("tbody");
  (t.knobs || []).forEach(knob => {
    const tr = document.createElement("tr");
    tr.append(cell(esc(knob.knob), "mono"), cell(num(knob.default), "num"), cell(num(knob.settled), "num"), cell(num(knob.base), "num"), cell(num(knob.candidate), "num"));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody); knobs.appendChild(table); root.appendChild(panel("Knobs", knobs));

  const ledgerPanel = panel("Tuner ledger"); root.appendChild(ledgerPanel);
  const ledgerHead = ledgerPanel.querySelector("h2"); ledgerHead.tabIndex = -1; ledgerHead.setAttribute("data-focus-key", "ledger");
  if ((t.ledger || []).length) {
    const scroll = element("div", "aid-scroll");
    const ledger = element("table", "aid-tbl"); ledger.setAttribute("data-ai-ledger", "1");
    ledger.innerHTML = "<thead><tr><th scope=col>When</th><th scope=col>Knob</th><th scope=col class=num>Move</th><th scope=col>Status</th><th scope=col>Evidence</th><th scope=col><span class=sr-only>Action</span></th></tr></thead>";
    const rows = document.createElement("tbody");
    t.ledger.forEach(l => {
      const tr = document.createElement("tr"); tr.setAttribute("data-ai-ledger-row", String(l.id)); tr.setAttribute("data-ai-ledger-status", esc(l.status));
      tr.append(cell(whenTime(l.ts)), cell(esc(l.knob), "mono"), cell(num(l.old) + " to " + num(l.new), "num"));
      const status = document.createElement("td"); const chip = element("span", "aid-chip" + (l.status === "applied" ? " warn" : ""), LEDGER_STATUS[l.status] || humanize(esc(l.status)));
      chip.title = esc(l.status); status.appendChild(chip); tr.appendChild(status);
      const ev = l.evidence || {};
      const evidenceCell = cell(evidenceWords(ev));
      if (ev.demonstration === true) evidenceCell.append(" ", element("span", "aid-chip demo", "demonstration evidence"));
      tr.appendChild(evidenceCell);
      const action = document.createElement("td");
      if (l.status === "applied") {
        const undo = element("button", "aid-revert", "Undo move"); undo.type = "button";
        undo.setAttribute("data-id", String(l.id)); undo.setAttribute("data-focus-key", "revert-" + l.id);
        action.appendChild(undo);
      }
      tr.appendChild(action); rows.appendChild(tr);
    });
    ledger.appendChild(rows); scroll.appendChild(ledger); ledgerPanel.appendChild(scroll);
  } else {
    const none = element("p", "aid-note", "No moves yet. Every change the tuner makes is listed here with its evidence, and each can be undone.");
    none.setAttribute("data-ai-ledger-empty", "1"); ledgerPanel.appendChild(none);
  }
  const knobCount = (t.knobs || []).length;
  const controls = element("div", "aid-controls");
  const reset = element("button", null, "Reset knobs to standard"); reset.type = "button"; reset.id = "aid-tuner-reset"; reset.setAttribute("data-focus-key", "reset");
  reset.setAttribute("aria-expanded", "false"); reset.setAttribute("aria-controls", "aid-reset-confirm");
  const tick = element("button", null, "Run attribution and evaluation now"); tick.type = "button"; tick.id = "aid-tick"; tick.setAttribute("data-focus-key", "tick");
  controls.append(reset, tick);
  const confirm = element("div", "confirm"); confirm.id = "aid-reset-confirm"; confirm.hidden = true; confirm.setAttribute("role", "group"); confirm.setAttribute("aria-labelledby", "aid-reset-question");
  const question = element("p", null, "Reset all " + num(knobCount, 0) + " knobs to their standard values and restart the experiment around them? " +
    "Applied moves become superseded and can no longer be undone one by one. The ledger keeps every row.");
  question.id = "aid-reset-question";
  const yes = element("button", "danger", "Reset " + num(knobCount, 0) + " knobs"); yes.type = "button"; yes.setAttribute("data-ai-reset-confirm", "1");
  const no = element("button", null, "Cancel"); no.type = "button"; no.setAttribute("data-ai-reset-cancel", "1");
  const choices = element("div", "aid-controls"); choices.append(yes, no);
  confirm.append(question, choices);
  ledgerPanel.append(controls, confirm, element("p", "aid-note", "Knob values are not enjoyment probabilities. Ledger rows are kept, never rewritten."));
  return root;
}

function skeleton() {
  const box = element("div", "engine-skeleton"); box.setAttribute("aria-hidden", "true");
  ["is-row", "is-panel", "is-panel", "is-panel"].forEach(cls => box.appendChild(element("div", "skeleton-block " + cls)));
  return box;
}

export function mountEngine(host) {
  const view = document.createElement("div");
  view.className = "view";
  view.setAttribute("aria-label", "Engine");
  view.innerHTML = '<div class="view-head"><h1>Engine</h1><p>Your taste profile, what the engine has measured, what it is testing, and every knob move it has made.</p>' +
    '<div class="engine-toolbar"><span class="updated num" data-ai-updated></span><button type="button" id="aid-refresh" class="action"><span>Refresh</span></button></div></div>' +
    '<p class="action-status" data-ai-engine-action role="status" aria-live="polite" aria-atomic="true"></p>' +
    '<div id="aid-tuning-body" class="engine-section"></div>';
  host.appendChild(view);
  const body = view.querySelector("#aid-tuning-body"), action = view.querySelector("[data-ai-engine-action]");
  const refresh = view.querySelector("#aid-refresh"), updated = view.querySelector("[data-ai-updated]");
  refresh.prepend(icon("refresh"));
  body.appendChild(skeleton());
  let disposed = false, loading = false, again = false, busy = false, loadedAt = 0, signature = null;

  function showUpdated() {
    if (!loadedAt) { updated.textContent = ""; return; }
    const seconds = Math.floor((Date.now() - loadedAt) / 1000);
    updated.textContent = seconds < 2 ? "Updated just now" : seconds < 60 ? "Updated " + seconds + " s ago" : "Updated " + Math.floor(seconds / 60) + " min ago";
  }

  const confirmOpen = () => { const box = body.querySelector("#aid-reset-confirm"); return !!box && !box.hidden; };

  function paint(root) {
    const active = document.activeElement;
    const key = active && body.contains(active) ? active.getAttribute("data-focus-key") : null;
    body.removeAttribute("data-ai-state");
    body.replaceChildren(root);
    wire();
    if (key) {
      const target = body.querySelector('[data-focus-key="' + CSS.escape(key) + '"]') || body.querySelector('[data-focus-key="ledger"]');
      if (target) target.focus({ preventScroll: true });
    }
  }

  function load(manual) {
    if (loading) { again = true; return; }
    loading = true; refresh.setAttribute("aria-disabled", "true");
    Promise.all([getJSON("scorecard"), config().catch(() => null)]).then(([sc, cfg]) => {
      if (disposed) return;
      loadedAt = Date.now(); showUpdated();
      const profile = readProfile();
      const next = JSON.stringify([sc, cfg, profile]);
      if (next !== signature && !(confirmOpen() && !manual)) { signature = next; paint(renderScorecard(sc, cfg, profile)); }
      if (manual) announce(action, "Refreshed.");
    }).catch(error => {
      if (disposed) return;
      if (signature === null) setState(body, error.state || "unavailable", () => load(true));
      else announce(action, "Could not refresh: the engine did not answer. Showing the last result.");
    }).finally(() => {
      loading = false; refresh.setAttribute("aria-disabled", "false");
      if (again && !disposed) { again = false; load(); }
    });
  }

  function change(route, done) {
    if (busy) return;
    busy = true; body.setAttribute("aria-busy", "true");
    action.removeAttribute("data-ai-tuner-error");
    announce(action, "Working\u2026");
    post(route).then(d => {
      if (d.ok !== true && !("attributed" in d)) throw new Error("Tuner change rejected");
      announce(action, done);
      load();
    }).catch(error => {
      action.setAttribute("data-ai-tuner-error", "1");
      announce(action, error.state === "configuration-required"
        ? "Tuner controls need the shared key for this address. Open the link the server printed at start-up. No change was made."
        : "The tuner change was not confirmed. Refresh to see the current state, then retry.");
    }).finally(() => { busy = false; body.setAttribute("aria-busy", "false"); });
  }

  function wire() {
    body.querySelectorAll(".aid-revert").forEach(button => button.addEventListener("click", () =>
      change("tuner/revert?ledger_id=" + button.getAttribute("data-id"), "Move " + button.getAttribute("data-id") + " undone.")));
    const reset = body.querySelector("#aid-tuner-reset"), box = body.querySelector("#aid-reset-confirm");
    function closeConfirm() { box.hidden = true; reset.setAttribute("aria-expanded", "false"); reset.focus(); }
    reset.addEventListener("click", () => { box.hidden = false; reset.setAttribute("aria-expanded", "true"); box.querySelector("[data-ai-reset-cancel]").focus(); });
    box.querySelector("[data-ai-reset-cancel]").addEventListener("click", closeConfirm);
    box.addEventListener("keydown", event => { if (event.key === "Escape") closeConfirm(); });
    box.querySelector("[data-ai-reset-confirm]").addEventListener("click", () => { closeConfirm(); change("tuner/reset", "Knobs reset to their standard values."); });
    body.querySelector("#aid-tick").addEventListener("click", () => change("tick", "Attribution and evaluation ran."));
  }

  refresh.addEventListener("click", () => { if (refresh.getAttribute("aria-disabled") !== "true") load(true); });
  const clock = setInterval(() => {
    showUpdated();
    if (document.visibilityState === "visible" && loadedAt && Date.now() - loadedAt >= REFRESH_S * 1000 && !busy && !confirmOpen()) load();
  }, 1000);
  load();
  return { dispose() { disposed = true; clearInterval(clock); }, load };
}
