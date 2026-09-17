/* Managed feedback: one pending operation per item, confirmed authority painted only
   from the ledger result, Undo by receipt (operation id), conflict and uncertain states. */
import { itemKey, newId, post, session } from "./api.js";
import { toast } from "./state.js";

const states = new Map();
const pending = JSON.parse(sessionStorage.getItem("feedloop-feedback-pending") || "{}");

function savePending(key, id) {
  if (id) pending[key] = id; else delete pending[key];
  sessionStorage.setItem("feedloop-feedback-pending", JSON.stringify(pending));
}

export function feedbackControls(item, options = {}) {
  const key = itemKey(item);
  const row = document.createElement("div");
  row.className = "feedback-row";
  const note = document.createElement("div");
  note.className = "feedback-note";
  note.setAttribute("aria-live", "polite");
  const buttons = [["like", "Like", { rating100: 90 }], ["dislike", "Dislike", { rating100: 10 }], ["clear", "Clear rating", { rating100: null }],
    ["engagement", "Count engagement", { engagement: true }]];
  let state = states.get(key);
  if (!state) { state = { busy: false, unresolved: !!pending[key], rating: item.rating100 == null ? null : item.rating100, engagement: null, last: null }; states.set(key, state); }
  state.unresolved = !!pending[key];
  const controls = {};
  buttons.forEach(([name, label, change]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("data-ai-feedback", name);
    button.textContent = label;
    button.onclick = () => submit(change);
    controls[name] = button;
    row.appendChild(button);
  });
  const undo = document.createElement("button");
  undo.type = "button";
  undo.setAttribute("data-ai-undo", "1");
  undo.textContent = "Undo last";
  undo.hidden = true;
  undo.onclick = () => { if (state.last) submit({ undo_of: state.last.operation_id }, state.last); };
  row.appendChild(undo);
  row.appendChild(note);

  /* While an operation is unresolved, new actions stay disabled: only "Check status" may
     run, so the pending operation identity is never overwritten. */
  function paint() {
    controls.like.setAttribute("aria-pressed", String(state.rating !== null && state.rating >= 60));
    controls.dislike.setAttribute("aria-pressed", String(state.rating !== null && state.rating < 50));
    controls.engagement.textContent = state.engagement == null ? "Count engagement" : "Count engagement (" + state.engagement + ")";
    undo.hidden = !state.last;
    const blocked = state.busy || state.unresolved;
    Object.values(controls).forEach(b => b.setAttribute("aria-disabled", String(blocked)));
    undo.setAttribute("aria-disabled", String(blocked));
    row.setAttribute("data-ai-feedback-rating", state.rating === null ? "none" : String(state.rating));
    row.setAttribute("data-ai-feedback-unresolved", String(state.unresolved));
  }

  function submit(change, undoOf) {
    if (state.busy || state.unresolved) return;
    if (item.session_id && item.session_id !== session()) { note.textContent = "This delivery belongs to an earlier session. Reload the feed before changing feedback."; return; }
    const operation = { operation_id: newId(), session_id: session(), kind: item.kind, item_id: item.id };
    if (undoOf) Object.assign(operation, { action: "undo", undo_of: change.undo_of });
    else if (change.engagement) operation.action = "engagement";
    else Object.assign(operation, { action: "rating", rating100: change.rating100 });
    if (item.request_id) operation.request_id = item.request_id;
    if (item.viewed_event_id) operation.viewed_event_id = item.viewed_event_id;
    state.busy = true; savePending(key, operation.operation_id); state.unresolved = true; paint(); row.setAttribute("aria-busy", "true");
    note.textContent = "Sending\u2026";
    return post("feedback", operation).then(result => {
      if (result.operation_id !== operation.operation_id) { uncertain(); return; }
      if (result.status === "conflict") { resolve(); note.textContent = "Nothing changed: the item was changed by another action. Nothing was confirmed."; return; }
      if (result.status !== "confirmed" || !result.after) { uncertain(); return; }
      resolve();
      state.rating = result.after.rating100 === undefined ? state.rating : result.after.rating100;
      state.engagement = Number.isSafeInteger(result.after.engagement_count) ? result.after.engagement_count : state.engagement;
      state.last = undoOf ? null : { operation_id: result.operation_id };
      note.textContent = undoOf ? "Original value restored." : "Confirmed: " + (operation.action === "engagement" ? "engagement counted" : operation.rating100 === null ? "rating cleared" : "rating " + operation.rating100) + ".";
      row.setAttribute("data-ai-feedback-status", undoOf ? "undone" : "confirmed");
      if (options.onConfirmed) options.onConfirmed(state);
    }).catch(error => {
      if (error && error.state === "configuration-required") { resolve(); note.textContent = "Managed feedback needs the shared key for this exact origin. Open the address printed by the server."; row.setAttribute("data-ai-feedback-status", "configuration-required"); return; }
      uncertain();
    }).finally(() => { state.busy = false; row.removeAttribute("aria-busy"); paint(); });
  }

  function resolve() { savePending(key, null); state.unresolved = false; }

  function uncertain(text) {
    state.unresolved = true;
    row.setAttribute("data-ai-feedback-status", "uncertain");
    note.textContent = text || "Result unconfirmed. Check status before another action.";
    const check = document.createElement("button");
    check.type = "button"; check.textContent = "Check status";
    check.setAttribute("data-ai-check-status", "1");
    check.onclick = () => reconcile();
    note.appendChild(check);
    paint();
  }

  function reconcile() {
    const id = pending[key];
    if (!id || state.busy) return;
    state.busy = true; paint();
    post("feedback/reconcile", { operation_id: id }).then(result => {
      if (result.operation_id !== id) { uncertain(); return; }
      if (result.status === "confirmed" && result.after) {
        resolve(); state.rating = result.after.rating100; state.engagement = result.after.engagement_count;
        state.last = { operation_id: id };
        note.textContent = "Confirmed after check."; row.setAttribute("data-ai-feedback-status", "confirmed");
      } else if (result.status === "conflict" || result.status === "released" || result.error_code === "unknown_operation") {
        resolve(); note.textContent = result.error_code === "unknown_operation" ? "The pending action never reached the engine." : "The pending action did not apply."; row.setAttribute("data-ai-feedback-status", "conflict");
      } else { uncertain(); }
    }).catch(() => uncertain()).finally(() => { state.busy = false; paint(); });
  }

  paint();
  if (pending[key]) uncertain("A previous action is unconfirmed. Check status before another action.");
  return row;
}

export { toast };
