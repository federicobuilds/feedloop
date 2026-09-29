/* Managed feedback: one pending operation per item, confirmed authority painted only
   from the ledger result, Undo by receipt (operation id), conflict and uncertain states.
   Like and Dislike are pressed toggles (pressing the active one clears the rating); Clear
   rating and Count engagement sit behind "More"; Undo sits next to the confirmation. */
import { itemKey, newId, post, session } from "./api.js";
import { toast } from "./state.js";
import { icon } from "./icons.js";

const states = new Map();
const pending = JSON.parse(sessionStorage.getItem("feedloop-feedback-pending") || "{}");

function savePending(key, id) {
  if (id) pending[key] = id; else delete pending[key];
  sessionStorage.setItem("feedloop-feedback-pending", JSON.stringify(pending));
}

function control(name, label, iconName, compact) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "action";
  if (name) button.setAttribute("data-ai-feedback", name);
  button.appendChild(icon(iconName));
  const text = document.createElement("span"); text.textContent = label;
  if (compact) { text.className = "sr-only"; button.title = label; }
  button.appendChild(text);
  return button;
}

/* `extras` are placed between the rating toggles and "More" (Next, More like this);
   `compact` renders icon buttons for cards. */
export function feedbackControls(item, { extras = [], compact = false } = {}) {
  const key = itemKey(item);
  const row = document.createElement("div");
  row.className = "feedback-row";
  const actions = document.createElement("div");
  actions.className = "feedback-actions";
  const like = control("like", "Like", "like", compact), dislike = control("dislike", "Dislike", "dislike", compact);
  const more = control(null, "More", "more", compact);
  const menuId = "fb-more-" + newId();
  more.setAttribute("aria-expanded", "false");
  more.setAttribute("aria-controls", menuId);
  actions.append(like, dislike, ...extras, more);
  const menu = document.createElement("div");
  menu.className = "feedback-more";
  menu.id = menuId;
  menu.hidden = true;
  const clear = document.createElement("button"), engagement = document.createElement("button");
  [[clear, "clear", "Clear rating", "Removes your Like or Dislike; the item is judged by watching again."],
    [engagement, "engagement", "Count engagement", "Records that this held your attention, without a rating."]].forEach(([button, name, label, help]) => {
    button.type = "button"; button.setAttribute("data-ai-feedback", name); button.textContent = label;
    const line = document.createElement("div"); line.className = "feedback-more-item";
    const note = document.createElement("p"); note.textContent = help;
    line.append(button, note); menu.appendChild(line);
  });
  more.onclick = () => { menu.hidden = !menu.hidden; more.setAttribute("aria-expanded", String(!menu.hidden)); };
  const status = document.createElement("div");
  status.className = "feedback-status";
  const note = document.createElement("span");
  note.className = "feedback-note";
  note.setAttribute("aria-live", "polite");
  const undo = document.createElement("button");
  undo.type = "button";
  undo.className = "link-button";
  undo.setAttribute("data-ai-undo", "1");
  undo.textContent = "Undo";
  undo.hidden = true;
  status.append(note, undo);
  row.append(actions, menu, status);

  let state = states.get(key);
  if (!state) { state = { busy: false, unresolved: !!pending[key], rating: item.rating100 == null ? null : item.rating100, engagement: null, last: null }; states.set(key, state); }
  state.unresolved = !!pending[key];
  const liked = () => state.rating !== null && state.rating >= 60, disliked = () => state.rating !== null && state.rating < 50;
  like.onclick = () => submit({ rating100: liked() ? null : 90 });
  dislike.onclick = () => submit({ rating100: disliked() ? null : 10 });
  clear.onclick = () => submit({ rating100: null });
  engagement.onclick = () => submit({ engagement: true });
  undo.onclick = () => { if (state.last) submit({ undo_of: state.last.operation_id }, state.last); };
  const controls = [like, dislike, clear, engagement];

  function say(text) { note.textContent = text; tidy(); }
  function tidy() { status.classList.toggle("is-empty", !note.textContent && undo.hidden); }

  /* While an operation is unresolved, new actions stay disabled: only "Check status" may
     run, so the pending operation identity is never overwritten. */
  function paint() {
    like.setAttribute("aria-pressed", String(liked()));
    dislike.setAttribute("aria-pressed", String(disliked()));
    engagement.textContent = state.engagement == null ? "Count engagement" : "Count engagement (" + state.engagement + ")";
    undo.hidden = !state.last;
    const blocked = state.busy || state.unresolved;
    controls.forEach(b => b.setAttribute("aria-disabled", String(blocked)));
    undo.setAttribute("aria-disabled", String(blocked));
    row.setAttribute("data-ai-feedback-rating", state.rating === null ? "none" : String(state.rating));
    row.setAttribute("data-ai-feedback-unresolved", String(state.unresolved));
    tidy();
  }

  function confirmation(operation) {
    if (operation.action === "engagement") return "Saved: engagement counted.";
    if (operation.rating100 === null) return "Saved: rating cleared.";
    return operation.rating100 >= 60 ? "Saved: liked." : "Saved: disliked.";
  }

  function submit(change, undoOf) {
    if (state.busy || state.unresolved) return;
    if (item.session_id && item.session_id !== session()) { say("This pick came from an earlier session. Reload the feed, then rate it again."); return; }
    const operation = { operation_id: newId(), session_id: session(), kind: item.kind, item_id: item.id };
    if (undoOf) Object.assign(operation, { action: "undo", undo_of: change.undo_of });
    else if (change.engagement) operation.action = "engagement";
    else Object.assign(operation, { action: "rating", rating100: change.rating100 });
    if (item.request_id) operation.request_id = item.request_id;
    if (item.viewed_event_id) operation.viewed_event_id = item.viewed_event_id;
    state.busy = true; savePending(key, operation.operation_id); state.unresolved = true; paint(); row.setAttribute("aria-busy", "true");
    say("Saving\u2026");
    return post("feedback", operation).then(result => {
      if (result.operation_id !== operation.operation_id) { uncertain(); return; }
      if (result.status === "conflict") { resolve(); say("Not saved: another action changed this item first. Try again."); return; }
      if (result.status !== "confirmed" || !result.after) { uncertain(); return; }
      resolve();
      state.rating = result.after.rating100 === undefined ? state.rating : result.after.rating100;
      state.engagement = Number.isSafeInteger(result.after.engagement_count) ? result.after.engagement_count : state.engagement;
      state.last = undoOf ? null : { operation_id: result.operation_id };
      say(undoOf ? "Undone: the previous value is back." : confirmation(operation));
      row.setAttribute("data-ai-feedback-status", undoOf ? "undone" : "confirmed");
    }).catch(error => {
      if (error && error.state === "configuration-required") {
        resolve(); say("Saving feedback needs the shared key for this address. Open the link the server printed at start-up.");
        row.setAttribute("data-ai-feedback-status", "configuration-required"); return;
      }
      uncertain();
    }).finally(() => { state.busy = false; row.removeAttribute("aria-busy"); paint(); });
  }

  function resolve() { savePending(key, null); state.unresolved = false; }

  function uncertain(text) {
    state.unresolved = true;
    row.setAttribute("data-ai-feedback-status", "uncertain");
    say(text || "Not confirmed yet. Check status before another action.");
    const check = document.createElement("button");
    check.type = "button"; check.className = "link-button"; check.textContent = "Check status";
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
        say("Confirmed after checking."); row.setAttribute("data-ai-feedback-status", "confirmed");
      } else if (result.status === "conflict" || result.status === "released" || result.error_code === "unknown_operation") {
        resolve(); say(result.error_code === "unknown_operation" ? "The action never reached the engine. Nothing changed." : "The action did not apply. Nothing changed.");
        row.setAttribute("data-ai-feedback-status", "conflict");
      } else { uncertain(); }
    }).catch(() => uncertain()).finally(() => { state.busy = false; paint(); });
  }

  paint();
  if (pending[key]) uncertain("An earlier action is not confirmed yet. Check status before another action.");
  return row;
}

export { toast };
