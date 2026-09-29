/* The status region, the Feed/Home building panel, partial notes and toasts. One
   mounted live region per host is reused across transitions; elapsed seconds are
   visual only; timers are always cleared when the host leaves the loading state. */

const MESSAGES = {
  loading: "Loading recommendations\u2026",
  unavailable: "The engine is not answering. It may still be starting; retry in a moment.",
  error: "The request failed. Retry, or reload the page if it keeps failing.",
  empty: "Nothing matched this request.",
  partial: "Some parts of the engine are unavailable, so this result may be incomplete.",
  "no-feature": "This search isn't set up for this library.",
  "configuration-required": "Configuration required: open the address the server printed at start-up. It carries the shared key for this exact address.",
};

/* Update an empty, mounted live region on the next task; superseded text is discarded. */
export function announce(region, text) {
  clearTimeout(region.__announcement);
  region.textContent = "";
  region.__announcement = setTimeout(() => { if (region.isConnected) region.textContent = text; }, 0);
}

/* `text` replaces the stock message when the caller can name the problem more exactly.
   Retry shows only when the caller passes one, which it does for network and server failures. */
export function setState(host, state, retry, text) {
  host.setAttribute("data-ai-state", state);
  host.removeAttribute("aria-busy");
  let message = host.querySelector("[data-ai-state-message]");
  if (!message) {
    host.textContent = "";
    const line = document.createElement("div");
    line.className = "ai-status-line";
    message = document.createElement("span");
    message.setAttribute("data-ai-state-message", "1");
    message.setAttribute("role", "status");
    message.setAttribute("aria-atomic", "true");
    line.appendChild(message);
    host.appendChild(line);
  }
  announce(message, text || MESSAGES[state]);
  let button = host.querySelector("[data-ai-retry]");
  if (!button && retry && state !== "loading") {
    button = document.createElement("button");
    button.type = "button";
    button.setAttribute("data-ai-retry", "1");
    button.textContent = "Retry";
    message.parentElement.appendChild(button);
  }
  if (button) {
    button.hidden = !retry && state !== "loading";
    button.setAttribute("aria-disabled", String(state === "loading" || !retry));
    button.onclick = () => { if (button.getAttribute("aria-disabled") !== "true") retry(); };
  }
}

/* Feed and Home only: a centered panel names the work and counts seconds; after 20 s
   it says why the first load can take long, in space reserved from the start. */
export function building(host, active) {
  clearInterval(host.__buildingTimer);
  host.__buildingTimer = null;
  let panel = host.querySelector("[data-ai-building]");
  const message = host.querySelector("[data-ai-state-message]");
  const line = message && message.parentElement;
  if (!active) {
    if (panel) panel.remove();
    if (line && line.__buildingCss != null) { line.style.cssText = line.__buildingCss; line.__buildingCss = null; }
    return;
  }
  if (line && line.__buildingCss == null) {
    line.__buildingCss = line.style.cssText;
    line.style.cssText = "position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap;";
  }
  if (!panel) {
    panel = document.createElement("div");
    panel.setAttribute("data-ai-building", "1");
    const title = document.createElement("div");
    title.setAttribute("data-ai-building-title", "1");
    title.textContent = "Building your recommendations";
    const elapsed = document.createElement("div");
    elapsed.setAttribute("data-ai-building-elapsed", "1");
    elapsed.setAttribute("aria-hidden", "true");
    const note = document.createElement("p");
    note.setAttribute("data-ai-building-note", "1");
    note.textContent = "The first load after a restart can take a few minutes";
    panel.append(title, elapsed, note);
    host.insertBefore(panel, host.firstChild);
  }
  const started = Date.now();
  function tick() {
    if (!panel.isConnected || host.getAttribute("data-ai-state") !== "loading") { building(host, false); return; }
    const seconds = Math.floor((Date.now() - started) / 1000);
    panel.querySelector("[data-ai-building-elapsed]").textContent = seconds + " s";
    panel.querySelector("[data-ai-building-note]").style.visibility = seconds >= 20 ? "visible" : "hidden";
  }
  tick();
  host.__buildingTimer = setInterval(tick, 1000);
}

/* Expose partial component failures without discarding cards or moving focus. */
export function partial(host, response, retry) {
  const components = response.components || {};
  const failed = Object.entries(components).filter(([, value]) => {
    const state = typeof value === "string" ? value : value && value.status;
    return ["unavailable", "error", "no-feature", "partial"].includes(state);
  });
  if (response.status !== "partial" && !failed.length) return;
  const note = document.createElement("div");
  note.setAttribute("data-ai-partial", "1");
  host.appendChild(note);
  setState(note, "partial", retry);
  failed.forEach(([name, value]) => {
    const line = document.createElement("div");
    line.textContent = name.replace(/_/g, " ") + ": " + (value.status || value);
    note.appendChild(line);
  });
}

export function toast(text, action, label) {
  const box = document.createElement("div");
  box.className = "toast";
  box.setAttribute("role", "status");
  box.textContent = text;
  if (action) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label || "Undo";
    button.onclick = () => { box.remove(); action(); };
    box.appendChild(button);
  }
  document.getElementById("toasts").appendChild(box);
  setTimeout(() => box.remove(), action ? 12000 : 6000);
  return box;
}

/* Displayed decimals go through the reader's locale, never toFixed. */
export function formatNumber(value, digits = 2, fixed = false) {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits, minimumFractionDigits: fixed ? digits : 0 }).format(value);
}

export function formatTime(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return null;
  const s = Math.floor(seconds);
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
}

export function whenTime(ts) {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(ts * 1000));
}
