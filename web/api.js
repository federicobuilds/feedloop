/* Transport and result classification. Every result is one of: data, or an Error
   carrying `state` (unavailable | error | empty | partial | no-feature |
   configuration-required), `code`, `components`, and for deliveries `retryDelivery`. */

const STALE_CURSOR_CODES = ["stale_ranking_cursor", "cursor_offset_mismatch", "invalid cursor", "invalid_cursor",
  "ranking_generation_changed", "ranking_context_conflict"];
const DENIED = { mutation_auth_unconfigured: 503, mutation_origin_unconfigured: 503, mutation_credential_required: 401, mutation_origin_denied: 403 };

export function newId() {
  const bytes = new Uint8Array(12);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
}

export function session() {
  let value = sessionStorage.getItem("feedloop-session");
  if (!value) { value = "s-" + newId(); sessionStorage.setItem("feedloop-session", value); }
  return value;
}

export function apiKey() {
  const match = /(?:^|[#&])key=([^&]+)/.exec(location.hash);
  if (match) {
    sessionStorage.setItem("feedloop-key", decodeURIComponent(match[1]));
    history.replaceState(null, "", location.pathname + location.search + "#/feed");
  }
  return sessionStorage.getItem("feedloop-key") || "";
}

export function isStaleCursor(error) {
  return STALE_CURSOR_CODES.includes(error && error.code);
}

/* Reject transport, declared backend and malformed results before rendering. */
export function classify(response, listKey) {
  if (!response.ok) {
    const error = new Error("Request failed");
    error.state = response.status === 401 || response.status === 403 ? "configuration-required" : response.status >= 500 ? "unavailable" : "error";
    return response.json().catch(() => ({})).then(data => {
      const code = data && (typeof data.detail === "string" ? data.detail : (data.detail || {}).error_code);
      if (DENIED[code] === response.status) { error.code = code; error.state = "configuration-required"; error.notDispatched = true; }
      throw error;
    });
  }
  return response.json().catch(() => {
    const error = new Error("Invalid JSON response"); error.state = "error"; throw error;
  }).then(data => {
    const declaredFailure = (data.error || data.error_code || (data.errors && data.errors.length)) && data.status !== "partial";
    if (!data || ["unavailable", "error", "no-feature"].includes(data.status) || declaredFailure || (listKey && !Array.isArray(data[listKey]))) {
      const error = new Error("Invalid backend result");
      error.state = data && ["unavailable", "no-feature"].includes(data.status) ? data.status : "error";
      error.code = data && data.error_code;
      error.components = data && data.components;
      throw error;
    }
    return data;
  });
}

export function getJSON(route, listKey) {
  return fetch("/api/" + route, { headers: { Accept: "application/json" } }).then(r => classify(r, listKey));
}

export function post(route, payload, listKey) {
  const headers = { "Content-Type": "application/json", "x-ai-api-key": apiKey() };
  const body = payload === undefined ? "" : JSON.stringify(payload);
  return fetch("/api/" + route, { method: "POST", headers, body }).then(response => {
    if (listKey || !response.ok) return classify(response, listKey);
    return response.json();
  });
}

/* One explicit delivery. `retryDelivery` replays the exact body and identities while the
   session is unchanged; the caller decides whether a refused cursor deserves a restart. */
export function deliverFeed(options) {
  const payload = JSON.parse(JSON.stringify(Object.assign({ intent: { revision: 0, tag_ids: [], excluded_items: [] } }, options,
    { session_id: session(), request_id: newId(), client_request_id: newId() })));
  if (payload.cursor == null) delete payload.cursor;
  function send() {
    if (payload.session_id !== session()) {
      const error = new Error("Delivery context changed"); error.state = "error"; return Promise.reject(error);
    }
    return post("feed", payload, "items").then(data => {
      if (data.session_id && data.session_id !== payload.session_id) {
        const error = new Error("Delivery session mismatch"); error.state = "error"; throw error;
      }
      return Object.assign({}, data, { session_id: payload.session_id });
    }).catch(error => {
      if (payload.session_id === session()) error.retryDelivery = send;
      throw error;
    });
  }
  return send();
}

export function itemKey(item) { return item.kind + ":" + item.id; }
