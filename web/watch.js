/* Measured presentation and player watch capture. A card counts as viewed after the
   visibility policy (60 percent visible for 1,200 ms in the foreground); a playing
   video posts committed progress batches bound to that qualified view. */
import { newId, post, session } from "./api.js";

export const VISIBILITY_POLICY = "foreground-60pct-1200ms-v1";

export function observeView(element, item, position, surface, onViewed) {
  if (!item.request_id || !item.served_item_id || !("IntersectionObserver" in window)) return null;
  let timer = null, done = false, disposed = false;
  function cancel() { if (timer !== null) { clearTimeout(timer); timer = null; } }
  function onVisibility() { if (document.visibilityState !== "visible") cancel(); }
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (done || disposed) return;
      if (entry.isIntersecting && entry.intersectionRatio >= 0.6 && document.visibilityState === "visible") {
        if (timer === null) timer = setTimeout(() => {
          timer = null;
          if (disposed || !element.isConnected || document.visibilityState !== "visible") return;
          done = true; observer.disconnect(); document.removeEventListener("visibilitychange", onVisibility);
          post("view", { client_event_id: newId(), session_id: session(), request_id: item.request_id, served_item_id: item.served_item_id,
            surface, position, dwell_ms: 1200, visible_fraction: 0.6, visibility_policy: VISIBILITY_POLICY, kind: item.kind, item_id: item.id,
            occurred_at: Date.now() / 1000 }).then(result => {
            if (result && result.status === "confirmed") { item.viewed_event_id = result.event_id; element.setAttribute("data-ai-viewed", result.event_id); if (onViewed) onViewed(result); }
            else element.setAttribute("data-ai-viewed", "unconfirmed:" + ((result && result.error_code) || "no_result"));
          }).catch(error => element.setAttribute("data-ai-viewed", "unconfirmed:" + ((error && error.code) || error.state || "transport")));
        }, 1200);
      } else cancel();
    });
  }, { threshold: [0, 0.6, 1] });
  observer.observe(element);
  document.addEventListener("visibilitychange", onVisibility);
  return { disconnect() { disposed = true; cancel(); observer.disconnect(); document.removeEventListener("visibilitychange", onVisibility); } };
}

/* One stream per playback: start once, then progress every few seconds and on pause/end.
   Each batch is posted with the positions the player reported; nothing is estimated. */
export function attachPlayer(video, item) {
  let stream = null, previous = null, sent = 0;
  function event(type) {
    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : null;
    if (duration === null || !item.viewed_event_id) return null;
    const row = { id: newId(), type, item_id: item.id, occurred_at: Date.now() / 1000, position: Math.min(video.currentTime, duration), duration,
      viewed_event_id: item.viewed_event_id, previous_event_id: previous, playback_rate: video.playbackRate || 1 };
    previous = row.id;
    return row;
  }
  function flush(rows) {
    if (!rows.length) return;
    post("watch", { capture_id: newId(), stream_session_id: stream, session_id: session(), events: rows }).then(result => {
      video.setAttribute("data-ai-watch", result && (result.status === "imported" || result.status === "duplicate") ? result.status : "unconfirmed");
    }).catch(() => video.setAttribute("data-ai-watch", "unconfirmed"));
  }
  video.addEventListener("play", () => {
    if (!stream) { stream = "stream-" + newId(); const row = event("view_start"); if (row) flush([row]); }
  });
  let last = 0;
  video.addEventListener("timeupdate", () => {
    if (!stream || video.currentTime - last < 4) return;
    last = video.currentTime; sent++;
    const row = event("view_progress"); if (row) flush([row]);
  });
  ["pause", "ended"].forEach(name => video.addEventListener(name, () => {
    if (!stream) return;
    const row = event(name === "ended" ? "view_complete" : "view_pause"); if (row) flush([row]);
  }));
}
