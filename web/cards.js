/* Shared card and media rendering. Media is the folder's own file; a video element or
   an image, never a generated thumbnail. */
import { explanation } from "./explain.js";
import { feedbackControls } from "./feedback.js";
import { attachPlayer } from "./watch.js";
import { formatTime } from "./state.js";

export function media(item, { autoplay = false, controls = true } = {}) {
  const box = document.createElement("div");
  box.className = "thumb";
  if (!item.media_url) {
    const placeholder = document.createElement("span"); placeholder.className = "placeholder"; placeholder.textContent = "No media file"; box.appendChild(placeholder);
    return box;
  }
  if (item.kind === "video") {
    const video = document.createElement("video");
    video.src = item.media_url; video.controls = controls; video.preload = "metadata"; video.muted = autoplay; video.playsInline = true;
    video.setAttribute("aria-label", item.title || ("Item " + item.id));
    attachPlayer(video, item);
    box.appendChild(video);
  } else {
    const image = document.createElement("img");
    image.src = item.media_url; image.alt = item.title || ""; image.loading = "lazy"; image.width = 640; image.height = 360;
    box.appendChild(image);
  }
  return box;
}

const SIDECAR_REASONS = { sidecar_path_too_long: "sidecar not read: file name too long", sidecar_invalid: "sidecar not read: invalid JSON",
  sidecar_unreadable: "sidecar not read" };

export function metaLine(item) {
  const meta = document.createElement("div");
  meta.className = "card-meta";
  const bits = [item.kind];
  if (typeof item.duration === "number" && item.duration > 0) bits.push(formatTime(item.duration));
  else if (typeof item.duration_s === "number" && item.duration_s > 0) bits.push(formatTime(item.duration_s));
  else bits.push("duration not measured");
  if (item.sidecar_reason) bits.push(SIDECAR_REASONS[item.sidecar_reason] || "sidecar not read");
  if (item.category && item.category !== item.kind) bits.push(item.category);
  if (item.explore) bits.push("exploration");
  if (item.control) bits.push("control");
  bits.forEach(text => { const span = document.createElement("span"); span.textContent = text; meta.appendChild(span); });
  return meta;
}

export function gridCard(item, { withFeedback = true, seedAction = null } = {}) {
  const card = document.createElement("article");
  card.className = "card";
  card.setAttribute("data-ai-home-key", item.kind + ":" + item.id);
  card.appendChild(media(item));
  const body = document.createElement("div");
  body.className = "body";
  const title = document.createElement("h3"); title.textContent = item.title || ("Item " + item.id);
  const reason = document.createElement("p"); reason.className = "reason";
  reason.textContent = item.reason || (item.search && item.search.best_t != null ? "Matching moment " + formatTime(item.search.best_t) : item.search ? "Matches your words; position not measured" : item.similar ? "Shares tags with the seed" : "Serving explanation unavailable");
  body.append(title, reason, metaLine(item));
  if (seedAction) {
    const button = document.createElement("button"); button.type = "button"; button.textContent = "Find similar"; button.onclick = () => seedAction(item); body.appendChild(button);
  }
  if (withFeedback) body.appendChild(feedbackControls(item));
  body.appendChild(explanation(item));
  card.appendChild(body);
  return card;
}
