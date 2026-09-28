/* Shared card and media rendering. Media is the folder's own file, never a generated
   thumbnail: a video shows the frame at its matching moment through a media fragment.
   A grid card previews muted on hover and becomes the real player on click; only the
   real player is attached to watch capture, because a hover preview is not watching. */
import { explanation } from "./explain.js";
import { feedbackControls } from "./feedback.js";
import { attachPlayer } from "./watch.js";
import { formatTime } from "./state.js";

const PREVIEW_DELAY_MS = 230;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function durationOf(item) {
  if (typeof item.duration === "number" && item.duration > 0) return item.duration;
  if (typeof item.duration_s === "number" && item.duration_s > 0) return item.duration_s;
  return null;
}

/* The served matching moment when there is one; otherwise a frame a tenth of the way in,
   which is only where the still is taken from, never a claim about the item. */
function startOf(item) {
  const moment = item.best_t != null ? item.best_t : item.search && item.search.best_t;
  if (typeof moment === "number" && moment >= 0) return moment;
  const duration = durationOf(item);
  return duration ? Math.round(duration * 10) / 100 : 0;
}

function titleOf(item) { return item.title || ("Item " + item.id); }

export function whyText(item) {
  if (item.reason) return item.reason;
  if (item.search) return item.search.best_t != null ? "Matching moment " + formatTime(item.search.best_t) : "Matches your words; position not measured";
  if (item.similar) return "Shares tags with the seed";
  return "Serving explanation unavailable";
}

function placeholder(box) {
  const span = document.createElement("span"); span.className = "placeholder"; span.textContent = "No media file"; box.appendChild(span);
  return box;
}

function badge(cls, text) { const span = document.createElement("span"); span.className = "badge " + cls; span.textContent = text; return span; }

/* The Feed player: controls on, attached to watch capture, opening at the matching moment. */
export function media(item) {
  const box = document.createElement("div");
  box.className = "thumb";
  if (!item.media_url) return placeholder(box);
  if (item.kind === "video") {
    const video = document.createElement("video");
    video.src = item.media_url + "#t=" + startOf(item); video.controls = true; video.preload = "metadata"; video.playsInline = true;
    video.setAttribute("aria-label", titleOf(item));
    attachPlayer(video, item);
    box.appendChild(video);
  } else {
    const image = document.createElement("img");
    image.src = item.media_url; image.alt = titleOf(item); image.loading = "lazy"; image.width = 640; image.height = 360;
    box.appendChild(image);
  }
  return box;
}

function thumb(item) {
  const box = document.createElement("div");
  box.className = "thumb";
  if (!item.media_url) return placeholder(box);
  if (item.kind !== "video") {
    const image = document.createElement("img");
    image.src = item.media_url; image.alt = titleOf(item); image.loading = "lazy"; image.width = 640; image.height = 360;
    box.appendChild(image);
    return box;
  }
  const start = startOf(item);
  const video = document.createElement("video");
  video.src = item.media_url + "#t=" + start; video.preload = "metadata"; video.muted = true; video.playsInline = true; video.tabIndex = -1;
  video.setAttribute("aria-label", titleOf(item));
  box.appendChild(video);
  const duration = durationOf(item);
  if (duration) box.appendChild(badge("duration", formatTime(duration)));
  box.appendChild(badge("play", "\u25B6 Play"));
  box.tabIndex = 0; box.setAttribute("role", "button"); box.setAttribute("aria-label", "Play " + titleOf(item));
  let timer = null, live = false;
  box.addEventListener("mouseenter", () => {
    if (live || reducedMotion.matches) return;
    timer = setTimeout(() => { video.loop = true; video.play().catch(() => {}); }, PREVIEW_DELAY_MS);
  });
  box.addEventListener("mouseleave", () => {
    clearTimeout(timer); timer = null;
    if (!live) { video.pause(); video.currentTime = start; }
  });
  function play() {
    if (live) return;
    live = true; clearTimeout(timer);
    video.pause(); video.loop = false; video.currentTime = start; video.muted = false; video.controls = true; video.tabIndex = 0;
    box.classList.add("is-live"); box.removeAttribute("role"); box.removeAttribute("tabindex"); box.removeAttribute("aria-label");
    attachPlayer(video, item);
    video.focus();
    video.play().catch(() => {});
  }
  box.addEventListener("click", play);
  box.addEventListener("keydown", event => { if (!live && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); play(); } });
  return box;
}

const SIDECAR_REASONS = { sidecar_path_too_long: "sidecar not read: file name too long", sidecar_invalid: "sidecar not read: invalid JSON",
  sidecar_unreadable: "sidecar not read" };

export function metaLine(item) {
  const meta = document.createElement("div");
  meta.className = "card-meta";
  const duration = durationOf(item);
  const bits = [[item.kind], [duration ? formatTime(duration) : "duration not measured"]];
  if (typeof item.score === "number" && Number.isFinite(item.score)) bits.push(["score " + item.score.toFixed(2), "meta-score"]);
  if (item.category && item.category !== item.kind) bits.push([item.category]);
  if (item.sidecar_reason) bits.push([SIDECAR_REASONS[item.sidecar_reason] || "sidecar not read", "meta-flag"]);
  if (item.explore) bits.push(["exploration", "meta-flag"]);
  if (item.control) bits.push(["control", "meta-flag"]);
  bits.forEach(([text, cls]) => { const span = document.createElement("span"); span.textContent = text; if (cls) span.className = cls; meta.appendChild(span); });
  return meta;
}

export function gridCard(item, { withFeedback = true, seedAction = null } = {}) {
  const card = document.createElement("article");
  card.className = "card";
  card.setAttribute("data-ai-home-key", item.kind + ":" + item.id);
  card.appendChild(thumb(item));
  const body = document.createElement("div");
  body.className = "body";
  const title = document.createElement("h3"); title.textContent = titleOf(item); title.title = titleOf(item);
  const reason = document.createElement("p"); reason.className = "reason"; reason.textContent = whyText(item); reason.title = reason.textContent;
  body.append(title, reason, metaLine(item));
  if (seedAction) {
    const actions = document.createElement("div"); actions.className = "card-actions";
    const button = document.createElement("button"); button.type = "button"; button.textContent = "More like this"; button.onclick = () => seedAction(item);
    actions.appendChild(button); body.appendChild(actions);
  }
  if (withFeedback) body.appendChild(feedbackControls(item));
  body.appendChild(explanation(item));
  card.appendChild(body);
  return card;
}
