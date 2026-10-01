/* Shared card and media rendering. Media is the folder's own file, never a generated
   thumbnail: a video shows the frame at its matching moment through a media fragment.
   A grid card previews muted on hover and becomes the real player on click; only the
   real player is attached to watch capture, because a hover preview is not watching.
   Videos get their source only when they come near the viewport; still images open a
   full-size viewer. */
import { evidenceStrip, explanation, strongestSignal } from "./explain.js";
import { feedbackControls } from "./feedback.js";
import { attachPlayer } from "./watch.js";
import { formatTime } from "./state.js";
import { icon } from "./icons.js";

const PREVIEW_DELAY_MS = 230;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const KINDS = { video: "Video", image: "Image" };

function sentence(text) { const value = String(text).replace(/_/g, " "); return value.charAt(0).toUpperCase() + value.slice(1); }

function durationOf(item) {
  if (typeof item.duration === "number" && item.duration > 0) return item.duration;
  if (typeof item.duration_s === "number" && item.duration_s > 0) return item.duration_s;
  return null;
}

/* The served matching moment when there is one; otherwise a frame a tenth of the way in,
   which is only where the still is taken from, never a claim about the item. A moment at
   or past the end of the clip opens at 0. */
function startOf(item) {
  const moment = item.best_t != null ? item.best_t : item.search && item.search.best_t;
  const duration = durationOf(item);
  if (typeof moment === "number" && moment >= 0) return duration && moment >= duration ? 0 : moment;
  return duration ? Math.round(duration * 10) / 100 : 0;
}

function titleOf(item) { return item.title || ("Item " + item.id); }

function reasonText(item, video) {
  if (item.reason) return item.reason;
  if (item.search) {
    const moment = item.search.best_t;
    const duration = durationOf(item) || (video && Number.isFinite(video.duration) && video.duration > 0 ? video.duration : null);
    return moment != null && !(duration && moment >= duration) ? "Matching moment " + formatTime(moment) : "Matches your words";
  }
  if (item.similar) return strongestSignal(item) || "Similar to this item";
  return "No reason was recorded for this pick.";
}

/* A moment at or past the clip's end is dropped, the same rule startOf uses for the
   start. When the duration is only known once the player's metadata loads, relabel then. */
export function whyText(item, reasonEl, video) {
  if (reasonEl && video && item.search && !item.reason && !durationOf(item)) {
    video.addEventListener("loadedmetadata", () => {
      reasonEl.textContent = reasonText(item, video);
      reasonEl.title = reasonEl.textContent;
    }, { once: true });
  }
  return reasonText(item, video);
}

export function similarHref(item) {
  return "#/similar?" + new URLSearchParams({ kind: item.kind, id: String(item.id), title: titleOf(item) });
}

export function similarLink(item, compact) {
  const link = document.createElement("a");
  link.className = "action";
  link.href = similarHref(item);
  link.appendChild(icon("similar"));
  const text = document.createElement("span"); text.textContent = "More like this";
  if (compact) { text.className = "sr-only"; link.title = "More like this"; }
  link.appendChild(text);
  return link;
}

/* One observer per scroll root assigns a video its source once it is within 600 px. */
let viewportObserver = null;
function nearSource(video, url, root) {
  const load = () => { if (!video.getAttribute("src")) video.src = url; };
  if (!("IntersectionObserver" in window)) { load(); return; }
  let observer = root ? root.__nearObserver : viewportObserver;
  if (!observer) {
    observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      observer.unobserve(entry.target);
      entry.target.__loadSource();
    }), { root: root || null, rootMargin: "600px" });
    if (root) root.__nearObserver = observer; else viewportObserver = observer;
  }
  video.__loadSource = load;
  observer.observe(video);
}

/* A view being left stops its players: the stream closes now, while the player still knows
   its position, and the source is dropped so detached media cannot keep playing. */
export function stopVideos(root) {
  root.querySelectorAll("video").forEach(video => {
    video.pause();
    if (video.__closeWatch) video.__closeWatch();
    video.removeAttribute("src");
    video.load();
  });
}

/* A host preview clip plays from its start; the item itself opens at the matching moment. */
function openVideo(video, item, root, preview) {
  video.__start = preview ? 0 : startOf(item);
  video.preload = "metadata"; video.playsInline = true;
  video.addEventListener("loadedmetadata", () => {
    if (video.__start > 0 && video.__start >= video.duration) { video.__start = 0; video.currentTime = 0; }
  });
  nearSource(video, preview || item.media_url + "#t=" + video.__start, root);
}

function image(item, eager) {
  const img = document.createElement("img");
  img.src = item.media_url; img.alt = titleOf(item); img.width = 640; img.height = 360; img.decoding = "async";
  if (eager) { img.loading = "eager"; img.fetchPriority = "high"; } else img.loading = "lazy";
  return img;
}

function placeholder(box) {
  const span = document.createElement("span"); span.className = "placeholder"; span.textContent = "No media file"; box.appendChild(span);
  return box;
}

/* The Feed player: controls on, attached to watch capture, opening at the matching moment. */
export function media(item, { root = null, eager = false } = {}) {
  const box = document.createElement("div");
  box.className = "thumb";
  if (!item.media_url) return placeholder(box);
  if (item.kind === "video" && !item.animated_image) {
    const video = document.createElement("video");
    video.controls = true;
    video.setAttribute("aria-label", titleOf(item));
    openVideo(video, item, root);
    attachPlayer(video, item);
    box.appendChild(video);
  } else box.appendChild(image(item, eager));
  return box;
}

let viewer = null, viewerOpener = null;
export function openViewer(item, opener) {
  if (!viewer) {
    viewer = document.createElement("dialog");
    viewer.className = "viewer";
    const close = document.createElement("button");
    close.type = "button"; close.className = "viewer-close"; close.setAttribute("aria-label", "Close");
    close.appendChild(icon("close"));
    close.onclick = () => viewer.close();
    const picture = document.createElement("img");
    const caption = document.createElement("p"); caption.className = "viewer-caption";
    viewer.append(close, picture, caption);
    viewer.addEventListener("click", event => { if (event.target === viewer) viewer.close(); });
    viewer.addEventListener("close", () => { if (viewerOpener && viewerOpener.isConnected) viewerOpener.focus(); });
    document.body.appendChild(viewer);
  }
  viewerOpener = opener;
  const picture = viewer.querySelector("img");
  picture.src = item.media_url; picture.alt = titleOf(item);
  viewer.querySelector(".viewer-caption").textContent = titleOf(item);
  viewer.setAttribute("aria-label", titleOf(item));
  viewer.showModal();
}

function durationBadge(box, seconds) {
  let badge = box.querySelector(".badge");
  if (!badge) { badge = document.createElement("span"); badge.className = "badge"; box.appendChild(badge); }
  badge.textContent = formatTime(seconds);
}

function thumb(item, eager) {
  const box = document.createElement("div");
  box.className = "thumb";
  if (!item.media_url) return placeholder(box);
  const page = item.open_url;
  const opener = document.createElement(page ? "a" : "button");
  if (page) opener.href = page; else opener.type = "button";
  opener.className = "thumb-open";
  if (item.kind !== "video" || item.animated_image) {
    box.appendChild(image(item, eager));
    const chip = document.createElement("span");
    chip.className = "play-chip view-chip";
    chip.setAttribute("aria-hidden", "true");
    chip.append(icon(item.animated_image ? "play" : "view"), "View");
    opener.appendChild(chip);
    if (page) opener.setAttribute("aria-label", "Open " + titleOf(item));
    else {
      opener.setAttribute("aria-label", "View image");
      opener.onclick = () => openViewer(item, opener);
    }
    box.appendChild(opener);
    return box;
  }
  const video = document.createElement("video");
  video.muted = true; video.tabIndex = -1;
  video.setAttribute("aria-label", titleOf(item));
  const preview = item.preview_url;
  openVideo(video, item, null, preview);
  box.appendChild(video);
  const duration = durationOf(item);
  if (duration) durationBadge(box, duration);
  else if (!preview) video.addEventListener("loadedmetadata", () => { if (Number.isFinite(video.duration)) durationBadge(box, video.duration); }, { once: true });
  const chip = document.createElement("span"); chip.className = "play-chip"; chip.setAttribute("aria-hidden", "true");
  chip.append(icon("play"), "Play");
  opener.appendChild(chip);
  opener.setAttribute("aria-label", (page ? "Open " : "Play ") + titleOf(item));
  box.appendChild(opener);
  let timer = null, live = false;
  const startPreview = () => {
    if (live || timer || reducedMotion.matches || !video.getAttribute("src")) return;
    timer = setTimeout(() => { video.loop = true; video.play().catch(() => {}); }, PREVIEW_DELAY_MS);
  };
  const stopPreview = () => {
    clearTimeout(timer); timer = null;
    if (!live) { video.pause(); if (video.readyState) video.currentTime = video.__start; }
  };
  box.addEventListener("mouseenter", startPreview);
  box.addEventListener("mouseleave", stopPreview);
  if (preview) { opener.addEventListener("focus", startPreview); opener.addEventListener("blur", stopPreview); }
  if (page) return box;
  opener.onclick = () => {
    if (live) return;
    live = true; clearTimeout(timer);
    if (preview) { video.__start = startOf(item); video.src = item.media_url + "#t=" + video.__start; }
    else if (!video.getAttribute("src")) video.__loadSource();
    video.pause(); video.loop = false; if (video.readyState) video.currentTime = video.__start;
    video.muted = false; video.controls = true; video.tabIndex = 0;
    box.classList.add("is-live"); opener.remove();
    attachPlayer(video, item);
    video.focus();
    video.play().catch(() => {});
  };
  return box;
}

const SIDECAR_REASONS = { sidecar_path_too_long: "Sidecar not read: file name too long", sidecar_invalid: "Sidecar not read: invalid JSON",
  sidecar_unreadable: "Sidecar not read" };

/* One quiet line: kind, duration (from the sidecar, else from the player once it knows),
   category. Still images have no duration. Flags that need attention follow. */
export function metaLine(item, video) {
  const meta = document.createElement("p");
  meta.className = "card-meta";
  function part(text, cls) { const span = document.createElement("span"); span.textContent = text; if (cls) span.className = cls; meta.appendChild(span); return span; }
  const kind = part(KINDS[item.kind] || sentence(item.kind));
  if (item.kind === "video" && !item.animated_image) {
    const duration = durationOf(item);
    if (duration) part(formatTime(duration), "num");
    else if (video) video.addEventListener("loadedmetadata", () => {
      if (!Number.isFinite(video.duration)) return;
      const span = document.createElement("span"); span.className = "num"; span.textContent = formatTime(video.duration); kind.after(span);
    }, { once: true });
  }
  if (item.category && item.category !== item.kind) part(sentence(item.category));
  if (item.explore) part("Exploration pick");
  if (item.control) part("Control pick");
  if (item.sidecar_reason) part(SIDECAR_REASONS[item.sidecar_reason] || "Sidecar not read", "meta-flag");
  return meta;
}

/* Home shelves use h3 under the shelf h2; Search and Similar use h2 under the view h1. */
export function gridCard(item, { headingLevel = 2, eager = false } = {}) {
  const card = document.createElement("article");
  card.className = "card";
  card.setAttribute("data-ai-home-key", item.kind + ":" + item.id);
  const box = thumb(item, eager);
  card.appendChild(box);
  const body = document.createElement("div");
  body.className = "body";
  const title = document.createElement("h" + headingLevel); title.className = "card-heading"; title.textContent = titleOf(item); title.title = titleOf(item);
  const why = explanation(item);
  const reason = document.createElement("p"); reason.className = "reason"; reason.textContent = whyText(item, reason, item.preview_url ? null : box.querySelector("video")); reason.title = reason.textContent;
  const extras = item.kind === "video" ? [similarLink(item, true)] : [];
  body.append(title, evidenceStrip(item, why), reason, metaLine(item, item.preview_url ? null : box.querySelector("video")), feedbackControls(item, { extras, compact: true }), why);
  card.appendChild(body);
  return card;
}
