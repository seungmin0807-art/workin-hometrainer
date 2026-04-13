const WARNING_SCORE_THRESHOLD = 76;
const VOICE_SCORE_THRESHOLD = 72;
const VOICE_MIN_GAP_MS = 8000;
const VOICE_REPEAT_GAP_MS = 15000;
const VOICE_STABLE_FRAMES = 4;
const BLINK_INTERVAL_MS = 260;

const state = {
  data: null,
  currentFrame: null,
  voiceEnabled: false,
  avatar: null,
  player: {
    isPlaying: false,
    timerId: null,
    index: 0,
    fps: 10,
  },
  voice: {
    candidateIssueId: null,
    stableCount: 0,
    lastSpokenIssueId: null,
    lastSpokenAt: 0,
  },
};

const elements = {
  overlayFrame: document.getElementById("overlayFrame"),
  currentIssue: document.getElementById("currentIssue"),
  scoreValue: document.getElementById("scoreValue"),
  playToggle: document.getElementById("playToggle"),
  voiceToggle: document.getElementById("voiceToggle"),
  phaseLabel: document.getElementById("phaseLabel"),
  averageScore: document.getElementById("averageScore"),
  repCount: document.getElementById("repCount"),
  thresholdValue: document.getElementById("thresholdValue"),
  summaryText: document.getElementById("summaryText"),
  coachText: document.getElementById("coachText"),
  metricGrid: document.getElementById("metricGrid"),
  findingList: document.getElementById("findingList"),
  timeline: document.getElementById("timeline"),
  avatarViewport: document.getElementById("avatarViewport"),
};

const metricLabels = {
  torso_lean_deg: "Torso lean",
  knee_angle: "Knee angle",
  depth_score: "Depth score",
  shin_lean_deg: "Shin lean",
};

const avatarPalette = {
  live: {
    fill0: "rgba(255, 255, 255, 0.52)",
    fill1: "rgba(255, 255, 255, 0.42)",
    fill2: "rgba(255, 255, 255, 0.34)",
    edge: "rgba(255, 255, 255, 0.22)",
    specular: "rgba(255, 255, 255, 0.22)",
    glow: "rgba(255, 255, 255, 0.08)",
    hot0: "#ffb0a5",
    hot1: "#ff5b4d",
    hotEdge: "rgba(255, 228, 222, 0.54)",
    hotGlow: "rgba(255, 87, 77, 0.28)",
    joint: "rgba(255, 255, 255, 0.58)",
  },
  reference: {
    fill0: "rgba(255, 255, 255, 0.18)",
    fill1: "rgba(255, 255, 255, 0.13)",
    fill2: "rgba(255, 255, 255, 0.08)",
    edge: "rgba(255, 255, 255, 0.12)",
    specular: "rgba(255, 255, 255, 0.08)",
    glow: "rgba(255, 255, 255, 0.04)",
    hot0: "rgba(255, 184, 176, 0.26)",
    hot1: "rgba(255, 109, 96, 0.16)",
    hotEdge: "rgba(255, 214, 175, 0.15)",
    hotGlow: "rgba(255, 170, 100, 0.08)",
    joint: "rgba(255, 255, 255, 0.22)",
  },
};

const BODY_SEGMENTS = [
  { a: "left_shoulder", b: "left_elbow", scale: 0.19 },
  { a: "left_elbow", b: "left_wrist", scale: 0.15 },
  { a: "right_shoulder", b: "right_elbow", scale: 0.19 },
  { a: "right_elbow", b: "right_wrist", scale: 0.15 },
  { a: "left_hip", b: "left_knee", scale: 0.24 },
  { a: "left_knee", b: "left_ankle", scale: 0.18 },
  { a: "right_hip", b: "right_knee", scale: 0.24 },
  { a: "right_knee", b: "right_ankle", scale: 0.18 },
  { a: "left_ankle", b: "left_foot_index", scale: 0.11 },
  { a: "right_ankle", b: "right_foot_index", scale: 0.11 },
  { a: "left_shoulder", b: "left_hip", scale: 0.18 },
  { a: "right_shoulder", b: "right_hip", scale: 0.18 },
];

function fmt(value, digits = 1, suffix = "") {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "--";
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function mix(a, b, t) {
  return a + (b - a) * t;
}

function midpoint(a, b) {
  return {
    x: (a.x + b.x) * 0.5,
    y: (a.y + b.y) * 0.5,
    scale: (a.scale + b.scale) * 0.5,
    depth: (a.depth + b.depth) * 0.5,
  };
}

function lerpPoint(a, b, t) {
  return {
    x: mix(a.x, b.x, t),
    y: mix(a.y, b.y, t),
    scale: mix(a.scale, b.scale, t),
    depth: mix(a.depth, b.depth, t),
  };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function averagePoints(points) {
  const valid = points.filter(Boolean);
  if (!valid.length) return null;
  return valid.reduce((acc, point) => ({
    x: acc.x + point.x / valid.length,
    y: acc.y + point.y / valid.length,
    scale: acc.scale + point.scale / valid.length,
    depth: acc.depth + point.depth / valid.length,
  }), { x: 0, y: 0, scale: 0, depth: 0 });
}

function getLandmarkIndex(name) {
  return state.data.landmark_index[name];
}

function getProjectedPoint(projected, name) {
  const index = getLandmarkIndex(name);
  return Number.isInteger(index) ? projected[index] : null;
}

function findNearestFrameIndex(timeSec) {
  const frames = state.data.frames;
  let low = 0;
  let high = frames.length - 1;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (frames[mid].time_sec < timeSec) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  const a = Math.max(0, high);
  const b = Math.min(frames.length - 1, low);
  return Math.abs(frames[a].time_sec - timeSec) <= Math.abs(frames[b].time_sec - timeSec) ? a : b;
}

function getFramePath(index) {
  const media = state.data.media;
  const padded = String(index).padStart(4, "0");
  return `${media.overlay_frame_dir}/${media.overlay_frame_pattern.replace("{index:04d}", padded)}`;
}

function buildTimeline() {
  elements.timeline.innerHTML = "";

  const base = document.createElement("div");
  base.className = "timeline-base";
  elements.timeline.appendChild(base);

  const duration = state.data.input_videos.wrong.duration_sec || 1;
  state.data.issue_segments.forEach((segment) => {
    const node = document.createElement("button");
    node.type = "button";
    node.className = "timeline-segment";
    node.dataset.issue = segment.id;
    node.style.left = `${(segment.start_time / duration) * 100}%`;
    node.style.width = `${Math.max(((segment.end_time - segment.start_time + 0.1) / duration) * 100, 0.6)}%`;
    node.title = `${segment.label} · ${segment.start_time.toFixed(1)}s`;
    node.addEventListener("click", () => {
      seekToTime(segment.start_time);
      play();
    });
    elements.timeline.appendChild(node);
  });

  const cursor = document.createElement("div");
  cursor.className = "timeline-cursor";
  cursor.id = "timelineCursor";
  elements.timeline.appendChild(cursor);
}

function renderFindings() {
  elements.findingList.innerHTML = "";
  state.data.overview.top_findings.forEach((item) => {
    const segment = state.data.issue_segments.find((entry) => entry.id === item.id);
    const node = document.createElement("button");
    node.type = "button";
    node.className = "finding-item";
    const suffix = item.id === "depth" ? "" : "°";
    node.innerHTML = `
      <strong>${item.label}</strong>
      <p>평균 이탈 ${fmt(item.avg_delta, 1, suffix)} · 최대 ${fmt(item.peak_delta, 1, suffix)} · ${item.frame_hits} sampled frames</p>
    `;
    node.addEventListener("click", () => {
      if (segment) {
        seekToTime(segment.start_time);
        play();
      }
    });
    elements.findingList.appendChild(node);
  });
}

function renderMetrics(frame) {
  const wrongMetrics = frame.wrong.metrics;
  const refMetrics = frame.reference.metrics;
  const rows = [
    ["torso_lean_deg", "°"],
    ["knee_angle", "°"],
    ["depth_score", ""],
    ["shin_lean_deg", "°"],
  ];

  elements.metricGrid.innerHTML = "";
  rows.forEach(([key, suffix]) => {
    const wrongValue = wrongMetrics?.[key];
    const refValue = refMetrics?.[key];
    const node = document.createElement("article");
    node.className = "metric-card";
    node.innerHTML = `
      <span>${metricLabels[key]}</span>
      <strong>${fmt(wrongValue, 1, suffix)}</strong>
      <em>ref ${fmt(refValue, 1, suffix)}</em>
    `;
    elements.metricGrid.appendChild(node);
  });
}

function ensureAvatarCanvas() {
  const canvas = document.createElement("canvas");
  canvas.className = "avatar-canvas";
  elements.avatarViewport.innerHTML = "";
  elements.avatarViewport.appendChild(canvas);
  return canvas;
}

function resizeAvatarCanvas() {
  if (!state.avatar) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(elements.avatarViewport.clientWidth * dpr));
  const height = Math.max(1, Math.floor(elements.avatarViewport.clientHeight * dpr));
  if (state.avatar.canvas.width !== width || state.avatar.canvas.height !== height) {
    state.avatar.canvas.width = width;
    state.avatar.canvas.height = height;
  }
}

function transformPoint(rawPoint, angle, tilt) {
  const x = rawPoint[0] || 0;
  const y = (rawPoint[1] || 0) - 0.14;
  const z = rawPoint[2] || 0;

  const cosY = Math.cos(angle);
  const sinY = Math.sin(angle);
  const x1 = x * cosY - z * sinY;
  const z1 = x * sinY + z * cosY;

  const cosX = Math.cos(tilt);
  const sinX = Math.sin(tilt);
  const y1 = y * cosX - z1 * sinX;
  const z2 = y * sinX + z1 * cosX;

  return { x: x1, y: y1, z: z2 };
}

function projectPoint(point, width, height) {
  const cameraZ = 4.8;
  const scale = 3.8 / (cameraZ - point.z);
  return {
    x: width * 0.5 + point.x * width * 0.24 * scale,
    y: height * 0.57 + point.y * height * 0.32 * scale,
    scale,
    depth: point.z,
  };
}

function drawBackdrop(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);

  const bg = ctx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, "rgba(9, 14, 18, 1)");
  bg.addColorStop(1, "rgba(5, 7, 11, 1)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  const halo = ctx.createRadialGradient(width * 0.5, height * 0.18, 10, width * 0.5, height * 0.18, width * 0.48);
  halo.addColorStop(0, "rgba(82, 223, 255, 0.16)");
  halo.addColorStop(1, "rgba(82, 223, 255, 0)");
  ctx.fillStyle = halo;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(82, 223, 255, 0.09)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 7; i += 1) {
    const y = height * (0.16 + i * 0.1);
    ctx.beginPath();
    ctx.moveTo(width * 0.08, y);
    ctx.lineTo(width * 0.92, y);
    ctx.stroke();
  }

  for (let i = 0; i <= 6; i += 1) {
    const x = width * (0.14 + i * 0.12);
    ctx.beginPath();
    ctx.moveTo(x, height * 0.12);
    ctx.lineTo(x, height * 0.9);
    ctx.stroke();
  }
}

function drawCapsule(ctx, start, end, width, palette, isHot = false) {
  if (!start || !end) return;

  const gradient = ctx.createLinearGradient(start.x, start.y, end.x, end.y);
  if (isHot) {
    gradient.addColorStop(0, palette.hot0);
    gradient.addColorStop(1, palette.hot1);
  } else {
    gradient.addColorStop(0, palette.fill0);
    gradient.addColorStop(0.5, palette.fill1);
    gradient.addColorStop(1, palette.fill2);
  }

  ctx.save();
  ctx.strokeStyle = gradient;
  ctx.lineCap = "round";
  ctx.lineWidth = width;
  ctx.shadowBlur = isHot ? width * 0.45 : width * 0.28;
  ctx.shadowColor = isHot ? palette.hotGlow : palette.glow;
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();

  ctx.strokeStyle = isHot ? "rgba(255, 238, 234, 0.28)" : palette.specular;
  ctx.lineWidth = Math.max(1.2, width * 0.18);
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();

  ctx.fillStyle = isHot ? palette.hot1 : palette.joint;
  ctx.beginPath();
  ctx.arc(start.x, start.y, width * 0.32, 0, Math.PI * 2);
  ctx.arc(end.x, end.y, width * 0.32, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawPolygon(ctx, points, palette, isHot = false) {
  const valid = points.filter(Boolean);
  if (valid.length < 3) return;

  const minX = Math.min(...valid.map((point) => point.x));
  const maxX = Math.max(...valid.map((point) => point.x));
  const minY = Math.min(...valid.map((point) => point.y));
  const maxY = Math.max(...valid.map((point) => point.y));

  const gradient = ctx.createLinearGradient(minX, minY, maxX, maxY);
  if (isHot) {
    gradient.addColorStop(0, palette.hot0);
    gradient.addColorStop(1, palette.hot1);
  } else {
    gradient.addColorStop(0, palette.fill0);
    gradient.addColorStop(0.55, palette.fill1);
    gradient.addColorStop(1, palette.fill2);
  }

  ctx.save();
  ctx.fillStyle = gradient;
  ctx.strokeStyle = isHot ? palette.hotEdge : palette.edge;
  ctx.lineWidth = 1.5;
  ctx.shadowBlur = isHot ? 18 : 10;
  ctx.shadowColor = isHot ? palette.hotGlow : palette.glow;
  ctx.beginPath();
  ctx.moveTo(valid[0].x, valid[0].y);
  for (let i = 1; i < valid.length; i += 1) {
    ctx.lineTo(valid[i].x, valid[i].y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawHead(ctx, headCenter, shoulderMid, radius, palette, isHot = false) {
  if (!headCenter || !shoulderMid) return;

  const headGradient = ctx.createLinearGradient(headCenter.x, headCenter.y - radius, headCenter.x, headCenter.y + radius);
  if (isHot) {
    headGradient.addColorStop(0, palette.hot0);
    headGradient.addColorStop(1, palette.hot1);
  } else {
    headGradient.addColorStop(0, palette.fill0);
    headGradient.addColorStop(0.55, palette.fill1);
    headGradient.addColorStop(1, palette.fill2);
  }

  ctx.save();
  ctx.fillStyle = headGradient;
  ctx.strokeStyle = isHot ? palette.hotEdge : palette.edge;
  ctx.lineWidth = 1.4;
  ctx.shadowBlur = isHot ? 18 : 12;
  ctx.shadowColor = isHot ? palette.hotGlow : palette.glow;
  ctx.beginPath();
  ctx.ellipse(headCenter.x, headCenter.y, radius * 0.82, radius, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.strokeStyle = palette.specular;
  ctx.lineWidth = Math.max(1, radius * 0.12);
  ctx.beginPath();
  ctx.moveTo(headCenter.x, headCenter.y - radius * 0.55);
  ctx.lineTo(headCenter.x, headCenter.y + radius * 0.28);
  ctx.stroke();
  ctx.restore();

  drawCapsule(ctx, shoulderMid, { x: headCenter.x, y: headCenter.y + radius * 0.86 }, radius * 0.32, palette, isHot);
}

function isSegmentHot(highlightedIndices, ...names) {
  return names.some((name) => highlightedIndices.has(getLandmarkIndex(name)));
}

function buildProjectedLandmarks(landmarks, width, height) {
  const angle = 0.3;
  const tilt = -0.18;
  return landmarks.map((point) => projectPoint(transformPoint(point, angle, tilt), width, height));
}

function drawMannequin(ctx, landmarks, highlightedIndices, palette, width, height, isGhost = false) {
  if (!Array.isArray(landmarks) || !landmarks.length) return;

  const projected = buildProjectedLandmarks(landmarks, width, height);
  const leftShoulder = getProjectedPoint(projected, "left_shoulder");
  const rightShoulder = getProjectedPoint(projected, "right_shoulder");
  const leftHip = getProjectedPoint(projected, "left_hip");
  const rightHip = getProjectedPoint(projected, "right_hip");
  const shoulderMid = averagePoints([leftShoulder, rightShoulder]);
  const hipMid = averagePoints([leftHip, rightHip]);
  if (!shoulderMid || !hipMid) return;

  const bodyScale = clamp(distance(leftShoulder, rightShoulder), width * 0.12, width * 0.21);
  const blinkOn = !isGhost && state.avatar ? state.avatar.blinkVisible : true;
  const leftChest = lerpPoint(leftShoulder, leftHip, 0.28);
  const rightChest = lerpPoint(rightShoulder, rightHip, 0.28);
  const leftWaist = lerpPoint(leftShoulder, leftHip, 0.72);
  const rightWaist = lerpPoint(rightShoulder, rightHip, 0.72);
  const sternum = lerpPoint(shoulderMid, hipMid, 0.34);
  const pelvis = lerpPoint(shoulderMid, hipMid, 0.88);
  const headAnchor = averagePoints([
    getProjectedPoint(projected, "nose"),
    getProjectedPoint(projected, "left_eye"),
    getProjectedPoint(projected, "right_eye"),
    getProjectedPoint(projected, "left_ear"),
    getProjectedPoint(projected, "right_ear"),
  ]) || {
    x: shoulderMid.x,
    y: shoulderMid.y - bodyScale * 1.08,
    scale: shoulderMid.scale,
    depth: shoulderMid.depth - 0.1,
  };
  const headRadius = bodyScale * 0.28;

  ctx.save();
  if (isGhost) {
    ctx.globalAlpha = 0.82;
  }

  const torsoHot = blinkOn && isSegmentHot(highlightedIndices, "left_shoulder", "right_shoulder", "left_hip", "right_hip");
  drawPolygon(ctx, [leftShoulder, rightShoulder, rightChest, leftChest], palette, torsoHot);
  drawPolygon(ctx, [leftChest, rightChest, rightWaist, pelvis, leftWaist], palette, torsoHot);
  drawPolygon(ctx, [leftWaist, rightWaist, rightHip, leftHip], palette, torsoHot);

  BODY_SEGMENTS.forEach((segment) => {
    const start = getProjectedPoint(projected, segment.a);
    const end = getProjectedPoint(projected, segment.b);
    const segmentHot = blinkOn && isSegmentHot(highlightedIndices, segment.a, segment.b);
    drawCapsule(ctx, start, end, bodyScale * segment.scale, palette, segmentHot);
  });

  drawCapsule(ctx, shoulderMid, sternum, bodyScale * 0.14, palette, torsoHot);
  drawHead(ctx, headAnchor, shoulderMid, headRadius, palette, torsoHot);
  ctx.restore();
}

function renderAvatar() {
  if (!state.avatar) return;

  resizeAvatarCanvas();
  const { canvas, ctx } = state.avatar;
  const width = canvas.width;
  const height = canvas.height;
  if (!width || !height) return;

  drawBackdrop(ctx, width, height);
  drawMannequin(ctx, state.avatar.referenceLandmarks, new Set(), avatarPalette.reference, width, height, true);
  drawMannequin(ctx, state.avatar.wrongLandmarks, state.avatar.highlightedIndices, avatarPalette.live, width, height, false);

  ctx.fillStyle = "rgba(245, 246, 247, 0.92)";
  ctx.font = `${Math.max(12, Math.round(height * 0.032))}px "Space Grotesk", sans-serif`;
  ctx.fillText("Mannequin pose twin", width * 0.06, height * 0.08);
  ctx.fillStyle = "rgba(155, 167, 179, 0.92)";
  ctx.font = `${Math.max(11, Math.round(height * 0.024))}px "Pretendard", sans-serif`;
  ctx.fillText("solid body = wrong, ghost body = correct", width * 0.06, height * 0.135);
}

function initAvatar() {
  const canvas = ensureAvatarCanvas();
  const ctx = canvas.getContext("2d");
  state.avatar = {
    canvas,
    ctx,
    wrongLandmarks: null,
    referenceLandmarks: null,
    highlightedIndices: new Set(),
    blinkVisible: true,
    blinkTimerId: null,
  };

  window.addEventListener("resize", renderAvatar);
  state.avatar.blinkTimerId = window.setInterval(() => {
    if (!state.avatar) return;
    state.avatar.blinkVisible = !state.avatar.blinkVisible;
    if (state.avatar.highlightedIndices.size) {
      renderAvatar();
    }
  }, BLINK_INTERVAL_MS);
  resizeAvatarCanvas();
  renderAvatar();
}

function resetVoiceTracking() {
  state.voice.candidateIssueId = null;
  state.voice.stableCount = 0;
}

function maybeSpeak(frame, warningIssue) {
  if (!state.voiceEnabled || !warningIssue || !("speechSynthesis" in window)) return;

  if (frame.score > VOICE_SCORE_THRESHOLD) {
    resetVoiceTracking();
    return;
  }

  const issueId = warningIssue.id;
  if (state.voice.candidateIssueId === issueId) {
    state.voice.stableCount += 1;
  } else {
    state.voice.candidateIssueId = issueId;
    state.voice.stableCount = 1;
  }

  if (state.voice.stableCount < VOICE_STABLE_FRAMES) return;
  if (window.speechSynthesis.speaking) return;

  const now = Date.now();
  if (now - state.voice.lastSpokenAt < VOICE_MIN_GAP_MS) return;
  if (state.voice.lastSpokenIssueId === issueId && now - state.voice.lastSpokenAt < VOICE_REPEAT_GAP_MS) return;

  const utterance = new SpeechSynthesisUtterance(frame.coach_text);
  utterance.lang = "ko-KR";
  utterance.rate = 0.98;
  window.speechSynthesis.speak(utterance);

  state.voice.lastSpokenIssueId = issueId;
  state.voice.lastSpokenAt = now;
  state.voice.stableCount = 0;
}

function updateFrame(frame) {
  if (!frame) return;
  state.currentFrame = frame;

  const topIssue = (frame.issues || [])[0] || null;
  const warningIssue = topIssue && frame.score <= WARNING_SCORE_THRESHOLD ? topIssue : null;

  elements.currentIssue.textContent = warningIssue ? warningIssue.label : "기준 자세 범위";
  elements.scoreValue.textContent = `${Math.round(frame.score)}`;
  elements.phaseLabel.textContent = `Rep ${frame.rep_index} · ${frame.phase}`;
  elements.coachText.textContent = warningIssue
    ? frame.coach_text
    : "현재 프레임은 기준 스쿼트 범위 안에 있어 음성 안내를 쉬고 있습니다.";

  renderMetrics(frame);

  const cursor = document.getElementById("timelineCursor");
  const duration = state.data.input_videos.wrong.duration_sec || 1;
  if (cursor) {
    cursor.style.left = `${(frame.time_sec / duration) * 100}%`;
  }

  const highlightedIndices = warningIssue
    ? new Set(
        (frame.highlighted_joint_names || [])
          .flatMap((name) => [
            state.data.landmark_index[name],
            state.data.landmark_index[`left_${name}`],
            state.data.landmark_index[`right_${name}`],
          ])
          .filter((value) => Number.isInteger(value)),
      )
    : new Set();

  if (state.avatar) {
    state.avatar.wrongLandmarks = frame.wrong.world_landmarks;
    state.avatar.referenceLandmarks = frame.reference.world_landmarks;
    state.avatar.highlightedIndices = highlightedIndices;
    renderAvatar();
  }

  maybeSpeak(frame, warningIssue);
}

function showFrame(index) {
  const clamped = Math.max(0, Math.min(index, state.data.frames.length - 1));
  state.player.index = clamped;
  elements.overlayFrame.src = getFramePath(clamped);
  updateFrame(state.data.frames[clamped]);
}

function pause() {
  if (state.player.timerId) {
    clearInterval(state.player.timerId);
    state.player.timerId = null;
  }
  state.player.isPlaying = false;
  elements.playToggle.textContent = "재생";
}

function play() {
  if (state.player.isPlaying) return;
  state.player.isPlaying = true;
  elements.playToggle.textContent = "일시정지";

  const intervalMs = 1000 / Math.max(state.player.fps, 1);
  state.player.timerId = window.setInterval(() => {
    if (state.player.index >= state.data.frames.length - 1) {
      pause();
      return;
    }
    showFrame(state.player.index + 1);
  }, intervalMs);
}

function seekToTime(timeSec) {
  const index = findNearestFrameIndex(timeSec);
  showFrame(index);
}

function bindControls() {
  elements.playToggle.addEventListener("click", () => {
    if (state.player.isPlaying) {
      pause();
    } else {
      play();
    }
  });

  elements.voiceToggle.addEventListener("click", () => {
    state.voiceEnabled = !state.voiceEnabled;
    elements.voiceToggle.textContent = state.voiceEnabled ? "음성 피드백 켬" : "음성 피드백 끔";
    if (!state.voiceEnabled && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      resetVoiceTracking();
    }
  });
}

async function bootstrap() {
  const response = await fetch("data/analysis.json");
  if (!response.ok) {
    throw new Error(`Failed to load analysis.json (${response.status})`);
  }

  state.data = await response.json();
  state.player.fps = state.data.input_videos.wrong.sampled_fps || 10;
  const findingLabels = state.data.overview.top_findings.map((item) => item.label).join(" / ");

  elements.averageScore.textContent = `${Math.round(state.data.overview.average_score)}`;
  elements.repCount.textContent = `${state.data.overview.rep_count}`;
  elements.thresholdValue.textContent = `${WARNING_SCORE_THRESHOLD}`;
  elements.summaryText.textContent = `주요 이탈: ${findingLabels}.`;

  buildTimeline();
  renderFindings();
  initAvatar();
  bindControls();
  showFrame(0);
}

bootstrap().catch((error) => {
  console.error(error);
  elements.summaryText.textContent = error.message;
  elements.coachText.textContent = "프로토타입 로드에 실패했습니다.";
});
