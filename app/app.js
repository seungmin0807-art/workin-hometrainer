const WARNING_SCORE_THRESHOLD = 76;
const VOICE_SCORE_THRESHOLD = 72;
const VOICE_MIN_GAP_MS = 8000;
const VOICE_REPEAT_GAP_MS = 15000;
const VOICE_STABLE_FRAMES = 4;
const BLINK_INTERVAL_MS = 280;

const state = {
  data: null,
  scoreBuckets: new Map(),
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

const ISSUE_HOT_NAMES = {
  knee_flexion: (side) => [`${side}_hip`, `${side}_knee`, `${side}_ankle`],
  shin_lean: (side) => [`${side}_knee`, `${side}_ankle`, `${side}_heel`, `${side}_foot_index`],
  torso_lean: (side) => [`${side}_shoulder`, `${side}_hip`],
  depth: () => ["left_hip", "right_hip", "left_knee", "right_knee"],
};

const avatarPalette = {
  live: {
    fill0: "rgba(255, 255, 255, 0.42)",
    fill1: "rgba(255, 255, 255, 0.34)",
    fill2: "rgba(255, 255, 255, 0.28)",
    edge: "rgba(255, 255, 255, 0.18)",
    glow: "rgba(255, 255, 255, 0.08)",
    joint: "rgba(255, 255, 255, 0.48)",
    hot0: "rgba(255, 160, 148, 0.92)",
    hot1: "rgba(255, 88, 76, 0.96)",
    hotEdge: "rgba(255, 226, 222, 0.52)",
    hotGlow: "rgba(255, 87, 77, 0.28)",
  },
  reference: {
    fill0: "rgba(92, 206, 255, 0.28)",
    fill1: "rgba(82, 176, 255, 0.22)",
    fill2: "rgba(52, 120, 255, 0.16)",
    edge: "rgba(118, 204, 255, 0.24)",
    glow: "rgba(82, 223, 255, 0.14)",
    joint: "rgba(164, 231, 255, 0.26)",
    hot0: "rgba(92, 206, 255, 0.28)",
    hot1: "rgba(52, 120, 255, 0.22)",
    hotEdge: "rgba(118, 204, 255, 0.24)",
    hotGlow: "rgba(82, 223, 255, 0.14)",
  },
};

function fmt(value, digits = 1, suffix = "") {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "--";
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function vecAdd(a, b) {
  return { x: a.x + b.x, y: a.y + b.y };
}

function vecSub(a, b) {
  return { x: a.x - b.x, y: a.y - b.y };
}

function vecScale(v, scale) {
  return { x: v.x * scale, y: v.y * scale };
}

function vecLength(v) {
  return Math.hypot(v.x, v.y);
}

function vecNormalize(v, fallback = { x: 0, y: 1 }) {
  const length = vecLength(v);
  if (length < 1e-4) return fallback;
  return { x: v.x / length, y: v.y / length };
}

function vecPerp(v) {
  return { x: -v.y, y: v.x };
}

function vecDot(a, b) {
  return a.x * b.x + a.y * b.y;
}

function vecLerp(a, b, t) {
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
  };
}

function averagePoints(points) {
  const valid = points.filter(Boolean);
  if (!valid.length) return null;
  const total = valid.reduce((acc, point) => ({
    x: acc.x + point.x,
    y: acc.y + point.y,
  }), { x: 0, y: 0 });
  return {
    x: total.x / valid.length,
    y: total.y / valid.length,
  };
}

function getLandmarkIndex(name) {
  return state.data.landmark_index[name];
}

function getFramePath(index) {
  const media = state.data.media;
  const padded = String(index).padStart(4, "0");
  return `${media.overlay_frame_dir}/${media.overlay_frame_pattern.replace("{index:04d}", padded)}`;
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

function buildScoreBuckets(frames) {
  const buckets = new Map();
  frames.forEach((frame) => {
    const second = Math.floor(frame.time_sec);
    const current = buckets.get(second) || { sum: 0, count: 0 };
    current.sum += frame.score;
    current.count += 1;
    buckets.set(second, current);
  });
  buckets.forEach((bucket, second) => {
    buckets.set(second, { ...bucket, avg: bucket.sum / bucket.count });
  });
  return buckets;
}

function getDisplayedScore(frame) {
  const bucket = state.scoreBuckets.get(Math.floor(frame.time_sec));
  return bucket ? bucket.avg : frame.score;
}

function getHotJointNames(issueId) {
  const primarySide = state.data.input_videos.wrong.primary_side || "left";
  const resolver = ISSUE_HOT_NAMES[issueId];
  return new Set(resolver ? resolver(primarySide) : []);
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
      <p>${fmt(item.avg_delta, 1, suffix)} avg · ${fmt(item.peak_delta, 1, suffix)} peak</p>
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
    const node = document.createElement("article");
    node.className = "metric-card";
    node.innerHTML = `
      <span>${metricLabels[key]}</span>
      <strong>${fmt(wrongMetrics?.[key], 1, suffix)}</strong>
      <em>ref ${fmt(refMetrics?.[key], 1, suffix)}</em>
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

function transformPoint(rawPoint) {
  const x = rawPoint[0] || 0;
  const y = (rawPoint[1] || 0) - 0.18;
  const z = rawPoint[2] || 0;

  const yaw = 0.18;
  const tilt = -0.12;

  const cosY = Math.cos(yaw);
  const sinY = Math.sin(yaw);
  const x1 = x * cosY - z * sinY;
  const z1 = x * sinY + z * cosY;

  const cosX = Math.cos(tilt);
  const sinX = Math.sin(tilt);
  const y1 = y * cosX - z1 * sinX;
  const z2 = y * sinX + z1 * cosX;

  return { x: x1, y: y1, z: z2 };
}

function projectPoint(point, width, height) {
  const cameraZ = 5.2;
  const scale = 4.1 / (cameraZ - point.z);
  return {
    x: width * 0.5 + point.x * width * 0.17 * scale,
    y: height * 0.49 + point.y * height * 0.23 * scale,
    depth: point.z,
  };
}

function buildProjectedLandmarks(landmarks, width, height) {
  if (!Array.isArray(landmarks) || !landmarks.length) return null;
  return landmarks.map((point) => projectPoint(transformPoint(point), width, height));
}

function getProjectedPoint(projected, name) {
  const index = getLandmarkIndex(name);
  return Number.isInteger(index) ? projected[index] : null;
}

function buildAdultRig(landmarks, width, height) {
  const projected = buildProjectedLandmarks(landmarks, width, height);
  if (!projected) return null;

  const leftShoulderRaw = getProjectedPoint(projected, "left_shoulder");
  const rightShoulderRaw = getProjectedPoint(projected, "right_shoulder");
  const leftHipRaw = getProjectedPoint(projected, "left_hip");
  const rightHipRaw = getProjectedPoint(projected, "right_hip");
  const shoulderMidRaw = averagePoints([leftShoulderRaw, rightShoulderRaw]);
  const hipMidRaw = averagePoints([leftHipRaw, rightHipRaw]);
  const ankleMidRaw = averagePoints([getProjectedPoint(projected, "left_ankle"), getProjectedPoint(projected, "right_ankle")]);
  const faceMidRaw = averagePoints([
    getProjectedPoint(projected, "nose"),
    getProjectedPoint(projected, "left_eye"),
    getProjectedPoint(projected, "right_eye"),
    getProjectedPoint(projected, "left_ear"),
    getProjectedPoint(projected, "right_ear"),
  ]);

  if (!shoulderMidRaw || !hipMidRaw || !ankleMidRaw) return null;

  const torsoAxis = vecNormalize(vecSub(hipMidRaw, shoulderMidRaw), { x: 0, y: 1 });
  let lateralAxis = vecNormalize(vecSub(rightShoulderRaw || shoulderMidRaw, leftShoulderRaw || shoulderMidRaw), vecPerp(torsoAxis));
  if (Math.abs(vecDot(torsoAxis, lateralAxis)) > 0.45) {
    lateralAxis = vecNormalize(vecPerp(torsoAxis), { x: 1, y: 0 });
  }

  const rawHeight = clamp(
    ankleMidRaw.y - ((faceMidRaw && faceMidRaw.y) || shoulderMidRaw.y - height * 0.08),
    height * 0.42,
    height * 0.82,
  );
  const headUnit = rawHeight / 8;

  const rig = {
    headUnit,
    neck: shoulderMidRaw,
    pelvis: vecAdd(shoulderMidRaw, vecScale(torsoAxis, headUnit * 2.55)),
  };

  rig.leftShoulder = vecAdd(rig.neck, vecScale(lateralAxis, -headUnit * 0.92));
  rig.rightShoulder = vecAdd(rig.neck, vecScale(lateralAxis, headUnit * 0.92));
  rig.leftHip = vecAdd(rig.pelvis, vecScale(lateralAxis, -headUnit * 0.58));
  rig.rightHip = vecAdd(rig.pelvis, vecScale(lateralAxis, headUnit * 0.58));
  rig.sternum = vecAdd(rig.neck, vecScale(torsoAxis, headUnit * 0.76));
  rig.waist = vecAdd(rig.neck, vecScale(torsoAxis, headUnit * 1.78));
  rig.headCenter = vecAdd(rig.neck, vecScale(torsoAxis, -headUnit * 1.02));

  const lengths = {
    upperArm: headUnit * 1.45,
    forearm: headUnit * 1.38,
    thigh: headUnit * 1.95,
    shin: headUnit * 2.02,
    foot: headUnit * 0.86,
  };

  function point(name) {
    return getProjectedPoint(projected, name);
  }

  function direction(fromName, toName, fallback) {
    const from = point(fromName);
    const to = point(toName);
    if (!from || !to) return fallback;
    return vecNormalize(vecSub(to, from), fallback);
  }

  const leftUpperArmDir = direction("left_shoulder", "left_elbow", vecNormalize(vecAdd(vecScale(lateralAxis, -1), vecScale(torsoAxis, 0.12))));
  const rightUpperArmDir = direction("right_shoulder", "right_elbow", vecNormalize(vecAdd(vecScale(lateralAxis, 1), vecScale(torsoAxis, 0.12))));
  const leftForearmDir = direction("left_elbow", "left_wrist", leftUpperArmDir);
  const rightForearmDir = direction("right_elbow", "right_wrist", rightUpperArmDir);
  const leftThighDir = direction("left_hip", "left_knee", vecNormalize(vecAdd(vecScale(lateralAxis, -0.12), vecScale(torsoAxis, 1))));
  const rightThighDir = direction("right_hip", "right_knee", vecNormalize(vecAdd(vecScale(lateralAxis, 0.12), vecScale(torsoAxis, 1))));
  const leftShinDir = direction("left_knee", "left_ankle", leftThighDir);
  const rightShinDir = direction("right_knee", "right_ankle", rightThighDir);
  const leftFootDir = direction("left_ankle", "left_foot_index", vecNormalize(vecAdd(vecScale(lateralAxis, -0.18), vecScale(torsoAxis, 0.08)), { x: 1, y: 0 }));
  const rightFootDir = direction("right_ankle", "right_foot_index", vecNormalize(vecAdd(vecScale(lateralAxis, 0.18), vecScale(torsoAxis, 0.08)), { x: 1, y: 0 }));

  rig.leftElbow = vecAdd(rig.leftShoulder, vecScale(leftUpperArmDir, lengths.upperArm));
  rig.rightElbow = vecAdd(rig.rightShoulder, vecScale(rightUpperArmDir, lengths.upperArm));
  rig.leftWrist = vecAdd(rig.leftElbow, vecScale(leftForearmDir, lengths.forearm));
  rig.rightWrist = vecAdd(rig.rightElbow, vecScale(rightForearmDir, lengths.forearm));
  rig.leftKnee = vecAdd(rig.leftHip, vecScale(leftThighDir, lengths.thigh));
  rig.rightKnee = vecAdd(rig.rightHip, vecScale(rightThighDir, lengths.thigh));
  rig.leftAnkle = vecAdd(rig.leftKnee, vecScale(leftShinDir, lengths.shin));
  rig.rightAnkle = vecAdd(rig.rightKnee, vecScale(rightShinDir, lengths.shin));
  rig.leftFoot = vecAdd(rig.leftAnkle, vecScale(leftFootDir, lengths.foot));
  rig.rightFoot = vecAdd(rig.rightAnkle, vecScale(rightFootDir, lengths.foot));

  return rig;
}

function drawBackdrop(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);

  const background = ctx.createLinearGradient(0, 0, 0, height);
  background.addColorStop(0, "rgba(9, 14, 18, 1)");
  background.addColorStop(1, "rgba(4, 7, 12, 1)");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);

  const halo = ctx.createRadialGradient(width * 0.5, height * 0.16, 10, width * 0.5, height * 0.16, width * 0.5);
  halo.addColorStop(0, "rgba(82, 223, 255, 0.18)");
  halo.addColorStop(1, "rgba(82, 223, 255, 0)");
  ctx.fillStyle = halo;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(82, 223, 255, 0.07)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 8; i += 1) {
    const y = height * (0.12 + i * 0.095);
    ctx.beginPath();
    ctx.moveTo(width * 0.06, y);
    ctx.lineTo(width * 0.94, y);
    ctx.stroke();
  }

  for (let i = 0; i <= 6; i += 1) {
    const x = width * (0.12 + i * 0.12);
    ctx.beginPath();
    ctx.moveTo(x, height * 0.12);
    ctx.lineTo(x, height * 0.88);
    ctx.stroke();
  }
}

function drawCapsule(ctx, start, end, width, palette, isHot) {
  if (!start || !end) return;

  const gradient = ctx.createLinearGradient(start.x, start.y, end.x, end.y);
  if (isHot) {
    gradient.addColorStop(0, palette.hot0);
    gradient.addColorStop(1, palette.hot1);
  } else {
    gradient.addColorStop(0, palette.fill0);
    gradient.addColorStop(0.52, palette.fill1);
    gradient.addColorStop(1, palette.fill2);
  }

  ctx.save();
  ctx.strokeStyle = gradient;
  ctx.lineCap = "round";
  ctx.lineWidth = width;
  ctx.shadowBlur = isHot ? width * 0.42 : width * 0.18;
  ctx.shadowColor = isHot ? palette.hotGlow : palette.glow;
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();

  ctx.fillStyle = isHot ? palette.hot1 : palette.joint;
  ctx.beginPath();
  ctx.arc(start.x, start.y, width * 0.28, 0, Math.PI * 2);
  ctx.arc(end.x, end.y, width * 0.28, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawPolygon(ctx, points, palette, isHot) {
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
  ctx.strokeStyle = isHot ? palette.hotEdge || palette.hot1 : palette.edge;
  ctx.lineWidth = 1.2;
  ctx.shadowBlur = isHot ? 18 : 8;
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

function drawAdultFigure(ctx, rig, hotNames, palette, isGhost = false) {
  if (!rig) return;

  const blinkOn = !isGhost && state.avatar ? state.avatar.blinkVisible : true;
  const isHot = (...names) => blinkOn && names.some((name) => hotNames.has(name));
  const head = rig.headUnit;

  const chestLeft = vecLerp(rig.leftShoulder, rig.leftHip, 0.24);
  const chestRight = vecLerp(rig.rightShoulder, rig.rightHip, 0.24);
  const waistLeft = vecLerp(rig.leftShoulder, rig.leftHip, 0.72);
  const waistRight = vecLerp(rig.rightShoulder, rig.rightHip, 0.72);

  ctx.save();
  if (isGhost) {
    ctx.globalAlpha = 0.88;
  }

  drawPolygon(
    ctx,
    [rig.leftShoulder, rig.rightShoulder, chestRight, chestLeft],
    palette,
    isHot("left_shoulder", "right_shoulder"),
  );
  drawPolygon(
    ctx,
    [chestLeft, chestRight, waistRight, rig.pelvis, waistLeft],
    palette,
    isHot("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
  );
  drawPolygon(ctx, [waistLeft, waistRight, rig.rightHip, rig.leftHip], palette, isHot("left_hip", "right_hip"));

  drawCapsule(ctx, rig.leftShoulder, rig.leftElbow, head * 0.34, palette, isHot("left_shoulder", "left_elbow"));
  drawCapsule(ctx, rig.leftElbow, rig.leftWrist, head * 0.28, palette, isHot("left_elbow", "left_wrist"));
  drawCapsule(ctx, rig.rightShoulder, rig.rightElbow, head * 0.34, palette, isHot("right_shoulder", "right_elbow"));
  drawCapsule(ctx, rig.rightElbow, rig.rightWrist, head * 0.28, palette, isHot("right_elbow", "right_wrist"));
  drawCapsule(ctx, rig.leftHip, rig.leftKnee, head * 0.46, palette, isHot("left_hip", "left_knee"));
  drawCapsule(ctx, rig.leftKnee, rig.leftAnkle, head * 0.36, palette, isHot("left_knee", "left_ankle"));
  drawCapsule(ctx, rig.rightHip, rig.rightKnee, head * 0.46, palette, isHot("right_hip", "right_knee"));
  drawCapsule(ctx, rig.rightKnee, rig.rightAnkle, head * 0.36, palette, isHot("right_knee", "right_ankle"));
  drawCapsule(ctx, rig.leftAnkle, rig.leftFoot, head * 0.18, palette, isHot("left_ankle", "left_foot_index", "left_heel"));
  drawCapsule(ctx, rig.rightAnkle, rig.rightFoot, head * 0.18, palette, isHot("right_ankle", "right_foot_index", "right_heel"));
  drawCapsule(ctx, rig.neck, rig.sternum, head * 0.18, palette, isHot("left_shoulder", "right_shoulder"));

  const headHot = isHot("left_shoulder", "right_shoulder");
  drawPolygon(
    ctx,
    [
      { x: rig.headCenter.x - head * 0.34, y: rig.headCenter.y - head * 0.15 },
      { x: rig.headCenter.x + head * 0.34, y: rig.headCenter.y - head * 0.15 },
      { x: rig.headCenter.x + head * 0.28, y: rig.headCenter.y + head * 0.55 },
      { x: rig.headCenter.x - head * 0.28, y: rig.headCenter.y + head * 0.55 },
    ],
    palette,
    headHot,
  );

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

  const referenceRig = buildAdultRig(state.avatar.referenceLandmarks, width, height);
  const wrongRig = buildAdultRig(state.avatar.wrongLandmarks, width, height);

  drawAdultFigure(ctx, referenceRig, new Set(), avatarPalette.reference, true);
  drawAdultFigure(ctx, wrongRig, state.avatar.hotNames || new Set(), avatarPalette.live, false);

  ctx.fillStyle = "rgba(245, 246, 247, 0.88)";
  ctx.font = `${Math.max(12, Math.round(height * 0.03))}px "Space Grotesk", sans-serif`;
  ctx.fillText("8-head mannequin", width * 0.06, height * 0.08);
  ctx.fillStyle = "rgba(155, 167, 179, 0.88)";
  ctx.font = `${Math.max(11, Math.round(height * 0.022))}px "Pretendard", sans-serif`;
  ctx.fillText("white = wrong, blue ghost = correct", width * 0.06, height * 0.13);
}

function initAvatar() {
  const canvas = ensureAvatarCanvas();
  const ctx = canvas.getContext("2d");
  state.avatar = {
    canvas,
    ctx,
    wrongLandmarks: null,
    referenceLandmarks: null,
    hotNames: new Set(),
    blinkVisible: true,
    blinkTimerId: window.setInterval(() => {
      if (!state.avatar) return;
      state.avatar.blinkVisible = !state.avatar.blinkVisible;
      if (state.avatar.hotNames.size) {
        renderAvatar();
      }
    }, BLINK_INTERVAL_MS),
  };

  window.addEventListener("resize", renderAvatar);
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
  utterance.rate = 0.97;
  window.speechSynthesis.speak(utterance);

  state.voice.lastSpokenIssueId = issueId;
  state.voice.lastSpokenAt = now;
  state.voice.stableCount = 0;
}

function updateFrame(frame) {
  if (!frame) return;
  state.currentFrame = frame;

  const displayedScore = getDisplayedScore(frame);
  const topIssue = (frame.issues || [])[0] || null;
  const warningIssue = topIssue && displayedScore <= WARNING_SCORE_THRESHOLD ? topIssue : null;

  elements.scoreValue.textContent = `${Math.round(displayedScore)}`;
  elements.currentIssue.textContent = warningIssue ? warningIssue.label : "기준 자세 범위";
  elements.phaseLabel.textContent = `Rep ${frame.rep_index} · ${frame.phase}`;
  elements.coachText.textContent = warningIssue
    ? frame.coach_text
    : "현재 1초 평균 점수 기준으로는 경고 임계값 아래로 떨어지지 않았습니다.";

  renderMetrics(frame);

  const cursor = document.getElementById("timelineCursor");
  const duration = state.data.input_videos.wrong.duration_sec || 1;
  if (cursor) {
    cursor.style.left = `${(frame.time_sec / duration) * 100}%`;
  }

  const hotNames = warningIssue ? getHotJointNames(warningIssue.id) : new Set();
  if (state.avatar) {
    state.avatar.wrongLandmarks = frame.wrong.world_landmarks;
    state.avatar.referenceLandmarks = frame.reference.world_landmarks;
    state.avatar.hotNames = hotNames;
    renderAvatar();
  }

  maybeSpeak({ ...frame, score: displayedScore }, warningIssue);
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
  showFrame(findNearestFrameIndex(timeSec));
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
  state.scoreBuckets = buildScoreBuckets(state.data.frames);
  state.player.fps = state.data.input_videos.wrong.sampled_fps || 10;

  elements.averageScore.textContent = `${Math.round(state.data.overview.average_score)}`;
  elements.repCount.textContent = `${state.data.overview.rep_count}`;
  elements.thresholdValue.textContent = `${WARNING_SCORE_THRESHOLD}`;
  elements.summaryText.textContent = `1초 평균 점수 기준으로 주요 이탈은 ${state.data.overview.top_findings.map((item) => item.label).join(" / ")} 입니다.`;

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
