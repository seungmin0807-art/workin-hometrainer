const WARNING_SCORE_THRESHOLD = 76;
const VOICE_SCORE_THRESHOLD = 72;
const VOICE_MIN_GAP_MS = 8000;
const VOICE_REPEAT_GAP_MS = 15000;
const VOICE_STABLE_FRAMES = 4;
const BLINK_INTERVAL_MS = 280;

const SUPPORT_JOINT_NAMES = [
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
  "left_heel",
  "right_heel",
  "left_foot_index",
  "right_foot_index",
];

const GROUND_JOINT_NAMES = [
  "left_ankle",
  "right_ankle",
  "left_heel",
  "right_heel",
  "left_foot_index",
  "right_foot_index",
];

const FOOT_LOCK_BLEND = {
  left_heel: 0.88,
  right_heel: 0.88,
  left_foot_index: 0.84,
  right_foot_index: 0.84,
  left_ankle: 0.72,
  right_ankle: 0.72,
};

const FACE_JOINT_NAMES = [
  "nose",
  "left_eye",
  "right_eye",
  "left_ear",
  "right_ear",
];

const ISSUE_HOT_NAMES = {
  knee_flexion: (side) => [`${side}_hip`, `${side}_knee`, `${side}_ankle`],
  shin_lean: (side) => [`${side}_knee`, `${side}_ankle`, `${side}_heel`, `${side}_foot_index`],
  torso_lean: (side) => [`${side}_shoulder`, `${side}_hip`],
  depth: () => ["left_hip", "right_hip", "left_knee", "right_knee"],
};

const avatarPalette = {
  live: {
    fill0: "rgba(255, 255, 255, 0.4)",
    fill1: "rgba(255, 255, 255, 0.32)",
    fill2: "rgba(255, 255, 255, 0.26)",
    edge: "rgba(255, 255, 255, 0.18)",
    glow: "rgba(255, 255, 255, 0.08)",
    joint: "rgba(255, 255, 255, 0.45)",
    hot0: "rgba(255, 160, 148, 0.96)",
    hot1: "rgba(255, 88, 76, 0.98)",
    hotEdge: "rgba(255, 228, 224, 0.52)",
    hotGlow: "rgba(255, 87, 77, 0.32)",
  },
  reference: {
    fill0: "rgba(92, 206, 255, 0.3)",
    fill1: "rgba(82, 176, 255, 0.22)",
    fill2: "rgba(52, 120, 255, 0.16)",
    edge: "rgba(118, 204, 255, 0.22)",
    glow: "rgba(82, 223, 255, 0.14)",
    joint: "rgba(164, 231, 255, 0.26)",
    hot0: "rgba(92, 206, 255, 0.3)",
    hot1: "rgba(52, 120, 255, 0.22)",
    hotEdge: "rgba(118, 204, 255, 0.22)",
    hotGlow: "rgba(82, 223, 255, 0.14)",
  },
};

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

function fmt(value, digits = 1, suffix = "") {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "--";
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function median(values) {
  const filtered = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!filtered.length) return null;
  const mid = Math.floor(filtered.length / 2);
  return filtered.length % 2 ? filtered[mid] : (filtered[mid - 1] + filtered[mid]) * 0.5;
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

function cloneLandmarks(landmarks) {
  return landmarks.map((point) => [
    point[0] ?? 0,
    point[1] ?? 0,
    point[2] ?? 0,
    point[3] ?? 1,
  ]);
}

function getJointPoint(landmarks, name) {
  const index = getLandmarkIndex(name);
  return Number.isInteger(index) && landmarks?.[index] ? landmarks[index] : null;
}

function averageWorldPoints(landmarks, names) {
  const points = names.map((name) => getJointPoint(landmarks, name)).filter(Boolean);
  if (!points.length) return null;
  return {
    x: points.reduce((sum, point) => sum + point[0], 0) / points.length,
    y: points.reduce((sum, point) => sum + point[1], 0) / points.length,
    z: points.reduce((sum, point) => sum + point[2], 0) / points.length,
  };
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

function getHotJointNames(issueId) {
  const primarySide = state.data.input_videos.wrong.primary_side || "left";
  const resolver = ISSUE_HOT_NAMES[issueId];
  return new Set(resolver ? resolver(primarySide) : []);
}

function emaPass(sequence, alpha, selectedIndices = null) {
  let previous = null;
  return sequence.map((landmarks) => {
    if (!landmarks) {
      return previous ? cloneLandmarks(previous) : null;
    }

    if (!previous) {
      previous = cloneLandmarks(landmarks);
      return cloneLandmarks(previous);
    }

    const next = cloneLandmarks(landmarks);
    const jointIndices = selectedIndices || landmarks.map((_, index) => index);
    jointIndices.forEach((jointIndex) => {
      for (let coord = 0; coord < 3; coord += 1) {
        next[jointIndex][coord] =
          alpha * landmarks[jointIndex][coord] + (1 - alpha) * previous[jointIndex][coord];
      }
      next[jointIndex][3] =
        alpha * (landmarks[jointIndex][3] ?? 1) + (1 - alpha) * (previous[jointIndex][3] ?? 1);
    });
    previous = cloneLandmarks(next);
    return next;
  });
}

function smoothSequence(sequence, alpha, selectedIndices = null) {
  const forward = emaPass(sequence, alpha, selectedIndices);
  const backward = emaPass([...sequence].reverse(), alpha, selectedIndices).reverse();
  return sequence.map((landmarks, frameIndex) => {
    if (!landmarks) return null;
    const forwardFrame = forward[frameIndex];
    const backwardFrame = backward[frameIndex];
    if (!forwardFrame || !backwardFrame) return cloneLandmarks(landmarks);

    const result = cloneLandmarks(landmarks);
    const jointIndices = selectedIndices || landmarks.map((_, index) => index);
    jointIndices.forEach((jointIndex) => {
      for (let coord = 0; coord < 4; coord += 1) {
        result[jointIndex][coord] =
          (forwardFrame[jointIndex][coord] + backwardFrame[jointIndex][coord]) * 0.5;
      }
    });
    return result;
  });
}

function buildSupportInfo(landmarks) {
  const points = GROUND_JOINT_NAMES.map((name) => getJointPoint(landmarks, name)).filter(Boolean);
  if (!points.length) return null;
  return {
    groundY: Math.max(...points.map((point) => point[1])),
    centerX: points.reduce((sum, point) => sum + point[0], 0) / points.length,
    centerZ: points.reduce((sum, point) => sum + point[2], 0) / points.length,
  };
}

function buildHeadInfo(landmarks) {
  const face = averageWorldPoints(landmarks, FACE_JOINT_NAMES);
  if (face) return face;
  const shoulders = averageWorldPoints(landmarks, ["left_shoulder", "right_shoulder"]);
  const hips = averageWorldPoints(landmarks, ["left_hip", "right_hip"]);
  if (!shoulders || !hips) return null;
  const torso = {
    x: shoulders.x,
    y: shoulders.y - (hips.y - shoulders.y) * 0.55,
    z: shoulders.z,
  };
  return torso;
}

function buildRootInfo(landmarks) {
  return averageWorldPoints(landmarks, ["left_hip", "right_hip"]);
}

function buildJointAnchors(sequence, names) {
  const anchors = {};
  names.forEach((name) => {
    const index = getLandmarkIndex(name);
    if (!Number.isInteger(index)) return;

    const xs = [];
    const ys = [];
    const zs = [];
    sequence.forEach((landmarks) => {
      if (!landmarks?.[index]) return;
      xs.push(landmarks[index][0]);
      ys.push(landmarks[index][1]);
      zs.push(landmarks[index][2]);
    });

    if (!xs.length) return;
    anchors[name] = {
      x: median(xs),
      y: median(ys),
      z: median(zs),
    };
  });
  return anchors;
}

function prepareAvatarTrack(roleKey) {
  const sequence = state.data.frames.map((frame) => {
    const landmarks = frame?.[roleKey]?.world_landmarks;
    return Array.isArray(landmarks) && landmarks.length ? cloneLandmarks(landmarks) : null;
  });

  const jointIndices = SUPPORT_JOINT_NAMES.map(getLandmarkIndex).filter(Number.isInteger);
  const smoothed = smoothSequence(sequence, 0.42);
  const supportSmoothed = smoothSequence(smoothed, 0.16, jointIndices);

  const supports = supportSmoothed.map((landmarks) => (landmarks ? buildSupportInfo(landmarks) : null));
  const heads = supportSmoothed.map((landmarks) => (landmarks ? buildHeadInfo(landmarks) : null));
  const roots = supportSmoothed.map((landmarks) => (landmarks ? buildRootInfo(landmarks) : null));

  const targetGroundY = median(supports.map((support) => support?.groundY));
  const targetRootX = median(roots.map((root) => root?.x));
  const targetRootZ = median(roots.map((root) => root?.z));
  const targetBodyHeight = median(
    supports.map((support, index) => {
      if (!support || !heads[index]) return null;
      return support.groundY - heads[index].y;
    }),
  );

  const stabilized = supportSmoothed.map((landmarks, index) => {
    if (!landmarks) return null;
    const support = supports[index];
    const head = heads[index];
    const root = roots[index];
    if (
      !support ||
      !head ||
      !root ||
      targetGroundY === null ||
      targetBodyHeight === null ||
      targetRootX === null ||
      targetRootZ === null
    ) {
      return cloneLandmarks(landmarks);
    }

    const currentBodyHeight = support.groundY - head.y;
    const scale = currentBodyHeight
      ? clamp(1 + ((targetBodyHeight / currentBodyHeight) - 1) * 0.2, 0.985, 1.015)
      : 1;

    return landmarks.map((point) => [
      targetRootX + (point[0] - root.x) * scale,
      targetGroundY + (point[1] - support.groundY) * scale,
      targetRootZ + (point[2] - root.z) * scale,
      point[3] ?? 1,
    ]);
  });

  const footAnchors = buildJointAnchors(stabilized, Object.keys(FOOT_LOCK_BLEND));
  const footLocked = stabilized.map((landmarks) => {
    if (!landmarks) return null;
    const next = cloneLandmarks(landmarks);

    Object.entries(FOOT_LOCK_BLEND).forEach(([name, amount]) => {
      const index = getLandmarkIndex(name);
      const anchor = footAnchors[name];
      if (!Number.isInteger(index) || !anchor || !next[index]) return;
      next[index][0] = next[index][0] + (anchor.x - next[index][0]) * amount;
      next[index][1] = next[index][1] + (anchor.y - next[index][1]) * amount;
      next[index][2] = next[index][2] + (anchor.z - next[index][2]) * amount;
    });

    return next;
  });

  const supportLocked = smoothSequence(footLocked, 0.12, jointIndices);
  const finalTrack = smoothSequence(supportLocked, 0.48);
  return finalTrack;
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
  const y = (rawPoint[1] || 0) - 0.17;
  const z = rawPoint[2] || 0;
  const yaw = 0.06;
  const tilt = -0.04;

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
  const scale = 4.05 / (cameraZ - point.z);
  return {
    x: width * 0.5 + point.x * width * 0.17 * scale,
    y: height * 0.5 + point.y * height * 0.23 * scale,
  };
}

function normalizeProjectedPoints(projected, width, height) {
  const pick = (name) => {
    const index = getLandmarkIndex(name);
    return Number.isInteger(index) ? projected[index] : null;
  };

  const supportPoints = [
    pick("left_ankle"),
    pick("right_ankle"),
    pick("left_heel"),
    pick("right_heel"),
    pick("left_foot_index"),
    pick("right_foot_index"),
  ].filter(Boolean);
  const shoulderMid = averagePoints([pick("left_shoulder"), pick("right_shoulder")]);
  const hipMid = averagePoints([pick("left_hip"), pick("right_hip")]);
  const facePoints = FACE_JOINT_NAMES.map(pick).filter(Boolean);

  if (!supportPoints.length || !shoulderMid || !hipMid) {
    return projected;
  }

  const torsoLength = vecLength(vecSub(hipMid, shoulderMid));
  const groundY = Math.max(...supportPoints.map((point) => point.y));
  const topY = facePoints.length
    ? Math.min(...facePoints.map((point) => point.y))
    : shoulderMid.y - torsoLength * 0.8;
  const bodyHeight = Math.max(groundY - topY, 1);
  const centerX = (shoulderMid.x + hipMid.x) * 0.5;
  const targetHeight = height * 0.72;
  const targetGroundY = height * 0.86;
  const targetCenterX = width * 0.5;
  const scale = clamp(targetHeight / bodyHeight, 1.15, 5.4);

  return projected.map((point) => ({
    x: (point.x - centerX) * scale + targetCenterX,
    y: (point.y - groundY) * scale + targetGroundY,
  }));
}

function buildProjectedFrame(landmarks, width, height) {
  if (!Array.isArray(landmarks) || !landmarks.length) return null;
  const projected = normalizeProjectedPoints(
    landmarks.map((point) => projectPoint(transformPoint(point), width, height)),
    width,
    height,
  );

  const pick = (name) => {
    const index = getLandmarkIndex(name);
    return Number.isInteger(index) ? projected[index] : null;
  };

  const leftShoulder = pick("left_shoulder");
  const rightShoulder = pick("right_shoulder");
  const leftHip = pick("left_hip");
  const rightHip = pick("right_hip");
  const leftElbow = pick("left_elbow");
  const rightElbow = pick("right_elbow");
  const leftWrist = pick("left_wrist");
  const rightWrist = pick("right_wrist");
  const leftKnee = pick("left_knee");
  const rightKnee = pick("right_knee");
  const leftAnkle = pick("left_ankle");
  const rightAnkle = pick("right_ankle");
  const leftFoot = pick("left_foot_index");
  const rightFoot = pick("right_foot_index");

  const shoulderMid = averagePoints([leftShoulder, rightShoulder]);
  const hipMid = averagePoints([leftHip, rightHip]);
  const supportMid = averagePoints([
    leftAnkle,
    rightAnkle,
    pick("left_heel"),
    pick("right_heel"),
    leftFoot,
    rightFoot,
  ]);
  const faceMid = averagePoints(FACE_JOINT_NAMES.map(pick).filter(Boolean));
  if (!shoulderMid || !hipMid || !supportMid) return null;

  const torsoAxis = vecNormalize(vecSub(hipMid, shoulderMid), { x: 0, y: 1 });
  let lateralAxis = vecNormalize(vecSub(rightShoulder || shoulderMid, leftShoulder || shoulderMid), vecPerp(torsoAxis));
  if (Math.abs(vecDot(torsoAxis, lateralAxis)) > 0.45) {
    lateralAxis = vecNormalize(vecPerp(torsoAxis), { x: 1, y: 0 });
  }

  const bodyHeight = clamp(
    supportMid.y - ((faceMid && faceMid.y) || shoulderMid.y - height * 0.08),
    height * 0.54,
    height * 0.76,
  );
  const headUnit = clamp(bodyHeight / 8.4, height * 0.035, height * 0.08);
  const measuredShoulderHalf = vecLength(vecSub(rightShoulder || shoulderMid, leftShoulder || shoulderMid)) * 0.5;
  const measuredHipHalf = vecLength(vecSub(rightHip || hipMid, leftHip || hipMid)) * 0.5;
  const shoulderHalf = clamp(measuredShoulderHalf * 0.94, headUnit * 0.64, headUnit * 1.16);
  const hipHalf = clamp(measuredHipHalf * 0.98, headUnit * 0.4, headUnit * 0.88);

  const torso = {
    leftShoulder: vecAdd(shoulderMid, vecScale(lateralAxis, -shoulderHalf)),
    rightShoulder: vecAdd(shoulderMid, vecScale(lateralAxis, shoulderHalf)),
    leftHip: vecAdd(hipMid, vecScale(lateralAxis, -hipHalf)),
    rightHip: vecAdd(hipMid, vecScale(lateralAxis, hipHalf)),
  };
  torso.chestLeft = vecLerp(torso.leftShoulder, torso.leftHip, 0.24);
  torso.chestRight = vecLerp(torso.rightShoulder, torso.rightHip, 0.24);
  torso.waistLeft = vecLerp(torso.leftShoulder, torso.leftHip, 0.72);
  torso.waistRight = vecLerp(torso.rightShoulder, torso.rightHip, 0.72);
  torso.neck = shoulderMid;
  torso.sternum = vecLerp(shoulderMid, hipMid, 0.28);
  torso.headCenter = faceMid ? vecAdd(faceMid, vecScale(torsoAxis, -headUnit * 0.06)) : vecAdd(shoulderMid, vecScale(torsoAxis, -headUnit * 1.0));

  return {
    headUnit,
    torsoAxis,
    torso,
    joints: {
      leftShoulder: torso.leftShoulder,
      rightShoulder: torso.rightShoulder,
      leftElbow,
      rightElbow,
      leftWrist,
      rightWrist,
      leftHip: torso.leftHip,
      rightHip: torso.rightHip,
      leftKnee,
      rightKnee,
      leftAnkle,
      rightAnkle,
      leftFoot,
      rightFoot,
    },
  };
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

function drawAdultFigure(ctx, frameData, hotNames, palette, isGhost = false) {
  if (!frameData) return;

  const blinkOn = !isGhost && state.avatar ? state.avatar.blinkVisible : true;
  const isHot = (...names) => blinkOn && names.some((name) => hotNames.has(name));
  const head = frameData.headUnit;
  const torso = frameData.torso;
  const joints = frameData.joints;

  ctx.save();
  if (isGhost) {
    ctx.globalAlpha = 0.88;
  }

  drawPolygon(ctx, [torso.leftShoulder, torso.rightShoulder, torso.chestRight, torso.chestLeft], palette, false);
  drawPolygon(ctx, [torso.chestLeft, torso.chestRight, torso.waistRight, torso.rightHip, torso.leftHip, torso.waistLeft], palette, false);

  drawCapsule(ctx, joints.leftShoulder, joints.leftElbow, head * 0.33, palette, isHot("left_shoulder", "left_elbow"));
  drawCapsule(ctx, joints.leftElbow, joints.leftWrist, head * 0.27, palette, isHot("left_elbow", "left_wrist"));
  drawCapsule(ctx, joints.rightShoulder, joints.rightElbow, head * 0.33, palette, isHot("right_shoulder", "right_elbow"));
  drawCapsule(ctx, joints.rightElbow, joints.rightWrist, head * 0.27, palette, isHot("right_elbow", "right_wrist"));
  drawCapsule(ctx, joints.leftHip, joints.leftKnee, head * 0.46, palette, isHot("left_hip", "left_knee"));
  drawCapsule(ctx, joints.leftKnee, joints.leftAnkle, head * 0.36, palette, isHot("left_knee", "left_ankle"));
  drawCapsule(ctx, joints.rightHip, joints.rightKnee, head * 0.46, palette, isHot("right_hip", "right_knee"));
  drawCapsule(ctx, joints.rightKnee, joints.rightAnkle, head * 0.36, palette, isHot("right_knee", "right_ankle"));
  drawCapsule(ctx, joints.leftAnkle, joints.leftFoot, head * 0.17, palette, isHot("left_ankle", "left_heel", "left_foot_index"));
  drawCapsule(ctx, joints.rightAnkle, joints.rightFoot, head * 0.17, palette, isHot("right_ankle", "right_heel", "right_foot_index"));
  drawCapsule(ctx, torso.neck, torso.sternum, head * 0.16, palette, false);

  drawPolygon(
    ctx,
    [
      { x: torso.headCenter.x - head * 0.33, y: torso.headCenter.y - head * 0.2 },
      { x: torso.headCenter.x + head * 0.33, y: torso.headCenter.y - head * 0.2 },
      { x: torso.headCenter.x + head * 0.29, y: torso.headCenter.y + head * 0.56 },
      { x: torso.headCenter.x - head * 0.29, y: torso.headCenter.y + head * 0.56 },
    ],
    palette,
    false,
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

  const referenceFrame = buildProjectedFrame(state.avatar.referenceLandmarks, width, height);
  const wrongFrame = buildProjectedFrame(state.avatar.wrongLandmarks, width, height);

  drawAdultFigure(ctx, referenceFrame, new Set(), avatarPalette.reference, true);
  drawAdultFigure(ctx, wrongFrame, state.avatar.hotNames || new Set(), avatarPalette.live, false);

  ctx.fillStyle = "rgba(245, 246, 247, 0.88)";
  ctx.font = `${Math.max(12, Math.round(height * 0.03))}px "Space Grotesk", sans-serif`;
  ctx.fillText("stabilized adult mannequin", width * 0.06, height * 0.08);
  ctx.fillStyle = "rgba(155, 167, 179, 0.88)";
  ctx.font = `${Math.max(11, Math.round(height * 0.022))}px "Pretendard", sans-serif`;
  ctx.fillText("white = wrong, blue ghost = correct", width * 0.06, height * 0.13);
}

function initAvatar(avatarTracks) {
  const canvas = ensureAvatarCanvas();
  const ctx = canvas.getContext("2d");
  state.avatar = {
    canvas,
    ctx,
    tracks: avatarTracks,
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
  const currentIndex = state.player.index;

  elements.scoreValue.textContent = `${Math.round(displayedScore)}`;
  elements.currentIssue.textContent = warningIssue ? warningIssue.label : "기준 자세 범위";
  elements.phaseLabel.textContent = `Rep ${frame.rep_index} · ${frame.phase}`;
  elements.coachText.textContent = warningIssue
    ? frame.coach_text
    : "아바타는 발 접지 기준과 시간축 스무딩을 적용한 안정화 모션으로 표시 중입니다.";

  renderMetrics(frame);

  const cursor = document.getElementById("timelineCursor");
  const duration = state.data.input_videos.wrong.duration_sec || 1;
  if (cursor) {
    cursor.style.left = `${(frame.time_sec / duration) * 100}%`;
  }

  const hotNames = warningIssue ? getHotJointNames(warningIssue.id) : new Set();
  if (state.avatar) {
    state.avatar.wrongLandmarks = state.avatar.tracks.wrong[currentIndex] || frame.wrong.world_landmarks;
    state.avatar.referenceLandmarks = state.avatar.tracks.reference[currentIndex] || frame.reference.world_landmarks;
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

  const avatarTracks = {
    wrong: prepareAvatarTrack("wrong"),
    reference: prepareAvatarTrack("reference"),
  };

  elements.averageScore.textContent = `${Math.round(state.data.overview.average_score)}`;
  elements.repCount.textContent = `${state.data.overview.rep_count}`;
  elements.thresholdValue.textContent = `${WARNING_SCORE_THRESHOLD}`;
  elements.summaryText.textContent = `1초 평균 점수와 접지 안정화를 기준으로 아바타 모션을 보정했습니다.`;

  buildTimeline();
  renderFindings();
  initAvatar(avatarTracks);
  bindControls();
  showFrame(0);
}

bootstrap().catch((error) => {
  console.error(error);
  elements.summaryText.textContent = error.message;
  elements.coachText.textContent = "프로토타입 로드에 실패했습니다.";
});
