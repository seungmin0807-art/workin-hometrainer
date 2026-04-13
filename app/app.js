const state = {
  data: null,
  currentFrame: null,
  voiceEnabled: false,
  lastSpokenText: "",
  lastSpokenAt: 0,
  avatar: null,
  player: {
    isPlaying: false,
    timerId: null,
    index: 0,
    fps: 10,
  },
};

const elements = {
  headline: document.getElementById("headline"),
  summary: document.getElementById("summary"),
  overlayFrame: document.getElementById("overlayFrame"),
  currentIssue: document.getElementById("currentIssue"),
  scoreValue: document.getElementById("scoreValue"),
  playToggle: document.getElementById("playToggle"),
  voiceToggle: document.getElementById("voiceToggle"),
  phaseLabel: document.getElementById("phaseLabel"),
  averageScore: document.getElementById("averageScore"),
  repCount: document.getElementById("repCount"),
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

const rigColors = {
  live: "#ff574d",
  liveBone: "rgba(82, 223, 255, 0.86)",
  ref: "rgba(255, 208, 79, 0.94)",
  refBone: "rgba(255, 208, 79, 0.35)",
  grid: "rgba(82, 223, 255, 0.12)",
  text: "rgba(245, 246, 247, 0.92)",
};

function fmt(value, digits = 1, suffix = "") {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "--";
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
  const x = -(rawPoint[0] || 0);
  const y = -(rawPoint[1] || 0) + 0.2;
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
  const depth = 3.7;
  const cameraZ = 4.8;
  const scale = depth / (cameraZ - point.z);
  return {
    x: width * 0.5 + point.x * width * 0.24 * scale,
    y: height * 0.53 + point.y * height * 0.34 * scale,
    scale,
    depth: point.z,
  };
}

function drawBackdrop(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);

  const bg = ctx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, "rgba(8, 13, 18, 1)");
  bg.addColorStop(1, "rgba(4, 6, 9, 1)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  const glow = ctx.createRadialGradient(width * 0.5, height * 0.18, 10, width * 0.5, height * 0.18, width * 0.42);
  glow.addColorStop(0, "rgba(82, 223, 255, 0.18)");
  glow.addColorStop(1, "rgba(82, 223, 255, 0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = rigColors.grid;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 8; i += 1) {
    const y = height * (0.15 + i * 0.09);
    ctx.beginPath();
    ctx.moveTo(width * 0.05, y);
    ctx.lineTo(width * 0.95, y);
    ctx.stroke();
  }

  for (let i = 0; i <= 8; i += 1) {
    const x = width * (0.14 + i * 0.09);
    ctx.beginPath();
    ctx.moveTo(x, height * 0.12);
    ctx.lineTo(x, height * 0.9);
    ctx.stroke();
  }
}

function drawRig(ctx, landmarks, highlightedIndices, palette, width, height, angle, tilt) {
  if (!Array.isArray(landmarks) || !landmarks.length) return;

  const projected = landmarks.map((point) => projectPoint(transformPoint(point, angle, tilt), width, height));
  const bones = state.data.connections
    .map(([aIndex, bIndex]) => ({ a: projected[aIndex], b: projected[bIndex] }))
    .filter(({ a, b }) => a && b)
    .sort((left, right) => ((left.a.depth + left.b.depth) * 0.5) - ((right.a.depth + right.b.depth) * 0.5));

  ctx.lineCap = "round";
  bones.forEach(({ a, b }) => {
    ctx.strokeStyle = palette.bone;
    ctx.lineWidth = Math.max(1, ((a.scale + b.scale) * 0.5) * 5.5);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });

  projected
    .map((point, index) => ({ point, index }))
    .sort((left, right) => left.point.depth - right.point.depth)
    .forEach(({ point, index }) => {
      const radius = Math.max(2.2, point.scale * 7.2);
      const isHighlighted = highlightedIndices.has(index);

      ctx.fillStyle = isHighlighted ? palette.hot : palette.joint;
      ctx.beginPath();
      ctx.arc(point.x, point.y, isHighlighted ? radius * 1.28 : radius, 0, Math.PI * 2);
      ctx.fill();

      if (isHighlighted) {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.18)";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius * 1.75, 0, Math.PI * 2);
        ctx.stroke();
      }
    });
}

function renderAvatar() {
  if (!state.avatar) return;

  resizeAvatarCanvas();

  const { canvas, ctx } = state.avatar;
  const width = canvas.width;
  const height = canvas.height;
  if (!width || !height) return;

  drawBackdrop(ctx, width, height);

  const baseAngle = state.avatar.angle;
  const tilt = -0.22;
  drawRig(ctx, state.avatar.referenceLandmarks, new Set(), {
    joint: rigColors.ref,
    bone: rigColors.refBone,
    hot: rigColors.ref,
  }, width, height, baseAngle - 0.07, tilt);
  drawRig(ctx, state.avatar.wrongLandmarks, state.avatar.highlightedIndices, {
    joint: rigColors.liveBone,
    bone: "rgba(82, 223, 255, 0.92)",
    hot: rigColors.live,
  }, width, height, baseAngle + 0.03, tilt);

  ctx.fillStyle = rigColors.text;
  ctx.font = `${Math.max(12, Math.round(height * 0.036))}px "Space Grotesk", sans-serif`;
  ctx.fillText("Perspective Skeleton", width * 0.05, height * 0.08);
  ctx.fillStyle = "rgba(155, 167, 179, 0.92)";
  ctx.font = `${Math.max(11, Math.round(height * 0.026))}px "Pretendard", sans-serif`;
  ctx.fillText("wrong vs correct world landmarks", width * 0.05, height * 0.14);
}

function startAvatarLoop() {
  const tick = () => {
    if (!state.avatar) return;
    state.avatar.angle += 0.006;
    renderAvatar();
    state.avatar.rafId = window.requestAnimationFrame(tick);
  };
  tick();
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
    angle: 0.42,
    rafId: null,
  };

  const resize = () => {
    renderAvatar();
  };
  window.addEventListener("resize", resize);
  resizeAvatarCanvas();
  renderAvatar();
  startAvatarLoop();
}

function maybeSpeak(text) {
  if (!state.voiceEnabled || !text || !("speechSynthesis" in window)) return;
  const now = Date.now();
  if (state.lastSpokenText === text && now - state.lastSpokenAt < 4500) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "ko-KR";
  utterance.rate = 1;
  window.speechSynthesis.speak(utterance);
  state.lastSpokenText = text;
  state.lastSpokenAt = now;
}

function updateFrame(frame) {
  if (!frame) return;
  state.currentFrame = frame;

  const issue = frame.issues[0];
  elements.currentIssue.textContent = issue ? issue.label : "기준 자세와 거의 동일";
  elements.scoreValue.textContent = `${Math.round(frame.score)}`;
  elements.phaseLabel.textContent = `Rep ${frame.rep_index} · ${frame.phase}`;
  elements.coachText.textContent = frame.coach_text;

  renderMetrics(frame);

  const cursor = document.getElementById("timelineCursor");
  const duration = state.data.input_videos.wrong.duration_sec || 1;
  if (cursor) {
    cursor.style.left = `${(frame.time_sec / duration) * 100}%`;
  }

  const highlightedIndices = new Set(
    (frame.highlighted_joint_names || [])
      .map((name) => state.data.landmark_index[name] ?? state.data.landmark_index[`left_${name}`] ?? state.data.landmark_index[`right_${name}`])
      .filter((value) => Number.isInteger(value)),
  );

  if (state.avatar) {
    state.avatar.wrongLandmarks = frame.wrong.world_landmarks;
    state.avatar.referenceLandmarks = frame.reference.world_landmarks;
    state.avatar.highlightedIndices = highlightedIndices;
    renderAvatar();
  }

  maybeSpeak(frame.coach_text);
}

function showFrame(index, shouldSpeak = true) {
  const clamped = Math.max(0, Math.min(index, state.data.frames.length - 1));
  state.player.index = clamped;
  elements.overlayFrame.src = getFramePath(clamped);
  updateFrame(state.data.frames[clamped]);
  if (!shouldSpeak) {
    state.lastSpokenText = "";
    state.lastSpokenAt = 0;
  }
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
  showFrame(index, false);
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

  elements.headline.textContent = state.data.overview.headline;
  elements.summary.textContent = state.data.overview.summary;
  elements.averageScore.textContent = `${Math.round(state.data.overview.average_score)}`;
  elements.repCount.textContent = `${state.data.overview.rep_count}`;

  buildTimeline();
  renderFindings();
  initAvatar();
  bindControls();
  showFrame(0, false);
}

bootstrap().catch((error) => {
  console.error(error);
  elements.headline.textContent = "Prototype load failed";
  elements.summary.textContent = error.message;
});
