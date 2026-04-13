import * as THREE from "./vendor/three.module.js";

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

function createRig(colorHex, opacity = 1, radius = 0.032) {
  const group = new THREE.Group();
  const jointMeshes = [];
  const boneMeshes = [];

  const jointMaterial = new THREE.MeshStandardMaterial({
    color: colorHex,
    emissive: colorHex,
    emissiveIntensity: opacity < 1 ? 0.2 : 0.45,
    transparent: opacity < 1,
    opacity,
    roughness: 0.35,
    metalness: 0.12,
  });
  const jointGeometry = new THREE.SphereGeometry(radius, 16, 16);

  for (let i = 0; i < 33; i += 1) {
    const mesh = new THREE.Mesh(jointGeometry, jointMaterial.clone());
    group.add(mesh);
    jointMeshes.push(mesh);
  }

  const boneGeometry = new THREE.CylinderGeometry(radius * 0.56, radius * 0.56, 1, 10);
  state.data.connections.forEach(() => {
    const mesh = new THREE.Mesh(
      boneGeometry,
      new THREE.MeshStandardMaterial({
        color: colorHex,
        emissive: colorHex,
        emissiveIntensity: opacity < 1 ? 0.12 : 0.3,
        transparent: opacity < 1,
        opacity,
        roughness: 0.45,
        metalness: 0.16,
      }),
    );
    group.add(mesh);
    boneMeshes.push(mesh);
  });

  return { group, jointMeshes, boneMeshes };
}

function toWorldVectors(landmarks) {
  if (!landmarks) return [];
  return landmarks.map((point) => new THREE.Vector3(-point[0] * 6, -point[1] * 6 + 1.1, -point[2] * 6));
}

function applyRigFrame(rig, landmarks, highlightIndices = new Set()) {
  if (!landmarks || !landmarks.length) {
    rig.group.visible = false;
    return;
  }

  rig.group.visible = true;
  const points = toWorldVectors(landmarks);
  const up = new THREE.Vector3(0, 1, 0);

  rig.jointMeshes.forEach((mesh, index) => {
    const point = points[index];
    mesh.position.copy(point);
    const isHot = highlightIndices.has(index);
    mesh.scale.setScalar(isHot ? 1.45 : 1);
    mesh.material.emissiveIntensity = isHot ? 0.8 : 0.35;
    if (mesh.material.opacity < 1) {
      mesh.material.color.setHex(0xffd04f);
    } else {
      mesh.material.color.setHex(isHot ? 0xff574d : 0x52dfff);
    }
  });

  rig.boneMeshes.forEach((mesh, idx) => {
    const [aIdx, bIdx] = state.data.connections[idx];
    const start = points[aIdx];
    const end = points[bIdx];
    const direction = new THREE.Vector3().subVectors(end, start);
    const length = direction.length();
    mesh.visible = length > 0.0001;
    if (!mesh.visible) return;
    mesh.position.copy(start).add(end).multiplyScalar(0.5);
    mesh.scale.set(1, length, 1);
    mesh.quaternion.setFromUnitVectors(up, direction.clone().normalize());
  });
}

function initAvatar() {
  const width = elements.avatarViewport.clientWidth;
  const height = elements.avatarViewport.clientHeight;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 100);
  camera.position.set(1.8, 1.5, 4.8);
  camera.lookAt(0, 1, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(width, height);
  elements.avatarViewport.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));

  const key = new THREE.DirectionalLight(0x84e4ff, 1.05);
  key.position.set(3, 6, 5);
  scene.add(key);

  const fill = new THREE.DirectionalLight(0xffd04f, 0.65);
  fill.position.set(-3, 3, -2);
  scene.add(fill);

  const grid = new THREE.GridHelper(10, 20, 0x1a3c46, 0x142129);
  grid.position.y = -1.9;
  scene.add(grid);

  const wrongRig = createRig(0x52dfff, 1, 0.04);
  const refRig = createRig(0xffd04f, 0.36, 0.028);
  scene.add(refRig.group);
  scene.add(wrongRig.group);

  state.avatar = { scene, camera, renderer, wrongRig, refRig };

  const resize = () => {
    const w = elements.avatarViewport.clientWidth;
    const h = elements.avatarViewport.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  window.addEventListener("resize", resize);

  const tick = () => {
    requestAnimationFrame(tick);
    const t = performance.now() * 0.00014;
    camera.position.x = Math.sin(t) * 0.7 + 1.5;
    camera.position.z = Math.cos(t) * 0.7 + 4.4;
    camera.lookAt(0, 0.9, 0);
    renderer.render(scene, camera);
  };
  tick();
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

  const side = state.data.input_videos.wrong.primary_side;
  const highlighted = new Set(
    frame.highlighted_joint_names
      .map((name) => state.data.landmark_index[`${side}_${name}`])
      .filter((value) => Number.isInteger(value)),
  );

  applyRigFrame(state.avatar.wrongRig, frame.wrong.world_landmarks, highlighted);
  applyRigFrame(state.avatar.refRig, frame.reference.world_landmarks);
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
