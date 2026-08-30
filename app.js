import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { PRESET_SIGNS } from "./presets.js";

const STORAGE_KEY = "sign2voice.templates.v2";
const MATCH_HISTORY_LEN = 8;
const MATCH_STABLE_RATIO = 0.7;
const SPEAK_COOLDOWN_MS = 1500;

const STATIC_CAPTURE_DURATION_MS = 1500;
const DYNAMIC_CAPTURE_DURATION_MS = 2000;
const CAPTURE_INTERVAL_MS = 80;

const DYNAMIC_KEYFRAMES = 8;
const DYNAMIC_BUFFER_MAX_MS = 2200;
const DYNAMIC_MIN_MS = 700;

const els = {
  video: document.getElementById("video"),
  overlay: document.getElementById("overlay"),
  statusBadge: document.getElementById("statusBadge"),
  liveLabel: document.getElementById("liveLabel"),
  sentence: document.getElementById("sentence"),
  speakAllBtn: document.getElementById("speakAllBtn"),
  clearBtn: document.getElementById("clearBtn"),
  labelInput: document.getElementById("labelInput"),
  captureBtn: document.getElementById("captureBtn"),
  thresholdRange: document.getElementById("thresholdRange"),
  thresholdVal: document.getElementById("thresholdVal"),
  dynamicThresholdRange: document.getElementById("dynamicThresholdRange"),
  dynamicThresholdVal: document.getElementById("dynamicThresholdVal"),
  signList: document.getElementById("signList"),
  signTypeRadios: document.getElementsByName("signType"),
  presetEnable: document.getElementById("presetEnable"),
};

const ctx = els.overlay.getContext("2d");

let handLandmarker = null;
let templates = loadTemplates();
let staticThreshold = parseFloat(els.thresholdRange.value);
let dynamicThreshold = parseFloat(els.dynamicThresholdRange.value);
let matchHistory = [];
let lastSpokenLabel = null;
let lastSpokenAt = 0;
let isCapturing = false;
let japaneseVoice = null;

let latestLandmarks = null;
let rollingFrames = []; // {landmarks, t} for dynamic recognition
let presetMatchHistory = [];

function currentSignType() {
  for (const r of els.signTypeRadios) {
    if (r.checked) return r.value;
  }
  return "static";
}

function setStatus(text, kind) {
  els.statusBadge.textContent = text;
  els.statusBadge.className = "badge" + (kind ? " " + kind : "");
}

// ---------- ローカルストレージ ----------
// templates[label] = [{ type: "static", vector: number[] } | { type: "dynamic", vector: number[], k: number }, ...]

function loadTemplates() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveTemplates() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(templates));
}

function renderSignList() {
  els.signList.innerHTML = "";
  const labels = Object.keys(templates);
  if (labels.length === 0) {
    const li = document.createElement("li");
    li.textContent = "まだ登録されたサインはありません";
    els.signList.appendChild(li);
    return;
  }
  for (const label of labels) {
    const entries = templates[label];
    const staticCount = entries.filter(e => e.type === "static").length;
    const dynamicCount = entries.filter(e => e.type === "dynamic").length;
    const li = document.createElement("li");
    const span = document.createElement("span");
    span.textContent = label;
    const countSpan = document.createElement("span");
    countSpan.className = "count";
    countSpan.textContent = `静止${staticCount}件 / 動き${dynamicCount}件`;
    const delBtn = document.createElement("button");
    delBtn.textContent = "削除";
    delBtn.addEventListener("click", () => {
      delete templates[label];
      saveTemplates();
      renderSignList();
    });
    const left = document.createElement("span");
    left.style.display = "flex";
    left.style.alignItems = "center";
    left.appendChild(span);
    left.appendChild(countSpan);
    li.appendChild(left);
    li.appendChild(delBtn);
    els.signList.appendChild(li);
  }
}

// ---------- 特徴量の正規化（形） ----------

function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];

  let pts = landmarks.map(p => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z,
  }));

  const scale = Math.hypot(pts[9].x, pts[9].y, pts[9].z) || 1;
  pts = pts.map(p => ({ x: p.x / scale, y: p.y / scale, z: p.z / scale }));

  const angle = Math.atan2(pts[9].x, -pts[9].y);
  const cos = Math.cos(-angle);
  const sin = Math.sin(-angle);
  pts = pts.map(p => ({
    x: p.x * cos - p.y * sin,
    y: p.x * sin + p.y * cos,
    z: p.z,
  }));

  return pts;
}

function toShapeVector(landmarks) {
  const norm = normalizeLandmarks(landmarks);
  const vec = [];
  for (const p of norm) vec.push(p.x, p.y, p.z);
  return vec;
}

function handScale(landmarks) {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  return Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y, middleMcp.z - wrist.z) || 1;
}

function distance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

// ---------- 動き（軌跡）の特徴量 ----------

function resampleFrames(frames, k) {
  if (frames.length === 0) return [];
  if (frames.length === 1) return new Array(k).fill(frames[0]);
  const result = [];
  for (let i = 0; i < k; i++) {
    const idx = Math.round((i / (k - 1)) * (frames.length - 1));
    result.push(frames[idx]);
  }
  return result;
}

function buildDynamicFeature(frames, k) {
  if (frames.length === 0) return null;
  const ref = frames[0].landmarks;
  const refWrist = ref[0];
  const refScale = handScale(ref);
  const resampled = resampleFrames(frames, k);
  const feat = [];
  for (const f of resampled) {
    const shape = toShapeVector(f.landmarks);
    const wrist = f.landmarks[0];
    const trajX = (wrist.x - refWrist.x) / refScale;
    const trajY = (wrist.y - refWrist.y) / refScale;
    feat.push(...shape, trajX, trajY);
  }
  return feat;
}

// ---------- マッチング ----------

function matchStaticLabel(vec) {
  let best = null;
  let bestDist = Infinity;
  for (const label of Object.keys(templates)) {
    for (const entry of templates[label]) {
      if (entry.type !== "static") continue;
      const d = distance(vec, entry.vector);
      if (d < bestDist) {
        bestDist = d;
        best = label;
      }
    }
  }
  return best !== null && bestDist <= staticThreshold ? best : null;
}

function matchDynamicLabel(vec) {
  let best = null;
  let bestDist = Infinity;
  for (const label of Object.keys(templates)) {
    for (const entry of templates[label]) {
      if (entry.type !== "dynamic" || entry.vector.length !== vec.length) continue;
      const d = distance(vec, entry.vector);
      if (d < bestDist) {
        bestDist = d;
        best = label;
      }
    }
  }
  return best !== null && bestDist <= dynamicThreshold ? best : null;
}

// ---------- 音声合成 ----------

function pickJapaneseVoice() {
  const voices = speechSynthesis.getVoices();
  japaneseVoice = voices.find(v => v.lang && v.lang.startsWith("ja")) || null;
}
speechSynthesis.onvoiceschanged = pickJapaneseVoice;
pickJapaneseVoice();

function speak(text) {
  if (!text) return;
  speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = "ja-JP";
  if (japaneseVoice) utter.voice = japaneseVoice;
  speechSynthesis.speak(utter);
}

function announceLabel(label) {
  els.sentence.value = (els.sentence.value ? els.sentence.value + " " : "") + label;
  speak(label);
  lastSpokenLabel = label;
  lastSpokenAt = performance.now();
}

// ---------- 描画 ----------

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

function drawLandmarks(landmarksList, w, h) {
  ctx.clearRect(0, 0, w, h);
  for (const landmarks of landmarksList) {
    ctx.strokeStyle = "#5ad1a3";
    ctx.lineWidth = 3;
    for (const [a, b] of HAND_CONNECTIONS) {
      const pa = landmarks[a];
      const pb = landmarks[b];
      ctx.beginPath();
      ctx.moveTo(pa.x * w, pa.y * h);
      ctx.lineTo(pb.x * w, pb.y * h);
      ctx.stroke();
    }
    ctx.fillStyle = "#ffffff";
    for (const p of landmarks) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// ---------- カメラ / モデル初期化 ----------

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  els.video.srcObject = stream;
  await new Promise(resolve => {
    els.video.onloadedmetadata = () => resolve();
  });
  await els.video.play();
  els.overlay.width = els.video.videoWidth;
  els.overlay.height = els.video.videoHeight;
}

async function setupHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
  });
}

let lastVideoTime = -1;

function predictLoop() {
  const video = els.video;
  if (video.readyState >= 2 && handLandmarker) {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const result = handLandmarker.detectForVideo(video, performance.now());
      latestLandmarks = result.landmarks && result.landmarks.length > 0 ? result.landmarks[0] : null;
      drawLandmarks(result.landmarks || [], els.overlay.width, els.overlay.height);

      if (!isCapturing) {
        updateRollingBuffer(latestLandmarks);
        handleStaticRecognition(latestLandmarks);
        handleDynamicRecognition();
        handlePresetRecognition(latestLandmarks);
      }
    }
  }
  requestAnimationFrame(predictLoop);
}

// ---------- 認識（静止） ----------

function handleStaticRecognition(landmarks) {
  if (!landmarks) {
    matchHistory = [];
    els.liveLabel.textContent = "-";
    return;
  }
  const vec = toShapeVector(landmarks);
  const label = matchStaticLabel(vec);

  matchHistory.push(label);
  if (matchHistory.length > MATCH_HISTORY_LEN) matchHistory.shift();

  els.liveLabel.textContent = label || "-";

  if (!label) return;

  const sameCount = matchHistory.filter(l => l === label).length;
  const stable = sameCount / matchHistory.length >= MATCH_STABLE_RATIO;
  if (!stable) return;

  const now = performance.now();
  if (label === lastSpokenLabel && now - lastSpokenAt < SPEAK_COOLDOWN_MS) return;

  announceLabel(label);
}

// ---------- 認識（指文字プリセット） ----------

function handlePresetRecognition(landmarks) {
  if (!els.presetEnable.checked) return;
  if (!landmarks) {
    presetMatchHistory = [];
    return;
  }

  let matched = null;
  for (const sign of PRESET_SIGNS) {
    if (sign.test(landmarks)) {
      matched = sign.label;
      break;
    }
  }

  presetMatchHistory.push(matched);
  if (presetMatchHistory.length > MATCH_HISTORY_LEN) presetMatchHistory.shift();

  if (!matched) return;

  const sameCount = presetMatchHistory.filter(l => l === matched).length;
  const stable = sameCount / presetMatchHistory.length >= MATCH_STABLE_RATIO;
  if (!stable) return;

  const now = performance.now();
  if (matched === lastSpokenLabel && now - lastSpokenAt < SPEAK_COOLDOWN_MS) return;

  announceLabel(matched);
}

// ---------- 認識（動き） ----------

function updateRollingBuffer(landmarks) {
  const now = performance.now();
  if (!landmarks) {
    rollingFrames = [];
    return;
  }
  rollingFrames.push({ landmarks, t: now });
  rollingFrames = rollingFrames.filter(f => now - f.t <= DYNAMIC_BUFFER_MAX_MS);
}

function handleDynamicRecognition() {
  if (rollingFrames.length < 2) return;
  const span = rollingFrames[rollingFrames.length - 1].t - rollingFrames[0].t;
  if (span < DYNAMIC_MIN_MS) return;

  const now = performance.now();
  if (lastSpokenLabel && now - lastSpokenAt < SPEAK_COOLDOWN_MS) return;

  const vec = buildDynamicFeature(rollingFrames, DYNAMIC_KEYFRAMES);
  const label = matchDynamicLabel(vec);
  if (!label) return;

  announceLabel(label);
  rollingFrames = []; // 連続発火を防ぐため軌跡をリセット
}

// ---------- 登録（キャプチャ） ----------

async function captureSign() {
  const label = els.labelInput.value.trim();
  if (!label) {
    alert("読み上げテキストを入力してください");
    return;
  }
  const type = currentSignType();
  const duration = type === "dynamic" ? DYNAMIC_CAPTURE_DURATION_MS : STATIC_CAPTURE_DURATION_MS;

  isCapturing = true;
  els.captureBtn.disabled = true;
  const collected = [];
  const startedAt = performance.now();

  await new Promise(resolve => {
    const timer = setInterval(() => {
      if (latestLandmarks) {
        collected.push({ landmarks: latestLandmarks, t: performance.now() });
      }
      if (performance.now() - startedAt >= duration) {
        clearInterval(timer);
        resolve();
      }
    }, CAPTURE_INTERVAL_MS);
  });

  isCapturing = false;
  els.captureBtn.disabled = false;

  if (collected.length === 0) {
    alert("手が検出できませんでした。カメラに手を映してもう一度お試しください。");
    return;
  }

  if (!templates[label]) templates[label] = [];

  if (type === "dynamic") {
    const vec = buildDynamicFeature(collected, DYNAMIC_KEYFRAMES);
    templates[label].push({ type: "dynamic", vector: vec, k: DYNAMIC_KEYFRAMES });
  } else {
    const dims = collected.length ? toShapeVector(collected[0].landmarks).length : 0;
    const avg = new Array(dims).fill(0);
    for (const f of collected) {
      const v = toShapeVector(f.landmarks);
      for (let i = 0; i < dims; i++) avg[i] += v[i] / collected.length;
    }
    templates[label].push({ type: "static", vector: avg });
  }

  saveTemplates();
  renderSignList();
  els.liveLabel.textContent = "登録完了✔";
}

function updateCaptureButtonLabel() {
  const type = currentSignType();
  els.captureBtn.textContent =
    type === "dynamic"
      ? "🤟 今の手の動きを登録（2秒間動かす）"
      : "✋ 今の手の形を登録（1.5秒間キープ）";
}

// ---------- イベント ----------

els.captureBtn.addEventListener("click", captureSign);

for (const r of els.signTypeRadios) {
  r.addEventListener("change", updateCaptureButtonLabel);
}

els.thresholdRange.addEventListener("input", () => {
  staticThreshold = parseFloat(els.thresholdRange.value);
  els.thresholdVal.textContent = staticThreshold.toFixed(2);
});

els.dynamicThresholdRange.addEventListener("input", () => {
  dynamicThreshold = parseFloat(els.dynamicThresholdRange.value);
  els.dynamicThresholdVal.textContent = dynamicThreshold.toFixed(2);
});

els.speakAllBtn.addEventListener("click", () => {
  speak(els.sentence.value);
});

els.clearBtn.addEventListener("click", () => {
  els.sentence.value = "";
  lastSpokenLabel = null;
});

// ---------- 起動 ----------

async function main() {
  try {
    setStatus("カメラを起動中…");
    await setupCamera();
    setStatus("モデルを読み込み中…");
    await setupHandLandmarker();
    setStatus("認識中", "ok");
    renderSignList();
    updateCaptureButtonLabel();
    els.thresholdVal.textContent = staticThreshold.toFixed(2);
    els.dynamicThresholdVal.textContent = dynamicThreshold.toFixed(2);
    predictLoop();
  } catch (err) {
    console.error(err);
    setStatus("エラー: " + err.message, "err");
  }
}

main();
