import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { PRESET_SIGNS } from "./presets.js";

const STORAGE_KEY = "sign2voice.templates.v3";
const VOICE_STORAGE_KEY = "sign2voice.voice.v1";
const MATCH_HISTORY_LEN = 8;
const MATCH_STABLE_RATIO = 0.7;
const SPEAK_COOLDOWN_MS = 1500;

const STATIC_CAPTURE_DURATION_MS = 1500;
const DYNAMIC_CAPTURE_DURATION_MS = 2000;
const CAPTURE_INTERVAL_MS = 80;

const DYNAMIC_KEYFRAMES = 8;
const DYNAMIC_BUFFER_MAX_MS = 2200;
const DYNAMIC_MIN_MS = 700;

const HAND_SHAPE_DIMS = 63; // 21 landmarks * (x,y,z)

const els = {
  video: document.getElementById("video"),
  overlay: document.getElementById("overlay"),
  flipCameraBtn: document.getElementById("flipCameraBtn"),
  captureProgress: document.getElementById("captureProgress"),
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
  voiceSelect: document.getElementById("voiceSelect"),
  rateRange: document.getElementById("rateRange"),
  rateVal: document.getElementById("rateVal"),
};

const ctx = els.overlay.getContext("2d");

let handLandmarker = null;
let templates = loadTemplates();
let staticThreshold = parseFloat(els.thresholdRange.value);
let dynamicThreshold = parseFloat(els.dynamicThresholdRange.value);
let matchHistory = [];
let lastSpokenLabel = null;
let lastSpokenAt = 0;
let facingMode = "user";
let selectedVoiceURI = null;
let speakRate = parseFloat(els.rateRange.value);

// latestHands: { left, right, primary, count } — landmarks are raw MediaPipe points or null
let latestHands = { left: null, right: null, primary: null, count: 0 };
let rollingFrames = []; // {hands, t} for dynamic recognition
let presetMatchHistory = [];
let captureState = null; // set while a training capture is running

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
// templates[label] = [{ type: "static", vector: number[126] } | { type: "dynamic", vector: number[], k: number }, ...]
// vector は 左手63次元 + 右手63次元 を連結したもの（不在の手は0埋め）。動きサインはそれに各キーフレームの軌跡2次元(主に検出された手の移動)を加える。

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

// ---------- 特徴量の正規化（片手の形） ----------

function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];

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

// ---------- 両手の結合特徴量 ----------

function combinedShapeVector(hands) {
  const leftVec = hands.left ? toShapeVector(hands.left) : new Array(HAND_SHAPE_DIMS).fill(0);
  const rightVec = hands.right ? toShapeVector(hands.right) : new Array(HAND_SHAPE_DIMS).fill(0);
  return [...leftVec, ...rightVec];
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
  const refHands = frames[0].hands;
  const refPrimary = refHands.primary;
  const refWrist = refPrimary ? refPrimary[0] : null;
  const refScale = refPrimary ? handScale(refPrimary) : 1;
  const resampled = resampleFrames(frames, k);
  const feat = [];
  for (const f of resampled) {
    feat.push(...combinedShapeVector(f.hands));
    if (refWrist && f.hands.primary) {
      const wrist = f.hands.primary[0];
      feat.push((wrist.x - refWrist.x) / refScale, (wrist.y - refWrist.y) / refScale);
    } else {
      feat.push(0, 0);
    }
  }
  return feat;
}

// ---------- マッチング ----------

function matchStaticLabel(vec) {
  let best = null;
  let bestDist = Infinity;
  for (const label of Object.keys(templates)) {
    for (const entry of templates[label]) {
      if (entry.type !== "static" || entry.vector.length !== vec.length) continue;
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

function populateVoiceList() {
  const voices = speechSynthesis.getVoices();
  const jaVoices = voices.filter(v => v.lang && v.lang.startsWith("ja"));
  const list = jaVoices.length > 0 ? jaVoices : voices;

  const prevSelection = els.voiceSelect.value;
  els.voiceSelect.innerHTML = "";
  for (const v of list) {
    const opt = document.createElement("option");
    opt.value = v.voiceURI;
    opt.textContent = `${v.name} (${v.lang})`;
    els.voiceSelect.appendChild(opt);
  }

  if (selectedVoiceURI && list.some(v => v.voiceURI === selectedVoiceURI)) {
    els.voiceSelect.value = selectedVoiceURI;
  } else if (prevSelection && list.some(v => v.voiceURI === prevSelection)) {
    els.voiceSelect.value = prevSelection;
  } else if (list.length > 0) {
    selectedVoiceURI = list[0].voiceURI;
    els.voiceSelect.value = selectedVoiceURI;
  }
}

speechSynthesis.onvoiceschanged = populateVoiceList;

function currentVoice() {
  const voices = speechSynthesis.getVoices();
  return voices.find(v => v.voiceURI === els.voiceSelect.value) || null;
}

function speak(text) {
  if (!text) return;
  speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = "ja-JP";
  utter.rate = speakRate;
  const voice = currentVoice();
  if (voice) utter.voice = voice;
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

function applyMirror() {
  const mirror = facingMode === "user";
  els.video.classList.toggle("mirrored", mirror);
  els.overlay.classList.toggle("mirrored", mirror);
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  els.video.srcObject = stream;
  await new Promise(resolve => {
    els.video.onloadedmetadata = () => resolve();
  });
  await els.video.play();
  els.overlay.width = els.video.videoWidth;
  els.overlay.height = els.video.videoHeight;
  applyMirror();
}

async function switchCamera() {
  els.flipCameraBtn.disabled = true;
  try {
    const oldStream = els.video.srcObject;
    if (oldStream) {
      oldStream.getTracks().forEach(t => t.stop());
    }
    facingMode = facingMode === "user" ? "environment" : "user";
    lastVideoTime = -1;
    await setupCamera();
  } catch (err) {
    console.error(err);
    facingMode = facingMode === "user" ? "environment" : "user"; // ロールバック
    try {
      await setupCamera();
      setStatus("認識中", "ok");
    } catch (err2) {
      console.error(err2);
      setStatus("カメラ切替エラー: " + err2.message, "err");
    }
  } finally {
    els.flipCameraBtn.disabled = false;
  }
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
    numHands: 2,
  });
}

let lastVideoTime = -1;

function updateLatestHands(result) {
  let left = null;
  let right = null;
  const list = result.landmarks || [];
  const handedness = result.handedness || [];
  for (let i = 0; i < list.length; i++) {
    const label = handedness[i] && handedness[i][0] ? handedness[i][0].categoryName : null;
    if (label === "Left" && !left) left = list[i];
    else if (label === "Right" && !right) right = list[i];
    else if (!left) left = list[i];
    else if (!right) right = list[i];
  }
  latestHands = { left, right, primary: left || right || null, count: list.length };
}

function predictLoop() {
  const video = els.video;
  if (video.readyState >= 2 && handLandmarker) {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const result = handLandmarker.detectForVideo(video, performance.now());
      updateLatestHands(result);
      drawLandmarks(result.landmarks || [], els.overlay.width, els.overlay.height);

      if (captureState) {
        advanceCapture();
      } else {
        updateRollingBuffer();
        handleStaticRecognition();
        handleDynamicRecognition();
        handlePresetRecognition();
      }
    }
  }
  requestAnimationFrame(predictLoop);
}

// ---------- 認識(静止) ----------

function handleStaticRecognition() {
  if (!latestHands.primary) {
    matchHistory = [];
    els.liveLabel.textContent = "-";
    return;
  }
  const vec = combinedShapeVector(latestHands);
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

// ---------- 認識(指文字プリセット、片手のみ) ----------

function handlePresetRecognition() {
  if (!els.presetEnable.checked) return;
  const landmarks = latestHands.primary;
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

// ---------- 認識(動き) ----------

function updateRollingBuffer() {
  const now = performance.now();
  if (!latestHands.primary) {
    rollingFrames = [];
    return;
  }
  rollingFrames.push({ hands: latestHands, t: now });
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

// ---------- 登録（キャプチャ、predictLoopに同期して収集） ----------

function advanceCapture() {
  const now = performance.now();
  if (now - captureState.lastSampleAt >= CAPTURE_INTERVAL_MS) {
    captureState.lastSampleAt = now;
    if (latestHands.primary) {
      captureState.collected.push({ hands: latestHands, t: now });
    }
  }

  const elapsed = now - captureState.startedAt;
  const remain = Math.max(0, (captureState.duration - elapsed) / 1000);
  els.captureProgress.hidden = false;
  els.captureProgress.textContent = `登録中… 残り${remain.toFixed(1)}秒`;

  if (elapsed >= captureState.duration) {
    const collected = captureState.collected;
    const resolve = captureState.resolve;
    captureState = null;
    els.captureProgress.hidden = true;
    resolve(collected);
  }
}

function startCapture(duration) {
  return new Promise(resolve => {
    captureState = {
      duration,
      startedAt: performance.now(),
      lastSampleAt: 0,
      collected: [],
      resolve,
    };
  });
}

async function captureSign() {
  const label = els.labelInput.value.trim();
  if (!label) {
    alert("読み上げテキストを入力してください");
    return;
  }
  const type = currentSignType();
  const duration = type === "dynamic" ? DYNAMIC_CAPTURE_DURATION_MS : STATIC_CAPTURE_DURATION_MS;

  els.captureBtn.disabled = true;
  const collected = await startCapture(duration);
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
    const dims = HAND_SHAPE_DIMS * 2;
    const avg = new Array(dims).fill(0);
    for (const f of collected) {
      const v = combinedShapeVector(f.hands);
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
els.flipCameraBtn.addEventListener("click", switchCamera);

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

els.voiceSelect.addEventListener("change", () => {
  selectedVoiceURI = els.voiceSelect.value;
  localStorage.setItem(VOICE_STORAGE_KEY, JSON.stringify({ voiceURI: selectedVoiceURI, rate: speakRate }));
});

els.rateRange.addEventListener("input", () => {
  speakRate = parseFloat(els.rateRange.value);
  els.rateVal.textContent = speakRate.toFixed(2);
  localStorage.setItem(VOICE_STORAGE_KEY, JSON.stringify({ voiceURI: selectedVoiceURI, rate: speakRate }));
});

// ---------- 起動 ----------

function loadVoicePrefs() {
  try {
    const raw = localStorage.getItem(VOICE_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (parsed.voiceURI) selectedVoiceURI = parsed.voiceURI;
    if (typeof parsed.rate === "number") speakRate = parsed.rate;
  } catch {
    // 無視
  }
}

async function main() {
  try {
    loadVoicePrefs();
    els.rateRange.value = String(speakRate);
    els.rateVal.textContent = speakRate.toFixed(2);
    populateVoiceList();

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
