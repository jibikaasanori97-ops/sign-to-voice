// 指文字プリセット（実験的）
//
// 出典（手・指の形を文章で説明している箇所を突き合わせて採用）:
// - https://www.benricho.org/yubimoji/01.html （あ行〜な行の形状説明）
// - https://www.suretalk.mb.softbank.jp/column/contents/000106.php （あ・い・う・か・き・す・ぬの形状説明）
//
// 注意: これらは筆者（Claude）が画像・映像を直接見て確認したものではなく、
// 上記サイトの文章記述を根拠にしたルールです。指の本数パターンだけで
// 判定しているため、向き（手のひら/手の甲の向き）だけで区別される文字
// （例: し と す、う と と・な・に など）は誤認識・衝突を避けるため
// このプリセットには含めていません。実際の手話として使う前に、必ず
// ご自身で正しさを確認してください。

const TIP = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 };
const PIP = { index: 6, middle: 10, ring: 14, pinky: 18 };
const MCP = { thumb: 2, index: 5, middle: 9, ring: 13, pinky: 17 };
const WRIST = 0;

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function isExtended(landmarks, tipKey, pipKey) {
  const tip = landmarks[TIP[tipKey]];
  const pip = landmarks[PIP[pipKey]];
  const wrist = landmarks[WRIST];
  return dist(tip, wrist) > dist(pip, wrist) * 1.05;
}

function isThumbExtended(landmarks) {
  const tip = landmarks[TIP.thumb];
  const thumbMcp = landmarks[MCP.thumb];
  const pinkyMcp = landmarks[MCP.pinky];
  const spread = dist(tip, pinkyMcp);
  const base = dist(thumbMcp, pinkyMcp);
  return spread > base * 1.15;
}

function fingerStates(landmarks) {
  return {
    thumb: isThumbExtended(landmarks),
    index: isExtended(landmarks, "index", "index"),
    middle: isExtended(landmarks, "middle", "middle"),
    ring: isExtended(landmarks, "ring", "ring"),
    pinky: isExtended(landmarks, "pinky", "pinky"),
  };
}

// 出典: benricho.org「手のひらを相手に向けた状態で、親指を横に出す。他の指は握る。」
//       suretalk「親指だけ伸ばしてその他の指は軽く握ります」
function isA(landmarks) {
  const f = fingerStates(landmarks);
  return f.thumb && !f.index && !f.middle && !f.ring && !f.pinky;
}

// 出典: benricho.org「小指を垂直に立てる。他の指は握る。」
//       suretalk「小指以外の指は握り、小指を立てます」
function isI(landmarks) {
  const f = fingerStates(landmarks);
  return !f.thumb && !f.index && !f.middle && !f.ring && f.pinky;
}

// 出典: benricho.org「人差指と中指を立てる。他の指は握る。」
//       suretalk「人さし指と中指だけを立てます」
function isU(landmarks) {
  const f = fingerStates(landmarks);
  return !f.thumb && f.index && f.middle && !f.ring && !f.pinky;
}

// 出典: benricho.org「中指だけ垂直に立てる。」
function isSe(landmarks) {
  const f = fingerStates(landmarks);
  return !f.thumb && !f.index && f.middle && !f.ring && !f.pinky;
}

// 出典: benricho.org「指を全部立てて手のひらを相手に見せる。」
function isTe(landmarks) {
  const f = fingerStates(landmarks);
  return f.thumb && f.index && f.middle && f.ring && f.pinky;
}

// 出典: benricho.org「親指だけ内側に折り曲げる。」（他の指は伸ばした状態が前提）
function isKe(landmarks) {
  const f = fingerStates(landmarks);
  return !f.thumb && f.index && f.middle && f.ring && f.pinky;
}

// 出典: benricho.org「人差し指を垂直に立て、親指の腹を中指の第二関節に付ける。」
function isKa(landmarks) {
  const f = fingerStates(landmarks);
  if (!f.index || f.middle || f.ring || f.pinky) return false;
  const thumbTip = landmarks[TIP.thumb];
  const middlePip = landmarks[PIP.middle];
  const scale = dist(landmarks[WRIST], landmarks[MCP.middle]) || 1;
  return dist(thumbTip, middlePip) / scale < 0.5;
}

export const PRESET_SIGNS = [
  { label: "あ", test: isA },
  { label: "い", test: isI },
  { label: "う", test: isU },
  { label: "せ", test: isSe },
  { label: "て", test: isTe },
  { label: "け", test: isKe },
  { label: "か", test: isKa },
];
