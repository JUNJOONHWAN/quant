#!/usr/bin/env node
/**
 * Compute Fusion-Newgate (aggressive_plus) regime series from precomputed data.
 * Ported from stock_bench/regime_core/calculations.py without numpy/pandas deps.
 */

const REQUIRED_PRICE_SYMBOLS = ['QQQ', 'IWM', 'SPY', 'TLT', 'GLD', 'BTC-USD'];
const SIGNAL_SYMBOLS = ['IWM', 'SPY', 'TLT', 'GLD', 'BTC-USD'];
const DEFAULT_FUSION_WINDOW = 30;

const BASE_PRESET = {
  on_thr: 0.30,
  off_thr: -0.30,
  a_chi: 0.40,
  b_eta: 0.20,
  c_R: 0.30,
  d_dR: 0.10,
  kappa: 2.5,
  theta: 0.0,
  wmin: 0.10,
  wmax: 0.90,
  shock_z: 2.0,
  neutral_band: [0.33, 0.66],
  ew_tilt: 0.80,
  ew_dR: 0.80,
  ew_chi: 0.60,
  ew_FQI: 0.0,
  ew_dT: -0.25,
  G1: 0.66,
  dr_tilt: 1.10,
  dr_dR: 1.10,
  dr_chi: 0.90,
  dr_FQI: -0.25,
  G2: 0.50,
  cr_enabled: true,
  cr_dR: 0.90,
  cr_tilt: 0.80,
  cr_eta: 0.80,
  cr_FQI: 0.0,
  cr_zR: 0.40,
  cr_rsi_max: 50.0,
  G3: 0.50,
  cr_hold_days: 3,
  use_classic_off_gate: false,
  use_ffl_off_gate: false,
  off_cap: 0.20,
  use_ffl_stab_off_gate: false,
  stab_off_cap: 0.0,
  use_ffl_stab_soft_cap: false,
  stab_soft_cap: 0.40,
};

const FUSION_PRESETS = {
  aggressive_plus: {
    kappa: 3.0,
    wmin: 0.05,
    wmax: 0.98,
    ew_tilt: 1.05,
    ew_dR: 1.05,
    ew_chi: 0.75,
    ew_FQI: -0.15,
    ew_dT: -0.35,
    G1: 1.00,
    dr_tilt: 1.25,
    dr_dR: 1.25,
    dr_chi: 1.10,
    dr_FQI: -0.35,
    G2: 1.00,
    cr_enabled: true,
    cr_dR: 1.10,
    cr_tilt: 0.95,
    cr_eta: 0.95,
    cr_FQI: -0.15,
    cr_zR: 0.60,
    cr_rsi_max: 45.0,
    G3: 0.80,
    cr_hold_days: 2,
    shock_z: 2.4,
    neutral_band: [0.30, 0.60],
    on_thr: 0.20,
    off_thr: -0.30,
  },
};

const RISK_CFG_FFL = {
  lookbacks: { momentum: 10, vol: 20, breadth: 5 },
  p: 1.5,
  zSat: 2.0,
  lambda: 0.25,
  thresholds: {
    jOn: +0.10,
    jOff: -0.08,
    scoreOn: 0.60,
    scoreOff: 0.40,
    breadthOn: 0.50,
    mmFragile: 0.88,
    mmOff: 0.96,
    pconOn: 0.55,
    pconOff: 0.40,
    mmHi: 0.90,
    downAll: 0.60,
    corrConeDays: 5,
    driftMinDays: 3,
    driftCool: 2,
    vOn: +0.05,
    vOff: -0.05,
  },
  stabTune: {
    fast: 21,
    slow: 63,
    zWin: 126,
    zUp: 2.50,
    zDown: 2.50,
    slopeMin: 0.02,
    neutralLo: 0.30,
    neutralHi: 0.40,
    lagUp: 3,
    lagDown: 4,
    onFluxEase: 0.02,
    confirmOnMin: 2,
    leadOnWindow: 6,
    downGrace: 6,
    hazardWindow: 9,
    offFluxTighten: 0.03,
    confirmOffMin: 1,
    onOverrideMargin: 0.01,
    upOffHarden: 0.02,
    upConfirmOffMin: 2,
  },
};

function cloneArrayLength(arr, length) {
  const out = new Array(length);
  for (let i = 0; i < length; i += 1) {
    const value = Array.isArray(arr) && i < arr.length ? Number(arr[i]) : Number.NaN;
    out[i] = Number.isFinite(value) ? value : Number.NaN;
  }
  return out;
}

function ensurePriceMap(priceSeries, length) {
  if (!priceSeries || typeof priceSeries !== 'object') return null;
  const out = {};
  for (const sym of REQUIRED_PRICE_SYMBOLS) {
    if (!Array.isArray(priceSeries[sym])) {
      return null;
    }
    out[sym] = cloneArrayLength(priceSeries[sym], length);
  }
  return out;
}

function meanFinite(values) {
  let sum = 0;
  let count = 0;
  values.forEach((value) => {
    if (Number.isFinite(value)) {
      sum += value;
      count += 1;
    }
  });
  if (count === 0) return Number.NaN;
  return sum / count;
}

function rollingMean(values, window) {
  const out = new Array(values.length).fill(Number.NaN);
  for (let i = 0; i < values.length; i += 1) {
    const start = Math.max(0, i - window + 1);
    let sum = 0;
    let count = 0;
    for (let j = start; j <= i; j += 1) {
      const v = values[j];
      if (Number.isFinite(v)) {
        sum += v;
        count += 1;
      }
    }
    out[i] = count > 0 ? sum / count : Number.NaN;
  }
  return out;
}

function rollingStd(values, window) {
  const means = rollingMean(values, window);
  const out = new Array(values.length).fill(Number.NaN);
  for (let i = 0; i < values.length; i += 1) {
    if (i < window - 1 || !Number.isFinite(means[i])) continue;
    const start = Math.max(0, i - window + 1);
    let sumSq = 0;
    let count = 0;
    for (let j = start; j <= i; j += 1) {
      const v = values[j];
      if (Number.isFinite(v)) {
        const diff = v - means[i];
        sumSq += diff * diff;
        count += 1;
      }
    }
    out[i] = count > 1 ? Math.sqrt(sumSq / count) : Number.NaN;
  }
  return out;
}

function zscoreSeries(values, window = 252, minp = 63) {
  const means = rollingMean(values, window);
  const stds = rollingStd(values, window);
  return values.map((value, idx) => {
    if (idx < minp - 1) return Number.NaN;
    const mean = means[idx];
    const std = stds[idx];
    if (!Number.isFinite(mean) || !Number.isFinite(std) || std === 0 || !Number.isFinite(value)) {
      return Number.NaN;
    }
    return (value - mean) / std;
  });
}

function pctRank(values, window = 252, minp = 63) {
  const out = new Array(values.length).fill(Number.NaN);
  for (let i = 0; i < values.length; i += 1) {
    const start = Math.max(0, i - window + 1);
    const win = [];
    for (let j = start; j <= i; j += 1) {
      const val = values[j];
      if (Number.isFinite(val)) {
        win.push(val);
      }
    }
    if (win.length < minp) {
      out[i] = Number.NaN;
      continue;
    }
    if (win.length === 1) {
      out[i] = Number.NaN;
      continue;
    }
    win.sort((a, b) => a - b);
    const last = values[i];
    if (!Number.isFinite(last)) {
      out[i] = Number.NaN;
      continue;
    }
    let rank = -1;
    for (let k = 0; k < win.length; k += 1) {
      if (win[k] <= last) {
        rank += 1;
      }
    }
    out[i] = rank >= 0 ? rank / (win.length - 1) : 0;
  }
  return out;
}

function rsi(series, period = 14) {
  const gains = [];
  const losses = [];
  const out = new Array(series.length).fill(Number.NaN);
  let prev = null;
  for (let i = 0; i < series.length; i += 1) {
    const price = series[i];
    if (!Number.isFinite(price)) {
      prev = price;
      continue;
    }
    if (!Number.isFinite(prev)) {
      prev = price;
      continue;
    }
    const change = price - prev;
    prev = price;
    gains.push(Math.max(0, change));
    losses.push(Math.max(0, -change));
    if (gains.length > period) gains.shift();
    if (losses.length > period) losses.shift();
    if (gains.length < period) continue;
    const avgGain = gains.reduce((sum, v) => sum + v, 0) / gains.length;
    const avgLoss = losses.reduce((sum, v) => sum + v, 0) / losses.length;
    if (avgLoss <= 1e-12) {
      out[i] = 100;
    } else {
      const rs = avgGain / avgLoss;
      out[i] = 100 - 100 / (1 + rs);
    }
  }
  return out;
}

function rollingSlope(series, window = 63, minp = 20) {
  const out = new Array(series.length).fill(Number.NaN);
  for (let i = 0; i < series.length; i += 1) {
    const start = Math.max(0, i - window + 1);
    const slice = [];
    for (let j = start; j <= i; j += 1) {
      const val = series[j];
      if (Number.isFinite(val)) {
        slice.push(val);
      }
    }
    if (slice.length < Math.max(minp, Math.floor(window * 0.3))) {
      continue;
    }
    const xs = slice.map((_, idx) => idx);
    const meanX = xs.reduce((sum, v) => sum + v, 0) / xs.length;
    const meanY = slice.reduce((sum, v) => sum + v, 0) / slice.length;
    let num = 0;
    let den = 0;
    for (let k = 0; k < xs.length; k += 1) {
      const dx = xs[k] - meanX;
      const dy = slice[k] - meanY;
      num += dx * dy;
      den += dx * dx;
    }
    if (den <= 0) continue;
    out[i] = num / den;
  }
  return out;
}

function logReturns(series) {
  const out = new Array(series.length).fill(Number.NaN);
  for (let i = 1; i < series.length; i += 1) {
    const prev = series[i - 1];
    const cur = series[i];
    if (Number.isFinite(prev) && Number.isFinite(cur) && prev !== 0) {
      out[i] = Math.log(cur / prev);
    } else {
      out[i] = Number.NaN;
    }
  }
  return out;
}

function arReturns(series) {
  const out = new Array(series.length).fill(0);
  for (let i = 1; i < series.length; i += 1) {
    const prev = series[i - 1];
    const cur = series[i];
    if (Number.isFinite(prev) && Number.isFinite(cur) && prev !== 0) {
      out[i] = cur / prev - 1;
    } else {
      out[i] = 0;
    }
  }
  return out;
}

function corrSafe(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || a.length < 2) {
    return Number.NaN;
  }
  const xs = [];
  const ys = [];
  for (let i = 0; i < a.length; i += 1) {
    const av = a[i];
    const bv = b[i];
    if (Number.isFinite(av) && Number.isFinite(bv)) {
      xs.push(av);
      ys.push(bv);
    }
  }
  if (xs.length < 2) return Number.NaN;
  const meanX = xs.reduce((sum, v) => sum + v, 0) / xs.length;
  const meanY = ys.reduce((sum, v) => sum + v, 0) / ys.length;
  let num = 0;
  let varX = 0;
  let varY = 0;
  for (let i = 0; i < xs.length; i += 1) {
    const dx = xs[i] - meanX;
    const dy = ys[i] - meanY;
    num += dx * dy;
    varX += dx * dx;
    varY += dy * dy;
  }
  if (varX <= 0 || varY <= 0) return 0;
  return Math.max(-1, Math.min(1, num / Math.sqrt(varX * varY)));
}

function dominantEigenvalue(matrix, maxIter = 50, tol = 1e-6) {
  const n = matrix.length;
  if (n === 0) return 0;
  let vec = new Array(n).fill(1 / Math.sqrt(n));
  let eigen = 0;
  for (let iter = 0; iter < maxIter; iter += 1) {
    const w = new Array(n).fill(0);
    for (let i = 0; i < n; i += 1) {
      let sum = 0;
      for (let j = 0; j < n; j += 1) {
        sum += matrix[i][j] * vec[j];
      }
      w[i] = sum;
    }
    const norm = Math.sqrt(w.reduce((sum, v) => sum + v * v, 0));
    if (!Number.isFinite(norm) || norm === 0) break;
    const next = w.map((v) => v / norm);
    const diff = Math.sqrt(next.reduce((sum, v, idx) => sum + (v - vec[idx]) ** 2, 0));
    vec = next;
    let dot = 0;
    for (let i = 0; i < n; i += 1) {
      let mv = 0;
      for (let j = 0; j < n; j += 1) {
        mv += matrix[i][j] * vec[j];
      }
      dot += vec[i] * mv;
    }
    eigen = dot;
    if (diff < tol) break;
  }
  return eigen;
}

function absorptionRatio(windowReturns) {
  const n = windowReturns.length;
  const len = windowReturns[0]?.length || 0;
  if (n === 0 || len === 0) return Number.NaN;
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < len; j += 1) {
      const v = windowReturns[i][j];
      if (!Number.isFinite(v)) {
        return Number.NaN;
      }
    }
  }
  const cov = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      if (i === j) {
        cov[i][j] = 1;
      } else {
        cov[i][j] = corrSafe(windowReturns[i], windowReturns[j]);
      }
    }
  }
  const top = dominantEigenvalue(cov);
  const trace = cov.reduce((sum, row, idx) => sum + (row[idx] || 0), 0);
  if (!Number.isFinite(top) || !Number.isFinite(trace) || trace === 0) {
    return Number.NaN;
  }
  return top / trace;
}

function pnlSeries(exposure, returns) {
  const out = new Array(returns.length).fill(0);
  for (let i = 1; i < returns.length; i += 1) {
    const exp = exposure[i - 1] || 0;
    out[i] = exp * returns[i];
  }
  return out;
}

function rollingIR(pnl, window = 63) {
  const out = new Array(pnl.length).fill(Number.NaN);
  for (let i = 0; i < pnl.length; i += 1) {
    const start = Math.max(0, i - window + 1);
    const slice = [];
    for (let j = start; j <= i; j += 1) {
      if (Number.isFinite(pnl[j])) {
        slice.push(pnl[j]);
      }
    }
    if (slice.length < 30) continue;
    const mean = slice.reduce((sum, v) => sum + v, 0) / slice.length;
    const variance = slice.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (slice.length - 1);
    const std = Math.sqrt(Math.max(variance, 0));
    out[i] = std > 1e-12 ? mean / std : Number.NaN;
  }
  return out;
}

function selectMatrixValue(record, symA, symB) {
  const symbols = SIGNAL_SYMBOLS;
  const idxMap = new Map(symbols.map((sym, idx) => [sym, idx]));
  const idxA = idxMap.get(symA);
  const idxB = idxMap.get(symB);
  if (idxA == null || idxB == null) return Number.NaN;
  const matrix = record?.matrix;
  if (!Array.isArray(matrix)) return Number.NaN;
  const row = matrix[idxA];
  if (!Array.isArray(row)) return Number.NaN;
  const value = row[idxB];
  return Number.isFinite(value) ? value : Number.NaN;
}

function computeClassicSeries(metrics, analysisDates) {
  const records = metrics?.records || [];
  const L = analysisDates.length;
  const offset = Math.max(0, L - records.length);
  const scCorr = new Array(L).fill(0);
  const safeNeg = new Array(L).fill(0);
  for (let i = 0; i < records.length; i += 1) {
    const rec = records[i];
    const corr = selectMatrixValue(rec, 'IWM', 'BTC-USD');
    const safe = Number.isFinite(rec?.sub?.safeNegative) ? rec.sub.safeNegative : 0;
    scCorr[offset + i] = Number.isFinite(corr) ? corr : 0;
    safeNeg[offset + i] = safe;
  }
  for (let i = 0; i < offset; i += 1) {
    scCorr[i] = scCorr[offset] || 0;
    safeNeg[i] = safeNeg[offset] || 0;
  }
  const weights = { sc: 0.70, safe: 0.30 };
  const thresholds = { scoreOn: 0.65, scoreOff: 0.30, corrOn: 0.50, corrMinOn: 0.20, corrOff: -0.10 };
  const score = scCorr.map((sc, idx) => {
    const safe = safeNeg[idx] || 0;
    return Math.max(0, Math.min(1, weights.sc * Math.max(0, sc) + weights.safe * safe));
  });
  const state = scCorr.map((sc, idx) => {
    const s = score[idx];
    if (sc >= thresholds.corrOn) return 1;
    if (sc <= thresholds.corrOff || s <= thresholds.scoreOff) return -1;
    if (s >= thresholds.scoreOn && sc >= thresholds.corrMinOn) return 1;
    return 0;
  });
  return { score, state, scCorr, safeNeg };
}

function buildPreset(name) {
  const base = { ...BASE_PRESET };
  if (name && FUSION_PRESETS[name]) {
    Object.assign(base, FUSION_PRESETS[name]);
  }
  return base;
}

function valueOrNull(value) {
  return Number.isFinite(value) ? Number(value) : null;
}

function computeFusionNewgate({
  analysisDates,
  priceSeries,
  metrics,
  window = DEFAULT_FUSION_WINDOW,
  preset = 'aggressive_plus',
} = {}) {
  if (!Array.isArray(analysisDates) || analysisDates.length === 0) {
    return null;
  }
  const L = analysisDates.length;
  const priceMap = ensurePriceMap(priceSeries, L);
  if (!priceMap) return null;
  const classic = computeClassicSeries(metrics, analysisDates);
  const cfg = buildPreset(preset);

  const QQQ = priceMap['QQQ'];
  const IWM = priceMap['IWM'];
  const SPY = priceMap['SPY'];
  const TLT = priceMap['TLT'];
  const GLD = priceMap['GLD'];
  const BTC = priceMap['BTC-USD'];

  const rI = logReturns(IWM);
  const rS = logReturns(SPY);
  const rT = logReturns(TLT);
  const rG = logReturns(GLD);
  const rB = logReturns(BTC);
  const zI = zscoreSeries(rI);
  const zS = zscoreSeries(rS);
  const zT = zscoreSeries(rT);
  const zG = zscoreSeries(rG);
  const zB = zscoreSeries(rB);

  const flow = new Array(L).fill(Number.NaN);
  for (let i = 0; i < L; i += 1) {
    const pos = meanFinite([zI[i], zS[i], zB[i]]);
    const neg = meanFinite([zT[i], zG[i]]);
    if (Number.isNaN(pos) && Number.isNaN(neg)) {
      flow[i] = Number.NaN;
    } else {
      const posVal = Number.isFinite(pos) ? pos : 0;
      const negVal = Number.isFinite(neg) ? neg : 0;
      flow[i] = posVal - negVal;
    }
  }

  const x = new Array(L).fill(Number.NaN);
  let acc = 0;
  for (let i = 0; i < L; i += 1) {
    if (Number.isFinite(flow[i])) {
      acc += flow[i];
      x[i] = acc;
    } else {
      x[i] = Number.NaN;
    }
  }

  const g = new Array(L).fill(0);
  for (let i = 1; i < L; i += 1) {
    const prev = flow[i - 1];
    const cur = flow[i];
    if (Number.isFinite(prev) && Number.isFinite(cur)) {
      g[i] = cur - prev;
    } else {
      g[i] = 0;
    }
  }

  const gPos = g.map((v) => Math.max(0, v));
  const gNeg = g.map((v) => Math.max(0, -v));
  const W = 63;
  const DPlus = gPos.map((_, idx) => {
    const squares = [];
    const start = Math.max(0, idx - W + 1);
    for (let j = start; j <= idx; j += 1) {
      squares.push(gPos[j] ** 2);
    }
    const mean = meanFinite(squares);
    return Math.sqrt(Math.max(0, Number.isFinite(mean) ? mean : 0));
  });
  const DMinus = gNeg.map((_, idx) => {
    const squares = [];
    const start = Math.max(0, idx - W + 1);
    for (let j = start; j <= idx; j += 1) {
      squares.push(gNeg[j] ** 2);
    }
    const mean = meanFinite(squares);
    return Math.sqrt(Math.max(0, Number.isFinite(mean) ? mean : 0));
  });

  const mu = rollingSlope(x, W);
  const muPos = mu.map((v) => (Number.isFinite(v) && v > 0 ? v : 0));
  const muNeg = mu.map((v) => (Number.isFinite(v) && v < 0 ? -v : 0));

  const JPlus = new Array(L).fill(0);
  const JMinus = new Array(L).fill(0);
  for (let i = 0; i < L; i += 1) {
    JPlus[i] = muPos[i] - DPlus[i] * gPos[i];
    JMinus[i] = muNeg[i] - DMinus[i] * gNeg[i];
  }
  const J = JPlus.map((val, idx) => val - JMinus[idx]);

  const R = new Array(L).fill(Number.NaN);
  for (let i = W - 1; i < L; i += 1) {
    const win = [
      rI.slice(i - W + 1, i + 1),
      rS.slice(i - W + 1, i + 1),
      rT.slice(i - W + 1, i + 1),
      rG.slice(i - W + 1, i + 1),
      rB.slice(i - W + 1, i + 1),
    ];
    R[i] = absorptionRatio(win);
  }
  const dR = R.map((val, idx) => {
    if (idx === 0) return Number.NaN;
    const prev = R[idx - 1];
    if (Number.isFinite(val) && Number.isFinite(prev)) {
      return Math.abs(val - prev);
    }
    return Number.NaN;
  });

  const eps = 1e-8;
  const chi = new Array(L).fill(Number.NaN);
  const eta = new Array(L).fill(Number.NaN);
  for (let i = 0; i < L; i += 1) {
    const numChi = DPlus[i] + DMinus[i];
    const denomChi = muPos[i] + muNeg[i] + eps;
    chi[i] = denomChi !== 0 ? numChi / denomChi : Number.NaN;
    const denomEta = DPlus[i] + DMinus[i] + eps;
    eta[i] = denomEta !== 0 ? Math.abs(DPlus[i] - DMinus[i]) / denomEta : Number.NaN;
  }

  const zChi = zscoreSeries(chi);
  const zEta = zscoreSeries(eta);
  const zR = zscoreSeries(R);
  const zDR = zscoreSeries(dR);
  const S = new Array(L).fill(0);
  for (let i = 0; i < L; i += 1) {
    const a = Number.isFinite(zChi[i]) ? zChi[i] : 0;
    const b = Number.isFinite(zEta[i]) ? zEta[i] : 0;
    const c = Number.isFinite(zR[i]) ? zR[i] : 0;
    const d = Number.isFinite(zDR[i]) ? zDR[i] : 0;
    const sVal = cfg.a_chi * a + cfg.b_eta * b + cfg.c_R * c + cfg.d_dR * d;
    if (Number.isFinite(sVal)) {
      S[i] = sVal;
    } else {
      S[i] = 0;
    }
  }

  const wTA = S.map((val) => {
    if (!Number.isFinite(val)) {
      return 0.5;
    }
    const w = 1 / (1 + Math.exp(-cfg.kappa * (val - cfg.theta)));
    return Math.min(cfg.wmax, Math.max(cfg.wmin, w));
  });

  const SMA50 = rollingMean(QQQ, 50);
  const SMA200 = rollingMean(QQQ, 200);
  const slope50 = rollingSlope(SMA50, 20, 10);
  const trendUp = QQQ.map((price, idx) => {
    if (Number.isFinite(price) && Number.isFinite(SMA200[idx]) && price > SMA200[idx]) {
      return 1;
    }
    return 0;
  });
  const rsi14 = rsi(QQQ, 14);
  const momScore = rsi14.map((value) => {
    if (!Number.isFinite(value)) return Number.NaN;
    return Math.min(1, Math.max(0, (value - 30) / 40));
  });
  const rv20 = rollingStd(logReturns(QQQ), 20);
  const volScore = pctRank(rv20);
  const slopeRank = pctRank(slope50);

  const TAScore = new Array(L).fill(0.5);
  for (let i = 0; i < L; i += 1) {
    const components = [];
    const comp1 = Number.isFinite(trendUp[i])
      ? 0.6 * trendUp[i] + 0.4 * (Number.isFinite(slopeRank[i]) ? slopeRank[i] : 0)
      : Number.NaN;
    if (Number.isFinite(comp1)) components.push(comp1);
    if (Number.isFinite(momScore[i])) components.push(momScore[i]);
    const comp3 = Number.isFinite(volScore[i]) ? 1 - volScore[i] : Number.NaN;
    if (Number.isFinite(comp3)) components.push(comp3);
    TAScore[i] = components.length > 0 ? components.reduce((sum, v) => sum + v, 0) / components.length : 0.5;
  }

  const TAState = new Array(L).fill(0);
  let taSt = 0;
  for (let i = 0; i < L; i += 1) {
    const v = TAScore[i];
    if (Number.isFinite(v)) {
      if (v >= 0.6) taSt = 1;
      else if (v <= 0.4) taSt = -1;
    }
    TAState[i] = taSt;
  }
  const posTA = TAState.map((s) => (s > 0 ? 1 : (s === 0 ? 1 / 3 : 0)));

  const zJ = zscoreSeries(J);
  const FFLState = new Array(L).fill(0);
  let ffSt = 0;
  for (let i = 0; i < L; i += 1) {
    const v = zJ[i];
    if (Number.isFinite(v)) {
      if (v >= cfg.on_thr) ffSt = 1;
      else if (v <= cfg.off_thr) ffSt = -1;
    }
    FFLState[i] = ffSt;
  }
  const posFFL = FFLState.map((s) => (s > 0 ? 1 : (s === 0 ? 1 / 3 : 0)));
  const posMix = posTA.map((v, idx) => {
    const w = wTA[idx];
    const flow = posFFL[idx];
    return Math.min(1, Math.max(0, w * v + (1 - w) * flow));
  });

  const zAbs = zscoreSeries(g.map((val) => Math.abs(val)));
  const shock = zAbs.map((val, idx) => (Number.isFinite(val) && val > cfg.shock_z && g[idx] < 0 ? 1 / 3 : 1));

  const rQQQ = arReturns(QQQ);
  const pnlTA = pnlSeries(posTA, rQQQ);
  const pnlFF = pnlSeries(posFFL, rQQQ);
  const TQI = rollingIR(pnlTA, 63);
  const FFQI = rollingIR(pnlFF, 63);
  const FQI = new Array(L).fill(Number.NaN);
  for (let i = 0; i < L; i += 1) {
    const start = Math.max(0, i - 62);
    const js = [];
    const rs = [];
    for (let j = start; j <= i; j += 1) {
      const fj = J[j - 1];
      const r = rQQQ[j];
      if (j - 1 >= 0 && Number.isFinite(fj) && Number.isFinite(r)) {
        js.push(fj);
        rs.push(r);
      }
    }
    if (js.length >= 30) {
      FQI[i] = corrSafe(js, rs);
    }
  }

  const tiltMinus = DMinus.map((val, idx) => {
    const denom = DPlus[idx] + DMinus[idx] + 1e-8;
    return denom !== 0 ? (val - DPlus[idx]) / denom : 0;
  });
  const zTilt = zscoreSeries(tiltMinus);

  const dT = new Array(L).fill(Number.NaN);
  for (let i = 0; i < L; i += 1) {
    const prev = i - 10;
    const cur =
      (Number.isFinite(TQI[i]) ? TQI[i] : 0) - (Number.isFinite(FFQI[i]) ? FFQI[i] : 0);
    const prevVal =
      prev >= 0
        ? (Number.isFinite(TQI[prev]) ? TQI[prev] : 0) - (Number.isFinite(FFQI[prev]) ? FFQI[prev] : 0)
        : 0;
    dT[i] = cur - prevVal;
  }

  const gateEW = new Array(L).fill(1);
  const gateDR = new Array(L).fill(1);
  for (let i = 0; i < L; i += 1) {
    let hitsEW = 0;
    if (Number.isFinite(zTilt[i]) && zTilt[i] >= cfg.ew_tilt) hitsEW += 1;
    if (Number.isFinite(zDR[i]) && zDR[i] >= cfg.ew_dR) hitsEW += 1;
    if (Number.isFinite(zChi[i]) && zChi[i] >= cfg.ew_chi) hitsEW += 1;
    if (Number.isFinite(FQI[i]) && FQI[i] <= cfg.ew_FQI) hitsEW += 1;
    if (Number.isFinite(dT[i]) && dT[i] <= cfg.ew_dT) hitsEW += 1;
    gateEW[i] = hitsEW >= 2 ? cfg.G1 : 1;

    let hitsDR = 0;
    if (Number.isFinite(zTilt[i]) && zTilt[i] >= cfg.dr_tilt) hitsDR += 1;
    if (Number.isFinite(zDR[i]) && zDR[i] >= cfg.dr_dR) hitsDR += 1;
    if (Number.isFinite(zChi[i]) && zChi[i] >= cfg.dr_chi) hitsDR += 1;
    const spread = (Number.isFinite(TQI[i]) ? TQI[i] : 0) - (Number.isFinite(FFQI[i]) ? FFQI[i] : 0);
    if (Number.isFinite(FQI[i]) && FQI[i] <= cfg.dr_FQI && spread <= 0) hitsDR += 1;
    gateDR[i] = hitsDR >= 2 ? cfg.G2 : 1;
  }

  const gateCR = new Array(L).fill(1);
  if (cfg.cr_enabled) {
    let streak = 0;
    for (let i = 0; i < L; i += 1) {
      let hc = 0;
      if (Number.isFinite(zDR[i]) && zDR[i] >= cfg.cr_dR) hc += 1;
      if (
        (Number.isFinite(zTilt[i]) && zTilt[i] >= cfg.cr_tilt)
        || (Number.isFinite(zEta[i]) && zEta[i] >= cfg.cr_eta)
      ) {
        hc += 1;
      }
      if (Number.isFinite(FQI[i]) && FQI[i] <= cfg.cr_FQI) hc += 1;
      if (Number.isFinite(zR[i]) && zR[i] >= cfg.cr_zR) hc += 1;
      if (Number.isFinite(rsi14[i]) && rsi14[i] <= cfg.cr_rsi_max) hc += 1;
      if (hc >= 3) streak = Math.max(streak, cfg.cr_hold_days);
      if (
        Number.isFinite(zChi[i]) && zChi[i] < 0
        && Number.isFinite(zDR[i]) && zDR[i] < 0
        && Number.isFinite(zTilt[i]) && zTilt[i] < 0
      ) {
        streak = Math.min(streak, 1);
      }
      if (streak > 0) {
        gateCR[i] = Math.min(gateCR[i], cfg.G3);
        streak -= 1;
      }
    }
  }

  const lo = cfg.neutral_band[0];
  const hi = cfg.neutral_band[1];
  const classicState = classic.state || [];
  const posGate = new Array(L).fill(0);
  for (let i = 0; i < L; i += 1) {
    let cap = Math.min(shock[i], gateEW[i], gateDR[i], gateCR[i]);
    if (cfg.off_cap < 1 && cfg.use_classic_off_gate && classicState[i] === -1) {
      cap = Math.min(cap, Math.max(0, cfg.off_cap));
    }
    if (cfg.off_cap < 1 && cfg.use_ffl_off_gate && FFLState[i] === -1) {
      cap = Math.min(cap, Math.max(0, cfg.off_cap));
    }
    posGate[i] = Math.min(1, Math.max(0, posMix[i] * cap));
  }

  const state = posGate.map((v) => {
    if (v >= hi) return 1;
    if (v <= lo * 0.5) return -1;
    return 0;
  });
  const executedState = state.map((_, idx) => (idx === 0 ? 0 : state[idx - 1]));
  const score = posGate.map((v) => Math.min(1, Math.max(0, v)));
  const wFlow = wTA.map((v) => 1 - v);

  const lastIndex = L - 1;
  const diag = buildDiagnostics({
    lastIndex,
    gateEW,
    gateDR,
    gateCR,
    shock,
    S,
    zChi,
    zEta,
    zR,
    zDR,
    zTilt,
    FQI,
    TQI,
    FFQI,
    dT,
    zJ,
    chi,
    cfg,
    J,
    rsi14,
    TAState,
    FFLState,
    wTA,
    analysisDates,
  });

  const toExport = (arr) => (arr || []).map((v) => (Number.isFinite(v) ? Number(v) : null));
  const toInt = (arr) => (arr || []).map((v) => (Number.isFinite(v) ? Math.trunc(v) : 0));

  return {
    preset,
    window,
    dates: analysisDates.slice(),
    state: toInt(state),
    executedState: toInt(executedState),
    score: toExport(score),
    wClassic: toExport(wTA),
    wFFL: toExport(wFlow),
    scCorr: toExport(classic.scCorr),
    safeNeg: toExport(classic.safeNeg),
    mm: toExport(R),
    diag,
  };
}

function buildDiagnostics({
  lastIndex,
  gateEW,
  gateDR,
  gateCR,
  shock,
  S,
  zChi,
  zEta,
  zR,
  zDR,
  zTilt,
  FQI,
  TQI,
  FFQI,
  dT,
  zJ,
  chi,
  cfg,
  J,
  rsi14,
  TAState,
  FFLState,
  wTA,
  analysisDates,
}) {
  if (lastIndex < 0) return {};
  const fval = (arr, idx = lastIndex) => {
    if (!Array.isArray(arr) || idx < 0 || idx >= arr.length) return null;
    const v = arr[idx];
    return Number.isFinite(v) ? Number(v) : null;
  };
  const zS = zscoreSeries(S);
  const qdiff = new Array(S.length).fill(Number.NaN);
  for (let i = 0; i < S.length; i += 1) {
    const tq = Number.isFinite(TQI[i]) ? TQI[i] : 0;
    const fq = Number.isFinite(FFQI[i]) ? FFQI[i] : 0;
    qdiff[i] = tq - fq;
  }
  const zQ = zscoreSeries(qdiff);
  const zFQI = zscoreSeries(FQI);
  let tfi = null;
  const zSLast = fval(zS);
  const zQLast = fval(zQ);
  const zFQILast = fval(zFQI);
  if (zSLast != null && zQLast != null && zFQILast != null) {
    tfi = zSLast + 0.5 * zQLast - 0.5 * zFQILast;
  }
  const ffi = tfi != null ? -tfi : null;
  const fdd = Number.isFinite(chi[lastIndex]) && chi[lastIndex] !== 0 ? 1 / chi[lastIndex] : null;
  const capEW = Number.isFinite(gateEW[lastIndex]) ? gateEW[lastIndex] : 1;
  const capDR = Number.isFinite(gateDR[lastIndex]) ? gateDR[lastIndex] : 1;
  const capCR = Number.isFinite(gateCR[lastIndex]) ? gateCR[lastIndex] : 1;
  const capShock = Number.isFinite(shock[lastIndex]) ? shock[lastIndex] : 1;
  const cap = Math.min(capEW, capDR, capCR, capShock);
  let capLabel = 'None';
  if (cap < 1) {
    if (cap === capShock) capLabel = 'ShockCap';
    else if (cap === capDR) capLabel = `DR(G2=${cfg.G2.toFixed(2)})`;
    else if (cap === capCR) capLabel = `CR(G3=${cfg.G3.toFixed(2)})`;
    else capLabel = `EW(G1=${cfg.G1.toFixed(2)})`;
  }

  const shockActive = capShock < 1 && capShock !== 1;
  const ewHits = [zTilt, zDR, zChi, FQI, dT].reduce((sum, arr, idx) => {
    const val = fval(arr);
    const thr = [cfg.ew_tilt, cfg.ew_dR, cfg.ew_chi, cfg.ew_FQI, cfg.ew_dT][idx];
    const op = idx >= 3 ? '<=' : '>=';
    if (val == null) return sum;
    if ((op === '>=' && val >= thr) || (op === '<=' && val <= thr)) {
      return sum + 1;
    }
    return sum;
  }, 0);
  const drHits = [zTilt, zDR, zChi, FQI].reduce((sum, arr, idx) => {
    const val = fval(arr);
    const thr = [cfg.dr_tilt, cfg.dr_dR, cfg.dr_chi, cfg.dr_FQI][idx];
    const op = idx === 3 ? '<=' : '>=';
    if (val == null) return sum;
    if ((op === '>=' && val >= thr) || (op === '<=' && val <= thr)) {
      return sum + 1;
    }
    return sum;
  }, 0);

  let regimeLabel = '혼합';
  const wt = fval(wTA);
  if (tfi != null && wt != null) {
    if (tfi >= 0.5 && wt >= 0.6) regimeLabel = 'TA-우위';
    else if (-tfi >= 0.5 && wt <= 0.4) regimeLabel = 'Flow-우위';
    else if (-tfi >= 0.2 && wt > 0.4 && wt < 0.6) regimeLabel = 'Flow-우위혼합';
  }
  const advice = [
    '오늘 운용 의견 (TQQQ 기준)',
    `- 현재 판단: ${regimeLabel}`,
    '- 운용 프레임: pos = wTA·pos_TA + (1−wTA)·pos_Flow; 급락 쇼크 시 pos≤1/3 캡',
    '- 엔트리/추세는 TA 기준 우선(SMA200 상회·RSI>50 ⇒ 노출 하한 1/3 유지)',
    '- Flow(zJ)·FDD 회복 시 비중 상향; 변동성 확산(η↑/ΔR↑) 시 전일 고점 리테스트 후 분할 증액',
    `- 게이트: EW ${ewHits}/5 (cap=${cfg.G1.toFixed(2)}), DR ${drHits}/4 (cap=${cfg.G2.toFixed(2)}), CR ${cfg.cr_enabled ? 'on' : 'off'} (cap=${cfg.G3.toFixed(2)}), Shock=${shockActive ? 'on' : 'off'}`,
  ].join('\n');

  return {
    date: analysisDates[lastIndex] || null,
    wTA: valueOrNull(wt),
    S: valueOrNull(S[lastIndex]),
    z_chi: valueOrNull(zChi[lastIndex]),
    z_eta: valueOrNull(zEta[lastIndex]),
    z_R: valueOrNull(zR[lastIndex]),
    z_dR: valueOrNull(zDR[lastIndex]),
    tilt_z: valueOrNull(zTilt[lastIndex]),
    FQI: valueOrNull(FQI[lastIndex]),
    TQI: valueOrNull(TQI[lastIndex]),
    FFQI: valueOrNull(FFQI[lastIndex]),
    dT: valueOrNull(dT[lastIndex]),
    zJ: valueOrNull(zJ[lastIndex]),
    FDD: valueOrNull(fdd),
    TFI: valueOrNull(tfi),
    FFI: valueOrNull(ffi),
    gate_cap: valueOrNull(cap),
    gate_label: capLabel,
    ew_cap: valueOrNull(capEW),
    dr_cap: valueOrNull(capDR),
    shock_cap: valueOrNull(capShock),
    cr_cap: valueOrNull(capCR),
    advice,
  };
}

module.exports = {
  computeFusionNewgate,
};
