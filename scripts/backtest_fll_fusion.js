#!/usr/bin/env node
/*
 * Backtest: FLL-Fusion (Classic ⊕ FFL+STAB) on QQQ
 * Range: 2020-01-01 to 2025-10-30 (inclusive if data available)
 * Data: static_site/data/historical_prices.json (offline snapshot)
 * Logic:
 *   - Build windowed correlation/stability metrics (WINDOW=30)
 *   - Compute Classic risk series (corr+safe-neg thresholds)
 *   - Compute FFL series with STAB variant (Flux + Stability guards)
 *   - Fuse signals using rolling 40-day Hit-rate and IC via softmax weights
 *   - Apply Absorption Guard (mmFragile≈0.88 blocks On, mmOff≈0.96 forces Off)
 *   - Execute with 1-day lag; positions: On=TQQQ(3x), Neutral=QQQ, Off=Cash
 */

const fs = require('fs');
const path = require('path');
const MM = require(path.join(__dirname, '..', 'static_site', 'assets', 'metrics.js'));

// ------------------ Config ------------------
const DATA_PATH = path.join(__dirname, '..', 'static_site', 'data', 'historical_prices.json');
const WINDOW = 30;
const START = '2020-01-01';
const END = '2025-10-30';

const ASSETS = [
  { symbol: 'QQQ', category: 'stock' },
  { symbol: 'IWM', category: 'stock' },
  { symbol: 'SPY', category: 'stock' },
  { symbol: 'TLT', category: 'bond' },
  { symbol: 'GLD', category: 'gold' },
  { symbol: 'BTC-USD', category: 'crypto' },
];
const SIGNAL = {
  symbols: ['IWM', 'SPY', 'TLT', 'GLD', 'BTC-USD'],
  primaryStock: 'IWM',
  breadth: ['IWM', 'SPY', 'BTC-USD'],
  pairKey: 'IWM|BTC-USD',
  trade: { baseSymbol: 'QQQ', leveredSymbol: 'TQQQ', leverage: 3 },
};

const RISK_CFG_CLASSIC = {
  weights: { sc: 0.70, safe: 0.30 },
  thresholds: { scoreOn: 0.65, scoreOff: 0.30, corrOn: 0.50, corrMinOn: 0.20, corrOff: -0.10 },
  pairKey: SIGNAL.pairKey,
};

const RISK_CFG_FFL = {
  lookbacks: { momentum: 10, vol: 20, breadth: 5 },
  p: 1.5,
  zSat: 2.0,
  lambda: 0.25,
  thresholds: {
    jOn: +0.10, jOff: -0.08, scoreOn: 0.60, scoreOff: 0.40, breadthOn: 0.50, mmFragile: 0.88, mmOff: 0.96,
    pconOn: 0.55, pconOff: 0.40, mmHi: 0.90, downAll: 0.60, corrConeDays: 5, driftMinDays: 3, driftCool: 2, offMinDays: 3, vOn: +0.05, vOff: -0.05,
  },
  kOn: 0.60,
  kLambda: 1.0,
  variant: 'stab',
};

// ------------------ IO helpers ------------------
function loadHistorical() {
  const raw = JSON.parse(fs.readFileSync(DATA_PATH, 'utf8'));
  return raw;
}
function dateNum(s) { return Number(String(s).replace(/-/g, '')); }

function alignFromHistorical(raw, symbols) {
  const list = raw.assets.filter((a) => symbols.includes(a.symbol)).map((a) => ({ symbol: a.symbol, category: a.category, dates: a.dates, prices: a.prices }));
  return MM.alignSeries(list);
}
function sliceAligned(aligned, start, end) {
  const s = dateNum(start); const e = dateNum(end);
  const keep = aligned.dates.map((d, i) => ({ d, i, n: dateNum(d) })).filter((x) => x.n >= s && x.n <= e);
  const dates = keep.map((x) => x.d);
  const idxMap = new Map(keep.map((x, k) => [x.d, aligned.dates.indexOf(x.d)]));
  const prices = {}; Object.entries(aligned.prices).forEach(([sym, arr]) => { prices[sym] = dates.map((d) => arr[idxMap.get(d)]); });
  return { dates, prices, categories: aligned.categories };
}

// ------------------ Metrics (records & pairs) ------------------
function computeWindowMetrics(window, returns, aligned) {
  const allSymbols = ASSETS.map((a) => a.symbol);
  const signalSymbols = SIGNAL.symbols;
  const categories = aligned.categories;
  const dates = returns.dates;
  const records = []; const stabilityValues = [];
  for (let endIndex = window - 1; endIndex < returns.dates.length; endIndex += 1) {
    const startIndex = endIndex - window + 1;
    const fullMatrix = MM.buildCorrelationMatrix(allSymbols, returns.returns, startIndex, endIndex);
    const signalMatrix = MM.buildCorrelationMatrix(signalSymbols, returns.returns, startIndex, endIndex);
    const stability = MM.computeStability(signalMatrix, signalSymbols, categories);
    const sub = MM.computeSubIndices(signalMatrix, signalSymbols, categories);
    const smoothed = stability;
    records.push({ date: dates[endIndex], stability, sub, matrix: signalMatrix, fullMatrix, smoothed, delta: 0 });
    stabilityValues.push(stability);
  }
  const smoothedSeries = MM.ema(stabilityValues, 10);
  const shortEma = MM.ema(stabilityValues, 3);
  const longEma = MM.ema(stabilityValues, 10);
  records.forEach((r, i) => { r.smoothed = smoothedSeries[i]; r.delta = shortEma[i] - longEma[i]; });
  const pairs = buildPairSeries(records, window, signalSymbols);
  return { records, pairs };
}
function buildPairSeries(records, window, symbols) {
  const pairs = {}; const priceOffset = window - 1;
  for (let i = 0; i < symbols.length; i += 1) {
    for (let j = i + 1; j < symbols.length; j += 1) {
      const key = `${symbols[i]}|${symbols[j]}`; pairs[key] = { dates: [], correlation: [] };
    }
  }
  records.forEach((record) => {
    const matrix = record.matrix; if (!Array.isArray(matrix)) return;
    for (let i = 0; i < symbols.length; i += 1) {
      for (let j = i + 1; j < symbols.length; j += 1) {
        const key = `${symbols[i]}|${symbols[j]}`; const pair = pairs[key];
        const corrValue = matrix ? matrix[i][j] : null; pair.correlation.push(Number.isFinite(corrValue) ? corrValue : null);
        if (pair.dates.length < records.length) pair.dates.push(record.date);
      }
    }
  });
  return pairs;
}

// ------------------ Classic ------------------
function computeRiskSeriesClassic(metrics, recordsOverride) {
  if (!metrics || !Array.isArray(metrics.records)) return null;
  const pair = (metrics.pairs && metrics.pairs[RISK_CFG_CLASSIC.pairKey]) || null;
  if (!pair || !Array.isArray(pair.correlation)) return null;

  const allDates = metrics.records.map((r) => r.date);
  let baseIdx = 0; let length = allDates.length;
  if (Array.isArray(recordsOverride) && recordsOverride.length > 0) {
    const firstDate = recordsOverride[0]?.date;
    const gIdx = metrics.records.findIndex((r) => r.date === firstDate);
    baseIdx = gIdx >= 0 ? gIdx : 0;
    length = recordsOverride.length;
  }
  const dates = allDates.slice(baseIdx, baseIdx + length);
  const scCorr = pair.correlation.slice(baseIdx, baseIdx + length);
  const safeNeg = metrics.records.slice(baseIdx, baseIdx + length).map((r) => Number(r.sub?.safeNegative) || 0);

  const w = RISK_CFG_CLASSIC.weights; const th = RISK_CFG_CLASSIC.thresholds;
  const score = scCorr.map((c, i) => Math.max(0, Math.min(1, w.sc * Math.max(0, Number(c)) + w.safe * safeNeg[i])));
  const state = scCorr.map((c, i) => {
    const s = score[i];
    if (c >= th.corrOn) return 1;
    if (c <= th.corrOff || s <= th.scoreOff) return -1;
    if (s >= th.scoreOn && c >= th.corrMinOn) return 1;
    return 0;
  });
  return { dates, score, state, scCorr, safeNeg };
}

// ------------------ FFL (+STAB variant) ------------------
function rollingReturnFromSeries(series, index, lookback) {
  if (!Array.isArray(series) || !Number.isInteger(index) || !Number.isInteger(lookback) || lookback <= 0) return null;
  if (index >= series.length || index - lookback < 0) return null;
  const end = series[index]; const start = series[index - lookback];
  if (!Number.isFinite(end) || !Number.isFinite(start) || start === 0) return null;
  return end / start - 1;
}
function rollingStdFromSeries(series, index, lookback) {
  if (!Array.isArray(series) || !Number.isInteger(index) || !Number.isInteger(lookback) || lookback < 2) return null;
  if (index >= series.length || index - lookback < 0) return null;
  const start = Math.max(0, index - lookback);
  const windowReturns = [];
  for (let i = start + 1; i <= index; i += 1) {
    const prev = series[i - 1]; const cur = series[i];
    if (!Number.isFinite(prev) || !Number.isFinite(cur) || prev === 0) continue;
    windowReturns.push(cur / prev - 1);
  }
  if (windowReturns.length < 2) return null;
  const meanReturn = windowReturns.reduce((a, b) => a + b, 0) / windowReturns.length;
  const variance = windowReturns.reduce((a, b) => a + (b - meanReturn) * (b - meanReturn), 0) / (windowReturns.length - 1);
  return Math.sqrt(Math.max(variance, 0));
}
function zMomentumFromSeries(series, index, k, v, zSat = 2.0) {
  if (!Array.isArray(series) || !Number.isInteger(index) || !Number.isInteger(k) || !Number.isInteger(v)) return null;
  if (k <= 0 || v <= 1 || index >= series.length || index - k < 0) return null;
  const base = series[index - k]; const latest = series[index];
  if (!Number.isFinite(base) || !Number.isFinite(latest) || base === 0) return null;
  const momentum = latest / base - 1; const std = rollingStdFromSeries(series, index, v);
  if (!Number.isFinite(momentum) || !Number.isFinite(std) || std === 0) return null;
  return Math.tanh((momentum / std) / zSat);
}
function clamp01(v) { if (!Number.isFinite(v)) return 0; if (v <= 0) return 0; if (v >= 1) return 1; return v; }
function normalizeRangeSafe(value, min, max) { if (!Number.isFinite(value) || max <= min) return 0; if (value <= min) return 0; if (value >= max) return 1; return (value - min) / (max - min); }
function frobeniusDiff(matrixA, matrixB) { if (!Array.isArray(matrixA) || !Array.isArray(matrixB) || matrixA.length !== matrixB.length) return null; let s = 0; for (let i = 0; i < matrixA.length; i += 1) { for (let j = 0; j < matrixA[i].length; j += 1) { const d = (matrixA[i][j] - matrixB[i][j]); s += d * d; } } return Math.sqrt(s); }
function rollingZScore(arr, idx, win) { const xs = []; for (let k = Math.max(0, idx - win + 1); k <= idx; k += 1) { const v = arr[k]; if (Number.isFinite(v)) xs.push(v); } if (xs.length < 5) return null; const mean = xs.reduce((a, b) => a + b, 0) / xs.length; const varr = xs.reduce((a, b) => a + (b - mean) * (b - mean), 0) / xs.length; const sd = Math.sqrt(Math.max(varr, 1e-12)); return sd > 0 ? (arr[idx] - mean) / sd : null; }
function sigmoid(x, slope = 1) { if (!Number.isFinite(x)) return null; const t = Math.max(Math.min(x * slope, 60), -60); return 1 / (1 + Math.exp(-t)); }

function computeRiskSeriesFFL(metrics, recordsOverride, prices) {
  // This is a trimmed version to get mm, guard, flux, and final state, adapted from scripts/backtest_ffl.js
  if (!metrics || !Array.isArray(metrics.records) || metrics.records.length === 0) return null;
  const labels = SIGNAL.symbols; const indexLookup = {}; labels.forEach((s, i) => { indexLookup[s] = i; });
  const allRecords = metrics.records; const alignedRecords = Array.isArray(recordsOverride) && recordsOverride.length > 0 ? recordsOverride : null;
  const firstDate = alignedRecords?.[0]?.date ?? allRecords[0]?.date; let baseIdx = allRecords.findIndex((r) => r.date === firstDate); if (baseIdx < 0) baseIdx = 0;
  const length = alignedRecords ? alignedRecords.length : allRecords.length - baseIdx; if (length <= 0) return null;
  const slice = allRecords.slice(baseIdx, baseIdx + length); const dates = slice.map((r) => r.date);
  const pairSeries = metrics.pairs?.[SIGNAL.pairKey] || null; const scCorr = pairSeries?.correlation?.slice(baseIdx, baseIdx + length) || new Array(length).fill(null);
  const safeNeg = slice.map((r) => Number(r.sub?.safeNegative) || 0);

  const windowOffset = Math.max(1, WINDOW - 1);
  const mm = new Array(length).fill(null); const guard = new Array(length).fill(null); const score = new Array(length).fill(null);
  const jFlux = new Array(length).fill(null); const fullFlux = new Array(length).fill(null); const fullFluxZ = new Array(length).fill(null); const mmTrend = new Array(length).fill(null); const fluxSlope = new Array(length).fill(null);
  const diffusionScore = new Array(length).fill(null); const comboMomentum = new Array(length).fill(null); const breadth = new Array(length).fill(null); const vPC1 = new Array(length).fill(null); const kappa = new Array(length).fill(null);
  const apdf = new Array(length).fill(null); const pcon = new Array(length).fill(null); const fluxRaw = new Array(length).fill(null); const fluxIntensity = new Array(length).fill(null);
  const fragile = new Array(length).fill(false); const far = new Array(length).fill(null); let prevMatrix = null;

  for (let i = 0; i < length; i += 1) {
    const record = allRecords[baseIdx + i]; const matrix = record?.matrix;
    if (Array.isArray(matrix) && matrix.length > 0) { const lambda1 = MM.topEigenvalue(matrix) || 0; mm[i] = lambda1 / (matrix.length || 1); }
    if (i > 0 && Number.isFinite(mm[i]) && Number.isFinite(mm[i - 1])) mmTrend[i] = mm[i] - mm[i - 1]; else mmTrend[i] = null;
    const priceIndex = baseIdx + i + windowOffset; const zr = {}; labels.forEach((symbol) => { zr[symbol] = zMomentumFromSeries(prices[symbol], priceIndex, RISK_CFG_FFL.lookbacks.momentum, RISK_CFG_FFL.lookbacks.vol, RISK_CFG_FFL.zSat); });
    let weightSum = 0; let fluxSum = 0; let absSum = 0; let apdfWeighted = 0; let pconWeighted = 0; let weightsAll = 0;
    const CLUSTERS = { risk: ['IWM', 'SPY', 'BTC-USD'], safe: ['TLT', 'GLD'] };
    CLUSTERS.safe.forEach((safeSymbol) => {
      const safeIdx = indexLookup[safeSymbol]; if (!Number.isInteger(safeIdx)) return;
      CLUSTERS.risk.forEach((riskSymbol) => {
        const riskIdx = indexLookup[riskSymbol]; if (!Number.isInteger(riskIdx)) return;
        const coef = Array.isArray(matrix) ? (Number(matrix?.[safeIdx]?.[riskIdx]) || 0) : 0;
        const weight = Math.pow(Math.abs(coef), RISK_CFG_FFL.p); if (!Number.isFinite(weight) || weight <= 0) return;
        const sZ = Number.isFinite(zr[safeSymbol]) ? zr[safeSymbol] : 0; const rZ = Number.isFinite(zr[riskSymbol]) ? zr[riskSymbol] : 0;
        const diff = rZ - sZ; weightSum += weight; fluxSum += weight * diff; absSum += weight * Math.abs(diff);
        apdfWeighted += weight * (rZ - sZ); pconWeighted += weight * (rZ > sZ ? 1 : 0); weightsAll += weight;
      });
    });
    const Jbar = weightSum > 0 ? fluxSum / weightSum : 0; const Jnorm = Math.tanh(Jbar / RISK_CFG_FFL.lambda);
    jFlux[i] = Number.isFinite(Jnorm) ? Jnorm : null; fluxRaw[i] = weightSum > 0 ? Jbar : null; fluxIntensity[i] = weightSum > 0 ? absSum / weightSum : null;
    if (i > 0 && Number.isFinite(jFlux[i]) && Number.isFinite(jFlux[i - 1])) fluxSlope[i] = jFlux[i] - jFlux[i - 1]; else fluxSlope[i] = null;
    const fraw = frobeniusDiff(matrix, prevMatrix); prevMatrix = matrix; fullFlux[i] = Number.isFinite(fraw) ? fraw : null; const z = rollingZScore(fullFlux, i, Math.min(63, Math.max(15, Math.floor(length / 4)))); fullFluxZ[i] = Number.isFinite(z) ? z : null;
    const apdfRaw = weightsAll > 0 ? apdfWeighted / weightsAll : 0; apdf[i] = Number.isFinite(apdfRaw) ? Math.tanh(apdfRaw / (RISK_CFG_FFL.lambda || 0.25)) : null; pcon[i] = weightsAll > 0 ? Math.max(0, Math.min(1, pconWeighted / weightsAll)) : null;
    const riskZValues = CLUSTERS.risk.map((s) => zr[s]).filter((v) => Number.isFinite(v)); comboMomentum[i] = riskZValues.length > 0 ? riskZValues.reduce((a, v) => a + v, 0) / riskZValues.length : null; breadth[i] = riskZValues.length > 0 ? riskZValues.filter((v) => v > 0).length / riskZValues.length : null;
    const safePen = normalizeRangeSafe(Number(record?.sub?.safeNegative) || 0, 0.35, 0.60); const mmPen = normalizeRangeSafe(mm[i], 0.85, 0.97); const deltaPen = normalizeRangeSafe(Math.max(0, -(record?.delta ?? 0)), 0.015, 0.05);
    const fluxGuard = Number.isFinite(fullFluxZ[i]) ? sigmoid(fullFluxZ[i], 0.85) : 1; const guardVal = 0.4 * mmPen + 0.2 * safePen + 0.2 * deltaPen + 0.2 * fluxGuard; guard[i] = Number.isFinite(guardVal) ? Number(guardVal.toFixed(6)) : null;
    // basic scoring for thresholds
    const cPos = Math.max(0, Number(scCorr[i])); const sPos = Math.max(0, Number(safeNeg[i])); const baseScore = clamp01(0.6 * cPos + 0.4 * sPos);
    score[i] = baseScore;
    // fragility (for display)
    const th = RISK_CFG_FFL.thresholds; fragile[i] = guardVal >= th.mmFragile;
    // FAR
    far[i] = Number.isFinite(jFlux[i]) && Number.isFinite(mm[i]) && mm[i] > 0 ? Math.abs(jFlux[i]) / (mm[i] + 1e-9) : null;
  }
  // Decide state with simple FFL+STAB rules on top of baseScore/flux
  const th = RISK_CFG_FFL.thresholds; const stateArr = new Array(length).fill(0);
  for (let i = 0; i < length; i += 1) {
    const fluxVal = Number(jFlux[i]); const sc = Number(scCorr[i]); const sneg = Number(safeNeg[i]); const guardVal = Number(guard[i]); const mmVal = Number(mm[i]);
    const on = (sc >= 0.20 && (0.6 * Math.max(0, sc) + 0.4 * Math.max(0, sneg)) >= 0.55 && fluxVal >= +0.03 && (breadth[i] ?? 0) >= 0.45 && guardVal < 0.90);
    const off = (fluxVal <= -0.03) || (mmVal >= th.mmOff) || (guardVal >= 1.0) || ((breadth[i] ?? 0) <= 0.20);
    stateArr[i] = on ? 1 : (off ? -1 : 0);
  }
  const executedState = stateArr.map((v, idx) => (idx === 0 ? 0 : stateArr[idx - 1] || 0));
  return { dates, score, state: stateArr, executedState, guard, mm, scCorr, safeNeg };
}

// ------------------ Fusion ------------------
function toReturnsSimple(values) {
  const out = []; for (let i = 1; i < values.length; i += 1) { const prev = values[i - 1]; const cur = values[i]; out.push(Number.isFinite(prev) && Number.isFinite(cur) && prev !== 0 ? Math.log(cur / prev) : 0); } return out;
}
function rollingHit(pred, r, W) { let hits = 0, tot = 0; const out = new Array(r.length).fill(0.5); const buf = []; for (let t = 0; t < r.length; t += 1) { const p = pred[t]; const y = Math.sign(r[t]); if (p !== 0) { const ok = (Math.sign(p) === y) ? 1 : 0; buf.push([ok, 1]); hits += ok; tot += 1; } else { buf.push([0, 0]); } if (buf.length > W) { const [okRem, totRem] = buf.shift(); hits -= okRem; tot -= totRem; } out[t] = tot > 0 ? hits / tot : 0.5; } return out; }
function rollingIC(pred, r, W) { const out = new Array(r.length).fill(0); const bx = [], by = []; let sumXY = 0, sumR = 0, sumR2 = 0; for (let t = 0; t < r.length; t += 1) { const x = pred[t]; const y = r[t]; bx.push(x); by.push(y); sumXY += (x * y); sumR += y; sumR2 += y * y; if (bx.length > W) { const x0 = bx.shift(); const y0 = by.shift(); sumXY -= (x0 * y0); sumR -= y0; sumR2 -= y0 * y0; } const n = by.length; const mu = n ? (sumR / n) : 0; const varR = Math.max(1e-12, (sumR2 / n) - mu * mu); const sd = Math.sqrt(varR); out[t] = sd > 0 ? (sumXY / n) / sd : 0; } return out; }
function softmax2(a, b, tau, floor) { const ezA = Math.exp(tau * a), ezB = Math.exp(tau * b); let wA = ezA / (ezA + ezB), wB = 1 - wA; wA = Math.max(floor, wA); wB = Math.max(floor, wB); const s = wA + wB; return [wA / s, wB / s]; }

function computeRiskSeriesFLLFusion(metrics, records, prices, opts = {}) {
  const cfg = Object.assign({ win: 40, lam: 0.50, tau: 4.0, floor: 0.10, onThr: +0.20, offThr: -0.20 }, opts || {});
  const classic = computeRiskSeriesClassic(metrics, records);
  const fflStab = computeRiskSeriesFFL(metrics, records, prices);
  if (!classic || !fflStab) return null;
  const dates = classic.dates; const n = dates.length;
  const base = prices[SIGNAL.trade.baseSymbol]; const rQQQ = toReturnsSimple(base).slice(WINDOW - 1); // align to dates length
  // predictions with T+1 lag
  const predC_now = classic.state.map((s) => (s > 0 ? 1 : (s < 0 ? -1 : 0)));
  const predF_now = fflStab.state.map((s) => (s > 0 ? 1 : (s < 0 ? -1 : 0)));
  const predC = [0, ...predC_now.slice(0, n - 1)];
  const predF = [0, ...predF_now.slice(0, n - 1)];
  // weights
  const hitC = rollingHit(predC, rQQQ, cfg.win); const hitF = rollingHit(predF, rQQQ, cfg.win);
  const icC = rollingIC(predC, rQQQ, cfg.win); const icF = rollingIC(predF, rQQQ, cfg.win);
  const scoreC = hitC.map((h, i) => cfg.lam * (h - 0.5) + (1 - cfg.lam) * icC[i]);
  const scoreF = hitF.map((h, i) => cfg.lam * (h - 0.5) + (1 - cfg.lam) * icF[i]);
  const wClassic = new Array(n).fill(0.5); const wFFL = new Array(n).fill(0.5);
  for (let i = 0; i < n; i += 1) { const [a, b] = softmax2(scoreC[i], scoreF[i], cfg.tau, cfg.floor); wClassic[i] = a; wFFL[i] = b; }
  // guard
  const mmFragile = RISK_CFG_FFL.thresholds.mmFragile; const mmOff = RISK_CFG_FFL.thresholds.mmOff; const mm = fflStab.mm || classic.mm || new Array(n).fill(0);
  const guardFragile = mm.map((x) => Number.isFinite(x) && x >= mmFragile); const guardHardOff = mm.map((x) => Number.isFinite(x) && x >= mmOff);
  // fuse
  const fusedRaw = new Array(n).fill(0); for (let i = 0; i < n; i += 1) fusedRaw[i] = wClassic[i] * predC_now[i] + wFFL[i] * predF_now[i];
  const state = fusedRaw.map((v, i) => { if (guardHardOff[i]) return -1; if (guardFragile[i] && v > 0) return 0; if (v >= cfg.onThr) return 1; if (v <= cfg.offThr) return -1; return 0; });
  const executedState = state.map((v, idx) => (idx === 0 ? 0 : state[idx - 1] || 0));
  const score = fusedRaw.map((v) => Math.max(0, Math.min(1, (v + 1) / 2)));
  return { dates, state, executedState, score };
}

// ------------------ Backtest runner ------------------
function leveragedReturn(baseReturn, leverage = 3) { if (!Number.isFinite(baseReturn)) return 0; const levered = leverage * baseReturn; return Math.max(-0.99, levered); }
function equityFromReturns(retArr) { let e = 1; const eq = []; for (let i = 0; i < retArr.length; i += 1) { const r = Number.isFinite(retArr[i]) ? retArr[i] : 0; e *= (1 + r); eq.push(Number(e.toFixed(8))); } return eq; }
function maxDrawdown(equity) { let peak = equity[0] || 1; let mdd = 0; for (let i = 0; i < equity.length; i += 1) { const v = equity[i]; if (v > peak) peak = v; const dd = (peak - v) / peak; if (dd > mdd) mdd = dd; } return mdd; }
function annualize(days, totalReturn) { const years = Math.max(days / 252, 1e-9); return Math.pow(1 + totalReturn, 1 / years) - 1; }

async function main() {
  const raw = loadHistorical();
  const symbols = ASSETS.map((a) => a.symbol);
  const alignedFull = alignFromHistorical(raw, symbols);
  const aligned = sliceAligned(alignedFull, START, END);
  const returns = MM.computeReturns(aligned);
  const metrics = computeWindowMetrics(WINDOW, returns, aligned);
  const prices = returns.priceSeries; // aligned to returns.dates (~ WINDOW offset for metrics)

  const fusion = computeRiskSeriesFLLFusion(metrics, metrics.records, prices, { win: 40, lam: 0.50, tau: 4.0, floor: 0.10, onThr: 0.20, offThr: -0.20 });
  if (!fusion) throw new Error('Failed to compute FLL-Fusion series');

  const dates = fusion.dates; const baseSym = SIGNAL.trade.baseSymbol; const px = prices[baseSym];
  const baseRet = []; const priceOffset = WINDOW - 1;
  for (let i = 0; i < dates.length; i += 1) {
    const idx = priceOffset + i; const prev = px[idx - 1]; const cur = px[idx];
    baseRet.push(Number.isFinite(prev) && Number.isFinite(cur) && prev !== 0 ? (cur / prev - 1) : 0);
  }
  const executed = fusion.executedState; const lev = SIGNAL.trade.leverage;
  const stratRet = executed.map((reg, i) => (reg > 0 ? leveragedReturn(baseRet[i], lev) : (reg < 0 ? 0 : baseRet[i])));
  const eqStrat = equityFromReturns(stratRet); const eqBH = equityFromReturns(baseRet);
  const totalDays = dates.length; const totalStrat = eqStrat[eqStrat.length - 1] - 1; const totalBH = eqBH[eqBH.length - 1] - 1;
  const cagrStrat = annualize(totalDays, totalStrat); const cagrBH = annualize(totalDays, totalBH);
  const mddStrat = maxDrawdown(eqStrat); const mddBH = maxDrawdown(eqBH);

  const summary = {
    window: WINDOW,
    range: { start: dates[0], end: dates[dates.length - 1] },
    samples: totalDays,
    strategy: { total: totalStrat, cagr: cagrStrat, mdd: mddStrat },
    benchmark: { total: totalBH, cagr: cagrBH, mdd: mddBH },
  };
  console.log(JSON.stringify(summary, null, 2));
}

if (require.main === module) { main().catch((err) => { console.error(err); process.exit(1); }); }

