#!/usr/bin/env node
/*
 * Offline backtest runner for 2020–2025 using local historical data.
 * - Reads static_site/data/historical_prices.json
 * - Rebuilds rolling correlation records (window=30 by default)
 * - Applies the FFL (Classic+Flux) gating identical to frontend logic, incl. DiffusionScore
 * - Computes equity vs. benchmark (QQQ) with 1-day execution lag
 */
const fs = require('fs');
const path = require('path');
const MM = require('../static_site/assets/metrics');

const DATA_PATH = path.join(__dirname, '..', 'static_site', 'data', 'historical_prices.json');
const WINDOW = Number(process.env.BACKTEST_WINDOW || 30);
const RANGE_START = process.env.BACKTEST_START || '2020-01-01';
const RANGE_END = process.env.BACKTEST_END || '2025-12-31';

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
const CLUSTERS = { risk: ['IWM', 'SPY', 'BTC-USD'], safe: ['TLT', 'GLD'] };
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
  exp: { lam: 1.0, rOn: 1.20, rOff: -1.10, breadth1dMin: 0.55, d0AnyPos: true, d0BothPosHiCorr: true },
  variant: (process.env.MODE === 'ffl_exp' || process.env.VARIANT === 'exp') ? 'exp' : 'classic',
};

// Allow ENV overrides for quick sweeps (e.g., MODE=ffl_exp RON=0.95 ROFF=-0.80 ELAM=0.9 BMIN=0.45 D0ANY=1 D0BOTH=0)
if (RISK_CFG_FFL.variant === 'exp') {
  const num = (v) => (v == null ? undefined : Number(v));
  const bool = (v) => (v == null ? undefined : (String(v) === '1' || String(v).toLowerCase() === 'true'));
  const rOn = num(process.env.RON);
  const rOff = num(process.env.ROFF);
  const lam = num(process.env.ELAM);
  const bmin = num(process.env.BMIN);
  const d0any = bool(process.env.D0ANY);
  const d0both = bool(process.env.D0BOTH);
  if (!RISK_CFG_FFL.exp) RISK_CFG_FFL.exp = {};
  if (Number.isFinite(rOn)) RISK_CFG_FFL.exp.rOn = rOn;
  if (Number.isFinite(rOff)) RISK_CFG_FFL.exp.rOff = rOff;
  if (Number.isFinite(lam)) RISK_CFG_FFL.exp.lam = lam;
  if (Number.isFinite(bmin)) RISK_CFG_FFL.exp.breadth1dMin = bmin;
  if (typeof d0any === 'boolean') RISK_CFG_FFL.exp.d0AnyPos = d0any;
  if (typeof d0both === 'boolean') RISK_CFG_FFL.exp.d0BothPosHiCorr = d0both;
}

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
function rollingMeanVariance(arr, index, lookback) { if (!Array.isArray(arr) || !Number.isInteger(index) || index < 0) return null; if (!Number.isInteger(lookback) || lookback <= 1) return null; const start = Math.max(0, index - lookback + 1); const window = []; for (let i = start; i <= index; i += 1) { const v = Number(arr[i]); if (Number.isFinite(v)) window.push(v); } if (window.length < 2) return null; const mean = window.reduce((a, b) => a + b, 0) / window.length; const variance = window.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (window.length - 1); return { mean, variance: Math.max(0, variance) }; }
function rollingZScore(arr, index, lookback) { const stats = rollingMeanVariance(arr, index, lookback); if (!stats) return null; const v = Number(arr[index]); const std = Math.sqrt(stats.variance); if (!Number.isFinite(v) || !Number.isFinite(std) || std === 0) return null; return (v - stats.mean) / std; }
function sigmoid(x, slope = 1) { if (!Number.isFinite(x)) return null; const t = Math.max(Math.min(x * slope, 60), -60); return 1 / (1 + Math.exp(-t)); }
function frobeniusDiff(matrix, prevMatrix) { if (!Array.isArray(matrix) || !Array.isArray(prevMatrix)) return null; let sumSq = 0; let count = 0; const n = Math.min(matrix.length, prevMatrix.length); for (let i = 0; i < n; i += 1) { const row = matrix[i]; const prow = prevMatrix[i]; if (!Array.isArray(row) || !Array.isArray(prow)) continue; const m = Math.min(row.length, prow.length); for (let j = 0; j < m; j += 1) { const a = Number(row[j]); const b = Number(prow[j]); if (!Number.isFinite(a) || !Number.isFinite(b)) continue; const d = a - b; sumSq += d * d; count += 1; } } if (count === 0) return null; return Math.sqrt(sumSq / count); }
function quantile(values, q) { if (!Array.isArray(values) || values.length === 0) return NaN; const sorted = [...values].sort((a, b) => a - b); const pos = Math.min(Math.max(q, 0), 1) * (sorted.length - 1); const lower = Math.floor(pos); const upper = Math.ceil(pos); if (lower === upper) return sorted[lower]; const weight = pos - lower; return sorted[lower] * (1 - weight) + sorted[upper] * weight; }
function median(xs){ const a=xs.slice().sort((x,y)=>x-y); const n=a.length; if(n===0) return null; const m=Math.floor(n/2); return n%2? a[m] : 0.5*(a[m-1]+a[m]); }
function rollingSigmaMAD(arr, end, win){ const xs=[]; for(let k=Math.max(0,end-win+1); k<=end; k+=1){ const v=arr[k]; if(Number.isFinite(v)) xs.push(v);} if(xs.length<5) return null; const med=median(xs); const dev=xs.map(v=>Math.abs(v-med)); const mad=median(dev); return Number.isFinite(mad)? 1.4826*mad : null; }

// Minimal power-iteration to get top eigenvector of symmetric matrix
function topEigenvectorLocal(matrix, maxIter = 60, tol = 1e-8) {
  if (!Array.isArray(matrix) || matrix.length === 0) return null;
  const n = matrix.length;
  let v = new Array(n).fill(1 / Math.sqrt(n));
  for (let it = 0; it < maxIter; it += 1) {
    const w = new Array(n).fill(0);
    for (let i = 0; i < n; i += 1) {
      let s = 0; const row = matrix[i];
      for (let j = 0; j < n; j += 1) s += row[j] * v[j];
      w[i] = s;
    }
    const norm = Math.sqrt(w.reduce((a, x) => a + x * x, 0)) || 1;
    const nv = w.map((x) => x / norm);
    let diff = 0; for (let i = 0; i < n; i += 1) { const d = nv[i] - v[i]; diff += d * d; }
    v = nv; if (diff < tol * tol) break;
  }
  return v;
}

const state = { window: WINDOW, priceSeries: {}, metrics: {} };

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
  const latest = records[records.length - 1]; const average180 = MM.mean(stabilityValues.slice(-180));
  const pairs = buildPairSeries(records, window, allSymbols);
  return { records, latest, average180, pairs };
}

function selectMatrixForSymbols(record, symbols) {
  if (!record || !Array.isArray(symbols)) return null;
  if (Array.isArray(record.fullMatrix) && record.fullMatrix.length === symbols.length) return record.fullMatrix;
  if (Array.isArray(record.matrix) && record.matrix.length === symbols.length) return record.matrix;
  return null;
}
function buildPairSeries(records, window, symbols) {
  const pairs = {}; const priceOffset = window - 1;
  for (let i = 0; i < symbols.length; i += 1) {
    for (let j = i + 1; j < symbols.length; j += 1) {
      const key = `${symbols[i]}|${symbols[j]}`; pairs[key] = { dates: [], correlation: [], priceA: [], priceB: [] };
    }
  }
  records.forEach((record, idx) => {
    const matrix = selectMatrixForSymbols(record, symbols); const priceIndex = priceOffset + idx;
    for (let i = 0; i < symbols.length; i += 1) {
      for (let j = i + 1; j < symbols.length; j += 1) {
        const key = `${symbols[i]}|${symbols[j]}`; const pair = pairs[key];
        pair.dates.push(record.date); const corrValue = matrix ? matrix[i][j] : null; pair.correlation.push(Number.isFinite(corrValue) ? corrValue : null);
      }
    }
  });
  return pairs;
}

function computeRiskSeriesFFL(metrics, recordsOverride) {
  if (!metrics || !Array.isArray(metrics.records) || metrics.records.length === 0) return null;
  const labels = SIGNAL.symbols; const indexLookup = {}; labels.forEach((s, i) => { indexLookup[s] = i; });
  const allRecords = metrics.records; const alignedRecords = Array.isArray(recordsOverride) && recordsOverride.length > 0 ? recordsOverride : null;
  const firstDate = alignedRecords?.[0]?.date ?? allRecords[0]?.date; let baseIdx = allRecords.findIndex((r) => r.date === firstDate); if (baseIdx < 0) baseIdx = 0;
  const length = alignedRecords ? alignedRecords.length : allRecords.length - baseIdx; if (length <= 0) return null;
  const slice = allRecords.slice(baseIdx, baseIdx + length); const dates = slice.map((r) => r.date);
  const pairSeries = metrics.pairs?.[SIGNAL.pairKey] || null; const scCorr = pairSeries?.correlation?.slice(baseIdx, baseIdx + length) || new Array(length).fill(null);
  const safeNeg = slice.map((r) => Number(r.sub?.safeNegative));
  const windowOffset = Math.max(1, Number(state.window) - 1);
  const prices = state.priceSeries || {};

  // --- BearGuard: downside-asymmetry damper (minimal) ---
  const riskSet = ['QQQ', 'IWM', 'BTC-USD'].filter((symbol) => Array.isArray(prices[symbol]) && prices[symbol].length > 0);
  const span60Alpha = 2 / (60 + 1);
  const EPS = 1e-12;
  const maxLen = riskSet.reduce((acc, symbol) => Math.max(acc, prices[symbol]?.length || 0), 0);
  const RStar = new Array(maxLen).fill(null);
  riskSet.forEach((symbol) => {
    const series = prices[symbol] || [];
    if (!Array.isArray(series) || series.length === 0) return;
    let posEwma = null;
    let negEwma = null;
    for (let idx = 1; idx < series.length; idx += 1) {
      const prev = Number(series[idx - 1]);
      const curr = Number(series[idx]);
      if (!Number.isFinite(prev) || !Number.isFinite(curr) || prev <= 0 || curr <= 0) continue;
      const ret = Math.log(curr) - Math.log(prev);
      if (!Number.isFinite(ret)) continue;
      const posSq = ret > 0 ? ret * ret : 0;
      const negSq = ret < 0 ? ret * ret : 0;
      posEwma = posEwma == null ? posSq : (span60Alpha * posSq) + ((1 - span60Alpha) * posEwma);
      negEwma = negEwma == null ? negSq : (span60Alpha * negSq) + ((1 - span60Alpha) * negEwma);
      const denom = (posEwma ?? 0) + EPS;
      const ratioRaw = denom > 0 ? (negEwma ?? 0) / denom : 0;
      if (!Number.isFinite(ratioRaw)) continue;
      const ratio = Math.max(0, Math.min(5, ratioRaw));
      const existing = RStar[idx];
      RStar[idx] = Number.isFinite(existing) ? Math.max(existing, ratio) : ratio;
    }
  });
  for (let idx = 0; idx < RStar.length; idx += 1) {
    if (!Number.isFinite(RStar[idx])) RStar[idx] = 1;
  }
  function bearDamp(idx, RThr = 1.2, beta = 0.5, gmin = 0.5) {
    if (!Number.isInteger(idx) || idx < 0 || idx >= RStar.length) return 1;
    const r = Number.isFinite(RStar[idx]) ? RStar[idx] : 1;
    const x = Math.max(0, r - RThr);
    return Math.max(gmin, 1 / (1 + beta * x));
  }
  // --- end BearGuard ---

  const mm = new Array(length).fill(null); const guard = new Array(length).fill(null); const score = new Array(length).fill(null);
  const jFlux = new Array(length).fill(null); const jRiskBeta = new Array(length).fill(null); const fullFlux = new Array(length).fill(null); const fullFluxZ = new Array(length).fill(null);
  const mmTrend = new Array(length).fill(null); const fluxSlope = new Array(length).fill(null); const diffusionScore = new Array(length).fill(null); const vPC1 = new Array(length).fill(null); const kappa = new Array(length).fill(null);
  const apdf = new Array(length).fill(null); const pcon = new Array(length).fill(null);
  const fluxRaw = new Array(length).fill(null); const fluxIntensity = new Array(length).fill(null); const comboMomentum = new Array(length).fill(null); const breadth = new Array(length).fill(null);
  const coDownAll = new Array(length).fill(null);
  const fragile = new Array(length).fill(false); const far = new Array(length).fill(null); let prevMatrix = null;

  // EXP timing accumulators
  let efPrev = null, esPrev = null; const alpha3 = 2 / (3 + 1), alpha10 = 2 / (10 + 1);

  for (let i = 0; i < length; i += 1) {
    const record = allRecords[baseIdx + i]; const matrix = record?.matrix;
    if (Array.isArray(matrix) && matrix.length > 0) { const lambda1 = MM.topEigenvalue(matrix) || 0; mm[i] = lambda1 / (matrix.length || 1); }
    if (i > 0 && Number.isFinite(mm[i]) && Number.isFinite(mm[i - 1])) mmTrend[i] = mm[i] - mm[i - 1]; else mmTrend[i] = null;

    const priceIndex = baseIdx + i + windowOffset; const zr = {}; labels.forEach((symbol) => {
      zr[symbol] = zMomentumFromSeries(prices[symbol], priceIndex, RISK_CFG_FFL.lookbacks.momentum, RISK_CFG_FFL.lookbacks.vol, RISK_CFG_FFL.zSat);
    });
    let weightSum = 0; let fluxSum = 0; let absSum = 0;
    let apdfWeighted = 0; let pconWeighted = 0; let weightsAll = 0;
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
    for (let a = 0; a < CLUSTERS.risk.length; a += 1) {
      for (let b = a + 1; b < CLUSTERS.risk.length; b += 1) {
        const ia = SIGNAL.symbols.indexOf(CLUSTERS.risk[a]); const ib = SIGNAL.symbols.indexOf(CLUSTERS.risk[b]);
        const coef = Array.isArray(matrix?.[ia]) ? (Number(matrix[ia][ib]) || 0) : 0; const w = Math.pow(Math.abs(coef), RISK_CFG_FFL.p);
        if (!Number.isFinite(w) || w <= 0) continue; const aZ = Number.isFinite(zr[CLUSTERS.risk[a]]) ? zr[CLUSTERS.risk[a]] : 0; const bZ = Number.isFinite(zr[CLUSTERS.risk[b]]) ? zr[CLUSTERS.risk[b]] : 0;
        apdfWeighted += w * (0.5 * (aZ + bZ)); pconWeighted += w * ((aZ > 0 && bZ > 0) ? 1 : 0); weightsAll += w;
      }
    }
    for (let a = 0; a < CLUSTERS.safe.length; a += 1) {
      for (let b = a + 1; b < CLUSTERS.safe.length; b += 1) {
        const ia = SIGNAL.symbols.indexOf(CLUSTERS.safe[a]); const ib = SIGNAL.symbols.indexOf(CLUSTERS.safe[b]);
        const coef = Array.isArray(matrix?.[ia]) ? (Number(matrix[ia][ib]) || 0) : 0; const w = Math.pow(Math.abs(coef), RISK_CFG_FFL.p);
        if (!Number.isFinite(w) || w <= 0) continue; const aZ = Number.isFinite(zr[CLUSTERS.safe[a]]) ? zr[CLUSTERS.safe[a]] : 0; const bZ = Number.isFinite(zr[CLUSTERS.safe[b]]) ? zr[CLUSTERS.safe[b]] : 0;
        apdfWeighted += w * (-0.5 * (aZ + bZ)); pconWeighted += w * ((aZ <= 0 && bZ <= 0) ? 1 : 0); weightsAll += w;
      }
    }
    const Jbar = weightSum > 0 ? fluxSum / weightSum : 0; const Jnorm = Math.tanh(Jbar / RISK_CFG_FFL.lambda);
    jFlux[i] = Number.isFinite(Jnorm) ? Jnorm * bearDamp(priceIndex) : null; fluxRaw[i] = weightSum > 0 ? Jbar : null; fluxIntensity[i] = weightSum > 0 ? absSum / weightSum : null;
    if (i > 0 && Number.isFinite(jFlux[i]) && Number.isFinite(jFlux[i - 1])) fluxSlope[i] = jFlux[i] - jFlux[i - 1]; else fluxSlope[i] = null;

    const fraw = frobeniusDiff(matrix, prevMatrix); prevMatrix = matrix; fullFlux[i] = Number.isFinite(fraw) ? fraw : null;
    const z = rollingZScore(fullFlux, i, Math.min(63, Math.max(15, Math.floor(length / 4)))); fullFluxZ[i] = Number.isFinite(z) ? z : null;
    const apdfRaw = weightsAll > 0 ? apdfWeighted / weightsAll : 0; apdf[i] = Number.isFinite(apdfRaw) ? Math.tanh(apdfRaw / (RISK_CFG_FFL.lambda || 0.25)) : null;
    pcon[i] = weightsAll > 0 ? Math.max(0, Math.min(1, pconWeighted / weightsAll)) : null;
    const riskZValues = CLUSTERS.risk.map((s) => zr[s]).filter((v) => Number.isFinite(v));
    const allZValues_i = SIGNAL.symbols.map((s) => zr[s]).filter((v) => Number.isFinite(v));
    const downAll_i = allZValues_i.length > 0 ? (allZValues_i.filter((v) => v < 0).length / allZValues_i.length) : null;
    coDownAll[i] = Number.isFinite(downAll_i) ? downAll_i : null;
    comboMomentum[i] = riskZValues.length > 0 ? riskZValues.reduce((a, v) => a + v, 0) / riskZValues.length : null;
    breadth[i] = riskZValues.length > 0 ? riskZValues.filter((v) => v > 0).length / riskZValues.length : null;
    const safePen = normalizeRangeSafe(safeNeg[i], 0.35, 0.60);
    const mmPen = normalizeRangeSafe(mm[i], 0.85, 0.97);
    const deltaPen = normalizeRangeSafe(Math.max(0, -(record?.delta ?? 0)), 0.015, 0.05);
    const fluxGuard = Number.isFinite(fullFluxZ[i]) ? sigmoid(fullFluxZ[i], 0.85) : 1;
    const guardVal = 0.4 * mmPen + 0.2 * safePen + 0.2 * deltaPen + 0.2 * fluxGuard; guard[i] = Number.isFinite(guardVal) ? Number(guardVal.toFixed(6)) : null;
    const fluxScore = jFlux[i] == null ? null : 0.5 * (1 + jFlux[i]);
    const comboNorm = comboMomentum[i] == null ? null : clamp01(((comboMomentum[i] ?? 0) + 1) / 2);
    const breadthNorm = breadth[i] == null ? null : clamp01(breadth[i] ?? 0);
    const guardRelief = guardVal == null ? null : clamp01(1 - guardVal);
    const components = [];
    if (Number.isFinite(fluxScore)) components.push({ weight: 0.5, value: fluxScore });
    if (Number.isFinite(comboNorm)) components.push({ weight: 0.2, value: comboNorm });
    if (Number.isFinite(breadthNorm)) components.push({ weight: 0.2, value: breadthNorm });
    if (Number.isFinite(guardRelief)) components.push({ weight: 0.1, value: guardRelief });
    const totalWeight = components.reduce((acc, it) => acc + it.weight, 0);
    const aggregated = totalWeight > 0 ? components.reduce((acc, it) => acc + it.weight * it.value, 0) / totalWeight : (Number.isFinite(fluxScore) ? fluxScore : 0.5);
    score[i] = Number(aggregated.toFixed(6));
    // Diffusion Score (Fick)
    const k1 = 0.50; const k2 = 0.15; const mt = Number.isFinite(mmTrend[i]) ? Math.max(0, mmTrend[i]) : 0; const fsNeg = Number.isFinite(fluxSlope[i]) ? Math.max(0, -fluxSlope[i]) : 0;
    const diff = (Number.isFinite(jFlux[i]) ? jFlux[i] : 0) - (k1 * mt) - (k2 * fsNeg); diffusionScore[i] = Number.isFinite(diff) ? Number(diff.toFixed(6)) : null;

    // PC1 velocity via simple power iteration eigenvector
    try {
      if (Array.isArray(matrix) && matrix.length > 0) {
        const e1 = topEigenvectorLocal(matrix);
        if (Array.isArray(e1)) {
          let num = 0; let den = 0;
          for (let ei = 0; ei < e1.length; ei += 1) {
            const sym = SIGNAL.symbols[ei];
            const zi = Number.isFinite(zr[sym]) ? zr[sym] : 0;
            num += e1[ei] * zi; den += Math.abs(e1[ei]);
          }
          const vproj = den > 0 ? (num / den) : 0;
          vPC1[i] = Math.tanh(vproj / 0.5);
        }
      }
    } catch (e) { vPC1[i] = null; }
  }

  const th = RISK_CFG_FFL.thresholds; const variant = RISK_CFG_FFL.variant; const validFlux = jFlux.filter((v) => Number.isFinite(v)); const validScore = score.filter((v) => Number.isFinite(v));
  let dynOnFlux = th.jOn; let dynOffFlux = th.jOff; let dynScoreOn = th.scoreOn; let dynScoreOff = th.scoreOff;
  if (variant !== 'classic' && validFlux.length >= 50) { dynOnFlux = Math.max(th.jOn, quantile(validFlux, 0.75)); dynOffFlux = Math.min(th.jOff, quantile(validFlux, 0.25)); }
  if (variant !== 'classic' && validScore.length >= 50) { dynScoreOn = Math.max(th.scoreOn, quantile(validScore, 0.75)); dynScoreOff = Math.min(th.scoreOff, quantile(validScore, 0.25)); }

  const stateArr = new Array(length).fill(0);
  const expRatio = new Array(length).fill(null); // EXP ratio buffer
  const st = new Array(length).fill(null); const dSt = new Array(length).fill(null); const sigma = new Array(length).fill(null);
  let prevState = 0; let onCand = 0; let offCand = 0; let offGuardSeq = 0;
  const DRIFT_MIN_DAYS = 5; const DRIFT_COOLDOWN_DAYS = 2;
  let driftSeq = 0; let driftCooldown = 0; let inDriftEpoch = false; let prevInDrift = false;
  let driftEpochs = 0; let driftDays = 0; let offDays = 0; let offFromDriftDays = 0; let suppressedByDrift = 0;
  for (let i = 0; i < length; i += 1) {
    const fluxVal = jFlux[i] ?? 0; const diffVal = diffusionScore[i] ?? fluxVal; const mmValue = mm[i] ?? 0; const comboValue = comboMomentum[i] ?? null; const breadthValue = breadth[i] ?? null; const guardVal = guard[i] ?? 1;
    const vpc1Val = Number.isFinite(vPC1[i]) ? vPC1[i] : 0; const lam = 1.0; const kappaVal = (Math.abs(diffVal)) / (Math.abs(diffVal) + lam * Math.abs(vpc1Val) + 1e-6); kappa[i] = Number.isFinite(kappaVal) ? kappaVal : null;
    // EXP ratio (if variant=exp), else use diffusionScore directly for timing
    const lamExp = Number.isFinite(RISK_CFG_FFL?.exp?.lam) ? RISK_CFG_FFL.exp.lam : (Number(process.env.ELAM) || 1.0);
    const denom = lamExp * Math.abs(vpc1Val) + 1e-6;
    const rawRatio = Number.isFinite(diffVal) ? (diffVal / denom) : null;
    if (variant === 'exp') {
      const prev = i > 0 ? expRatio[i - 1] : null;
      expRatio[i] = (Number.isFinite(rawRatio) && Number.isFinite(prev)) ? (0.5 * prev + 0.5 * rawRatio) : (Number.isFinite(rawRatio) ? rawRatio : prev);
    }
    const x = (process.env.TI_SRC === 'ratio') ? (expRatio[i] ?? rawRatio) : (diffusionScore[i] ?? null);
    if (Number.isFinite(x)) {
      efPrev = (efPrev == null) ? x : (alpha3 * x + (1 - alpha3) * efPrev);
      esPrev = (esPrev == null) ? x : (alpha10 * x + (1 - alpha10) * esPrev);
      st[i] = efPrev - esPrev;
      dSt[i] = (i > 0 && Number.isFinite(st[i - 1])) ? (st[i] - st[i - 1]) : null;
      sigma[i] = rollingSigmaMAD(st, i, Number.isFinite(Number(process.env.WIN)) ? Number(process.env.WIN) : 63);
    }
    const guardValue = Number.isFinite(guardVal) ? guardVal : 1; const guardSoft = 0.95; const guardHard = 0.98; const breadthGate = (breadthValue ?? 0) >= ((th.breadthOn ?? 0.5) * 0.6);
    const dynOnAdj = dynOnFlux + (mmValue >= 0.94 ? 0.05 : mmValue >= 0.90 ? 0.03 : 0);
    const pconOkBase = !Number.isFinite(pcon[i]) || pcon[i] >= (RISK_CFG_FFL.thresholds.pconOn ?? 0.55);
    const apdfOk = !Number.isFinite(apdf[i]) || apdf[i] >= -0.05;
    const pconOk = pconOkBase || (diffVal >= (dynOnAdj + 0.07));
    const benchIdx = baseIdx + i + windowOffset;
    const bench10 = rollingReturnFromSeries(prices[SIGNAL.trade.baseSymbol], benchIdx, 10);
    const bench20 = rollingReturnFromSeries(prices[SIGNAL.trade.baseSymbol], benchIdx, 20);
    const hiCorrBear = (mmValue >= 0.90) && (Number.isFinite(bench10) ? bench10 <= 0 : true);
    const hiCorrDrift = (((Number.isFinite(coDownAll[i]) ? coDownAll[i] >= 0.60 : false) && ((fluxVal ?? 0) <= 0)) ||
      ((mmValue >= 0.90) && (Number.isFinite(bench20) ? bench20 <= 0 : true)));
    if (hiCorrDrift) { driftSeq += 1; driftCooldown = 0; } else { driftSeq = 0; driftCooldown += 1; }
    if (driftSeq >= DRIFT_MIN_DAYS) inDriftEpoch = true;
    if (inDriftEpoch && driftCooldown >= DRIFT_COOLDOWN_DAYS) inDriftEpoch = false;
    if (!prevInDrift && inDriftEpoch) driftEpochs += 1;
    prevInDrift = inDriftEpoch;
    const stricter = !hiCorrBear || ((diffVal >= (dynOnAdj + 0.05)) && (Number.isFinite(pcon[i]) ? pcon[i] >= 0.65 : true) && (Number.isFinite(apdf[i]) ? apdf[i] >= 0 : true) && (Number.isFinite(comboValue) ? comboValue >= 0.10 : true));
    const onClassicMain = (diffVal >= dynOnAdj) && pconOk && apdfOk && stricter && guardValue < guardSoft && mmValue < th.mmOff && breadthGate;
    const onClassicAlt = !hiCorrBear && (Number.isFinite(jRiskBeta[i]) && jRiskBeta[i] >= 0.06) && (Number.isFinite(comboValue) && comboValue >= 0.10) && guardValue < 0.90;
    const strongOn = (diffVal >= (dynOnAdj + 0.03)) && (Number.isFinite(kappaVal) ? kappaVal >= (RISK_CFG_FFL.thresholds.kOn ?? 0.60) : true) && (Number.isFinite(pcon[i]) ? pcon[i] >= Math.max(0.65, (RISK_CFG_FFL.thresholds.pconOn ?? 0.55)) : true) && (Number.isFinite(vpc1Val) ? vpc1Val >= (RISK_CFG_FFL.thresholds.vOn ?? 0.05) : true) && guardValue < guardSoft && mmValue < th.mmOff;
    // EXP gates layered on top
    let expOkOn = true; let expForceOff = false;
    if (variant === 'exp') {
      const ratioS = expRatio[i];
      const rOn = Number.isFinite(RISK_CFG_FFL?.exp?.rOn) ? RISK_CFG_FFL.exp.rOn : 1.20;
      const rOff = Number.isFinite(RISK_CFG_FFL?.exp?.rOff) ? RISK_CFG_FFL.exp.rOff : -1.10;
      const breadthMin = Number.isFinite(RISK_CFG_FFL?.exp?.breadth1dMin) ? RISK_CFG_FFL.exp.breadth1dMin : 0.55;
      const idxPrice0 = baseIdx + i + windowOffset;
      const rQQQ = rollingReturnFromSeries(prices[SIGNAL.trade.baseSymbol], idxPrice0, 1);
      const rIWM = rollingReturnFromSeries(prices['IWM'], idxPrice0, 1);
      const anyPos = (Number.isFinite(rQQQ) ? rQQQ > 0 : false) || (Number.isFinite(rIWM) ? rIWM > 0 : false);
      const bothPos = (Number.isFinite(rQQQ) ? rQQQ > 0 : false) && (Number.isFinite(rIWM) ? rIWM > 0 : false);
      const needBoth = (RISK_CFG_FFL?.exp?.d0BothPosHiCorr ? (mmValue >= 0.90) : false);
      const day0Ok = needBoth ? bothPos : (RISK_CFG_FFL?.exp?.d0AnyPos ? anyPos : true);
      const breadth1dOk = (mmValue >= 0.90) ? ((breadthValue ?? 0) >= breadthMin) : true;
      const onK = Number.isFinite(Number(process.env.ONK)) ? Number(process.env.ONK) : 0.75;
      const offK = Number.isFinite(Number(process.env.OFFK)) ? Number(process.env.OFFK) : 0.50;
      const hiScale = Number.isFinite(Number(process.env.HCORR)) ? Number(process.env.HCORR) : 1.25;
      const sig = sigma[i]; const tauOn = Number.isFinite(sig) ? onK * sig : 0; const tauOff = Number.isFinite(sig) ? offK * sig : 0; const scale = mmValue >= 0.90 ? hiScale : 1.0;
      const stOn = Number.isFinite(st[i]) && Number.isFinite(dSt[i]) && (st[i] > scale * tauOn) && (dSt[i] > 0) && (Number(ratioS) > 0);
      const stOff = Number.isFinite(st[i]) && Number.isFinite(dSt[i]) && (st[i] < -scale * tauOff) && (dSt[i] < 0) && (Number(ratioS) < 0);
      expOkOn = (Number.isFinite(ratioS) ? (ratioS >= rOn) : true) && day0Ok && breadth1dOk && (stOn || !Number.isFinite(sig));
      expForceOff = (Number.isFinite(ratioS) ? (ratioS <= rOff) : false) || stOff;
    }
    const onExpAlt = (variant === 'exp') && expOkOn && guardValue < guardSoft && mmValue < th.mmOff;
    const onClassic = ((onClassicMain || onClassicAlt || strongOn) || onExpAlt) && !hiCorrDrift;
    const offFlux = fluxVal <= dynOffFlux; const offGuard = (guardValue >= guardHard) || (mmValue >= th.mmOff); const guardOnly = offGuard && !offFlux;
    const breadthGateLoosen = (breadthValue ?? 0) >= ((th.breadthOn ?? 0.5) * 0.5); const trendSupport = (fluxVal >= (dynOnFlux - 0.03)) && breadthGateLoosen && (Number.isFinite(comboValue) ? comboValue >= 0 : true);
    const hiCorr = mmValue >= 0.92; const guardConfirmDays = (hiCorr && trendSupport) ? 3 : 2; if (guardOnly) offGuardSeq += 1; else offGuardSeq = 0; const offGuardConfirmed = guardOnly && offGuardSeq >= guardConfirmDays;
    // Terminal breaker: end-of-run cut using EXP timing + high absorption
    let endBreaker = false;
    {
      const TR_ON = String(process.env.TRON || '1') === '1';
      if (TR_ON) {
        const tr_mm = Number.isFinite(Number(process.env.TRMM)) ? Number(process.env.TRMM) : 0.95;
        const tr_kappa = Number.isFinite(Number(process.env.TRKAPPA)) ? Number(process.env.TRKAPPA) : 0.60;
        const onK = Number.isFinite(Number(process.env.ONK)) ? Number(process.env.ONK) : 0.75;
        const hiScale = Number.isFinite(Number(process.env.HCORR)) ? Number(process.env.HCORR) : 1.25;
        const sig = sigma[i];
        const tauOff = Number.isFinite(sig) ? (Number.isFinite(Number(process.env.OFFK)) ? Number(process.env.OFFK) : 0.50) * sig : 0;
        const scale = mmValue >= 0.90 ? hiScale : 1.0;
        endBreaker = (mmValue >= tr_mm)
          && Number.isFinite(st[i]) && Number.isFinite(dSt[i])
          && (st[i] < -scale * tauOff) && (dSt[i] < 0)
          && ((fluxSlope[i] ?? 0) < 0)
          && ((Number.isFinite(kappaVal) ? kappaVal : 1) < tr_kappa || (vpc1Val <= 0));
      }
    }

    const offByRel = ((Number.isFinite(vpc1Val) ? vpc1Val <= (RISK_CFG_FFL.thresholds.vOff ?? -0.05) : false) && (Math.abs(vpc1Val) >= 0.05)) || ((diffVal <= dynOffFlux) && (Number.isFinite(kappaVal) ? kappaVal < 0.55 : false)) || (endBreaker) || (variant === 'exp' && expForceOff);
    let offClassic = offByRel || offFlux || offGuardConfirmed || (Number.isFinite(pcon[i]) && pcon[i] <= (RISK_CFG_FFL.thresholds.pconOff ?? 0.40) && mmValue >= 0.92);
    if (hiCorrDrift) {
      offClassic = true;
    }
    const idxPrice = baseIdx + i + windowOffset; const rList = CLUSTERS.risk.map((sym) => rollingReturnFromSeries(prices[sym], idxPrice, 3)).filter(Number.isFinite);
    const risk3 = rList.length ? rList.reduce((a, b) => a + b, 0) / rList.length : null; const blockOnHighCorrDown = Number.isFinite(risk3) && risk3 <= 0 && mmValue >= 0.90;
    const rawOn = onClassic && !blockOnHighCorrDown; const rawOff = offClassic;
    const hiCorrRisk = (mmValue >= 0.90) || ((mmTrend[i] ?? 0) > 0.005) || ((fullFluxZ[i] ?? 0) >= 1.5);
    const accel = (fluxSlope[i] ?? 0) > 0 && (mmTrend[i] ?? 0) <= 0;
    const strongPcon = Number.isFinite(pcon[i]) && pcon[i] >= 0.68;
    let confirmOnDays = strongOn ? 1 : (hiCorrRisk ? 3 : strongPcon ? 1 : (accel ? 1 : 2));
    if (variant === 'exp') {
      const strongK = Number.isFinite(Number(process.env.STRONGK)) ? Number(process.env.STRONGK) : 1.5;
      const sig = sigma[i]; const tOn = Number.isFinite(sig) ? (Number.isFinite(Number(process.env.ONK)) ? Number(process.env.ONK) : 0.75) * sig : null;
      if (Number.isFinite(st[i]) && Number.isFinite(tOn) && st[i] >= strongK * tOn) confirmOnDays = 1;
    }
    onCand = rawOn ? onCand + 1 : 0; offCand = rawOff ? offCand + 1 : 0;
    let decided = prevState; if (prevState === 1) { if (rawOff) decided = -1; else decided = 1; }
    else if (prevState === -1) { if (onCand >= confirmOnDays) decided = 1; else decided = -1; }
    else { if (offCand >= 1) decided = -1; else if (onCand >= confirmOnDays) decided = 1; else decided = 0; }
    const decidedBeforeDrift = decided;
    if (inDriftEpoch) decided = -1;
    stateArr[i] = decided; prevState = decided;
    if (inDriftEpoch) driftDays += 1;
    if (decided === -1) offDays += 1;
    if (inDriftEpoch && decided === -1) offFromDriftDays += 1;
    if (inDriftEpoch && decidedBeforeDrift !== -1 && decided === -1) suppressedByDrift += 1;
    far[i] = Number.isFinite(jFlux[i]) && Number.isFinite(mmValue) && mmValue > 0 ? Math.abs(jFlux[i]) / (mmValue + 1e-9) : null;
  }
  const executedState = stateArr.map((v, i) => (i === 0 ? 0 : stateArr[i - 1] || 0));
  return { dates, score, state: stateArr, executedState, fragile, guard, mm, far, fflFlux: jFlux, riskBetaFlux: jRiskBeta, apdf, pcon, diffusionScore, fluxSlope, mmTrend, fullFlux, fullFluxZ, fluxRaw, fluxIntensity, comboMomentum, breadth, scCorr, safeNeg, drift: { epochs: driftEpochs, days: driftDays, offDays, offFromDriftDays, suppressed: suppressedByDrift } };
}

function computeHitRate(stateArr, fwdRetArr) { let wins = 0; let cnt = 0; for (let i = 0; i < Math.min(stateArr.length, fwdRetArr.length); i += 1) { const s = stateArr[i]; const r = fwdRetArr[i]; if (!Number.isFinite(r)) continue; if (s > 0 && r > 0) wins += 1; else if (s < 0 && r < 0) wins += 1; else if (s === 0 && r >= 0) wins += 1; cnt += 1; } return cnt > 0 ? wins / cnt : 0; }

function loadHistorical() { const raw = fs.readFileSync(DATA_PATH, 'utf8'); const json = JSON.parse(raw); if (!Array.isArray(json.assets)) throw new Error('historical_prices.json: assets[] missing'); const bySymbol = new Map(json.assets.map((a) => [a.symbol, a])); const seriesList = ASSETS.map(({ symbol, category }) => { const a = bySymbol.get(symbol); if (!a) throw new Error(`Missing asset ${symbol} in historical file`); return { symbol, category, dates: a.dates, prices: a.prices }; }); return seriesList; }
function yearlyReturnsFromEquity(dates, equity) { const out = new Map(); for (let i = 1; i < dates.length; i += 1) { const y = dates[i].slice(0, 4); if (!out.has(y)) out.set(y, { start: equity[i - 1], end: equity[i] }); else out.set(y, { start: out.get(y).start, end: equity[i] }); } return Array.from(out.entries()).map(([year, { start, end }]) => ({ year: Number(year), ret: (end / start - 1) })); }

function main() {
  const seriesList = loadHistorical(); const aligned = MM.alignSeries(seriesList); const returns = MM.computeReturns(aligned); state.priceSeries = returns.priceSeries;
  const metrics = computeWindowMetrics(WINDOW, returns, aligned); state.metrics[WINDOW] = metrics;
  const filteredRecords = metrics.records.filter((r) => r.date >= RANGE_START && r.date <= RANGE_END);
  const ffl = computeRiskSeriesFFL(metrics, filteredRecords); if (!ffl) throw new Error('Failed to compute FFL series');
  const baseSymbol = SIGNAL.trade.baseSymbol; const windowOffset = Math.max(1, WINDOW - 1); const firstDate = filteredRecords?.[0]?.date; let baseIdx = (metrics.records || []).findIndex((r) => r.date === firstDate); if (baseIdx < 0) baseIdx = 0;
  const dates = ffl.dates; const prices = state.priceSeries[baseSymbol] || []; const baseReturns = []; const leveredReturns = [];
  for (let idx = 0; idx < dates.length; idx += 1) { const priceIndex = windowOffset + baseIdx + idx; const prevIndex = priceIndex - 1; let daily = 0; if (prices[priceIndex] != null && prices[prevIndex] != null && prices[prevIndex] !== 0) daily = prices[priceIndex] / prices[prevIndex] - 1; baseReturns.push(daily); leveredReturns.push(Math.max(-0.99, SIGNAL.trade.leverage * daily)); }
  const executedState = Array.isArray(ffl.executedState) && ffl.executedState.length === ffl.state.length ? ffl.executedState : ffl.state.map((v, i) => (i === 0 ? 0 : ffl.state[i - 1] || 0));
  const stratReturns = executedState.map((regime, i) => {
    if (regime > 0) {
      const idxPrice = baseIdx + i + windowOffset;
      const bench10 = rollingReturnFromSeries(state.priceSeries[SIGNAL.trade.baseSymbol], idxPrice, 10);
      const hiCorrBear = (ffl.mm?.[i] ?? 0) >= 0.90 && (Number.isFinite(bench10) ? bench10 <= 0 : true);
      // Dynamic leverage (available for any variant)
      let lev = SIGNAL.trade.leverage;
      if ((process.env.LEV_MODE === '1')) {
        const r = Array.isArray(ffl?.diffusionScore) && ffl.diffusionScore[i] != null && Number.isFinite(ffl.vPC1?.[i])
          ? (ffl.diffusionScore[i] / ((Number(process.env.ELAM) || 1.0) * Math.abs(ffl.vPC1[i] || 0) + 1e-6))
          : null;
        const r0 = Number.isFinite(Number(process.env.LEV_R0)) ? Number(process.env.LEV_R0) : 0.50;
        const r1 = Number.isFinite(Number(process.env.LEV_R1)) ? Number(process.env.LEV_R1) : 1.20;
        const lmin = Number.isFinite(Number(process.env.LEV_MIN)) ? Number(process.env.LEV_MIN) : 1.0;
        const lmax = Number.isFinite(Number(process.env.LEV_MAX)) ? Number(process.env.LEV_MAX) : 3.0;
        let p = 0;
        if (Number.isFinite(r)) {
          p = (r - r0) / (r1 - r0);
          if (p < 0) p = 0; if (p > 1) p = 1;
        }
        lev = lmin + (lmax - lmin) * p;
        // In high-correlation bearish drift, damp leverage further
        const dampCap = Number.isFinite(Number(process.env.LEV_DAMP)) ? Number(process.env.LEV_DAMP) : 1.25;
        if (hiCorrBear) lev = Math.max(lmin, Math.min(lev, dampCap));
      } else if (hiCorrBear) {
        lev = 1;
      }
      return Math.max(-0.99, lev * baseReturns[i]);
    }
    if (regime < 0) return 0;
    return baseReturns[i];
  });
  const eqStrat = []; const eqBH = []; let s = 1; let b = 1; for (let i = 0; i < stratReturns.length; i += 1) { s *= 1 + (stratReturns[i] || 0); b *= 1 + (baseReturns[i] || 0); eqStrat.push(Number(s.toFixed(6))); eqBH.push(Number(b.toFixed(6))); }
  const ret = baseReturns; const fwd1 = ret.slice(1).map((_, i) => ret[i + 1]); const state1 = ffl.state.slice(0, ffl.state.length - 1); const hr1 = computeHitRate(state1, fwd1);
  const horizon = 5; const fwd5 = []; for (let i = 0; i < ret.length; i += 1) { let prod = 1; for (let k = 1; k <= horizon && i + k < ret.length; k += 1) prod *= 1 + ret[i + k]; fwd5.push(prod - 1); } const state5 = ffl.state.slice(0, ffl.state.length - 1); const hr5 = computeHitRate(state5, fwd5.slice(0, state5.length));
  const yrStrat = yearlyReturnsFromEquity(dates, eqStrat).filter((x) => x.year >= 2020 && x.year <= 2025); const yrBH = yearlyReturnsFromEquity(dates, eqBH).filter((x) => x.year >= 2020 && x.year <= 2025);
  // Transition timing metrics (all-time within filtered range)
  function fwdCum(i, k) { let prod = 1; for (let t = 1; t <= k && i + t < ret.length; t += 1) prod *= 1 + (ret[i + t] || 0); return prod - 1; }
  function timingHits(k) {
    let onC=0,onH=0, offC=0,offH=0; const st = ffl.state;
    for (let i=1;i<st.length;i+=1){ const prev=st[i-1], cur=st[i]; if (cur===1 && prev!==1){ onC+=1; const r=fwdCum(i,k); if (Number.isFinite(r) && r>0) onH+=1; }
      if (cur===-1 && prev!==-1){ offC+=1; const r=fwdCum(i,k); if (Number.isFinite(r) && r<0) offH+=1; } }
    return { on: onC>0? onH/onC : 0, off: offC>0? offH/offC : 0, counts:{on:onC,off:offC} };
  }
  const t5 = timingHits(5); const t10 = timingHits(10);
  // MDD for strategy
  function maxDrawdown(series){ let peak=series[0]||1, mdd=0; for(const v of series){ if(v>peak) peak=v; const dd=(v/peak)-1; if(dd<mdd) mdd=dd; } return mdd; }
  const mdd = maxDrawdown(eqStrat);
  // Regime proportions and miss metrics (using executed state and 1d forward returns)
  const N = Math.min(executedState.length, baseReturns.length);
  let onDays = 0, offDays = 0, neutralDays = 0;
  let onCapture = 0, onCount = 0;
  let neutralMiss = 0, neutralCount = 0;
  let offMiss = 0, offCount = 0;
  for (let i = 0; i < N - 1; i += 1) {
    const s = executedState[i];
    const fwd = baseReturns[i + 1];
    if (s > 0) { onDays += 1; onCount += 1; if (Number.isFinite(fwd) && fwd > 0) onCapture += 1; }
    else if (s < 0) { offDays += 1; offCount += 1; if (Number.isFinite(fwd) && fwd > 0) offMiss += 1; }
    else { neutralDays += 1; neutralCount += 1; if (Number.isFinite(fwd) && fwd > 0) neutralMiss += 1; }
  }
  const regime = {
    ratios: {
      on: N > 0 ? onDays / N : 0,
      off: N > 0 ? offDays / N : 0,
      neutral: N > 0 ? neutralDays / N : 0,
    },
    onCapture: onCount > 0 ? onCapture / onCount : 0,
    misses: {
      neutralOn: neutralCount > 0 ? neutralMiss / neutralCount : 0,
      offOn: offCount > 0 ? offMiss / offCount : 0,
    },
  };
  const out = { window: WINDOW, range: { start: RANGE_START, end: dates[dates.length - 1] }, equity: { strategy: eqStrat[eqStrat.length - 1], benchmark: eqBH[eqBH.length - 1] }, hitRate: { d1: hr1, d5: hr5 }, yearly: { strategy: Object.fromEntries(yrStrat.map((x) => [x.year, x.ret])), benchmark: Object.fromEntries(yrBH.map((x) => [x.year, x.ret])) }, drift: ffl.drift, regime, timing: { k5: t5, k10: t10 }, metrics: { mdd } };
  console.log(JSON.stringify(out, null, 2));
}

if (require.main === module) { try { main(); } catch (err) { console.error('[backtest_ffl] failed:', err?.stack || err); process.exit(1); } }
