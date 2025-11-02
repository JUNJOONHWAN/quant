#!/usr/bin/env node
/*
 * FFL+STAB Auto Tuner
 * - Estimates Stability cycle length via peak detection
 * - Samples stabTune around that cycle (random search)
 * - Scores across time splits: 2017-2019, 2020-2022, 2023-2025
 * - Objective: blended CAGR + hit rate − MDD penalty − churn penalty
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const SEED = Number(process.env.SEED || 42);
let _rand = mulberry32(SEED);
function mulberry32(a) { return function() { let t = a += 0x6D2B79F5; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function rand() { return _rand(); }
function randi(min, max) { return Math.floor(min + rand() * (max - min + 1)); }
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

const TUNE_TIME_MS = Number(process.env.TUNE_TIME_MS || 60000);
const TUNE_ITERS = Number(process.env.TUNE_ITERS || 200);

const SYMBOLS = ['QQQ', 'IWM', 'SPY', 'TLT', 'GLD', 'BTC-USD'];
const START_FULL = '2017-01-01';
const END_FULL = '2025-10-30';

function dateToNum(s) { return Number(s.replace(/-/g, '')); }

function loadJSON(p) { return JSON.parse(fs.readFileSync(p, 'utf8')); }

function createAppContext(MM) {
  const context = {
    window: { location: { hostname: 'localhost' }, MarketMetrics: MM },
    document: { readyState: 'loading', addEventListener: () => {} },
    console, setTimeout, clearTimeout,
  };
  vm.createContext(context);
  const appPath = path.join(__dirname, '..', 'static_site', 'assets', 'app.js');
  const code = fs.readFileSync(appPath, 'utf8');
  vm.runInContext(code, context, { filename: 'app.js' });
  return context;
}

function alignFromHistorical(MM, raw, symbols) {
  const seriesList = raw.assets
    .filter((a) => symbols.includes(a.symbol))
    .map((a) => ({ symbol: a.symbol, category: a.category, dates: a.dates, prices: a.prices }));
  return MM.alignSeries(seriesList);
}

function sliceAligned(aligned, startDate, endDate) {
  const s = dateToNum(startDate);
  const e = dateToNum(endDate);
  const dates = aligned.dates.filter((d) => { const n = dateToNum(d); return n >= s && n <= e; });
  const idx = new Map(aligned.dates.map((d, i) => [d, i]));
  const prices = {}; Object.entries(aligned.prices).forEach(([sym, arr]) => { prices[sym] = dates.map((d) => arr[idx.get(d)]); });
  return { dates, prices, categories: aligned.categories };
}

function leveragedReturn(r, lev = 3, weight = 1) {
  if (!Number.isFinite(r)) return 0;
  const effWeight = Number.isFinite(weight) ? weight : 1;
  const x = lev * r * effWeight;
  return Math.max(-0.99, x);
}
function equityFromReturns(ret) { let e = 1; return ret.map((r) => { e *= 1 + (Number.isFinite(r) ? r : 0); return Number(e.toFixed(8)); }); }
function maxDrawdown(eq) { let p = eq[0] || 1; let m = 0; for (const v of eq) { if (v > p) p = v; const dd = (p - v) / p; if (dd > m) m = dd; } return m; }
function annualize(days, total) { const y = Math.max(days / 252, 1e-9); return Math.pow(1 + total, 1 / y) - 1; }
function hitRate(sig, fwd, skipNeutral = true) { let w = 0; let c = 0; for (let i = 0; i < Math.min(sig.length, fwd.length); i += 1) { const s = sig[i]; if (skipNeutral && s === 0) continue; const r = fwd[i]; if (Number.isFinite(r)) { if ((s > 0 && r > 0) || (s < 0 && r < 0)) w += 1; c += 1; } } return c > 0 ? (w / c) : 0; }

function computeBaseReturns(pricesQQQ, windowSize) {
  const windowOffset = Math.max(1, windowSize - 1);
  const baseRet = [];
  for (let i = 0; i < pricesQQQ.length; i += 1) {
    const idx = windowOffset + i; const prev = pricesQQQ[idx - 1]; const cur = pricesQQQ[idx];
    baseRet.push(Number.isFinite(prev) && Number.isFinite(cur) && prev !== 0 ? (cur / prev - 1) : 0);
  }
  return baseRet;
}

function splitWindows() {
  return [
    ['2017-01-01', '2019-12-31'],
    ['2020-01-01', '2022-12-31'],
    ['2023-01-01', '2025-10-30'],
  ];
}

function detectCycleDays(stabilitySeries) {
  // simple peak/trough detection on EMA(10) of stability
  const ema = (arr, n) => { const a = []; const k = 2 / (n + 1); let prev = undefined; for (let i=0;i<arr.length;i++){ const v = Number(arr[i]); if (!Number.isFinite(v)) { a.push(prev==undefined?0:prev); continue; } prev = (prev==undefined)? v : prev + k*(v - prev); a.push(prev); } return a; };
  const s = stabilitySeries.filter(Number.isFinite);
  if (s.length < 40) return 42; // default ~2 months
  const se = ema(s, 10);
  const peaks = [];
  const troughs = [];
  for (let i = 2; i < se.length - 2; i += 1) {
    const v = se[i]; if (!Number.isFinite(v)) continue;
    if (se[i-2] < se[i-1] && se[i-1] < v && v > se[i+1] && se[i+1] > se[i+2]) peaks.push(i);
    if (se[i-2] > se[i-1] && se[i-1] > v && v < se[i+1] && se[i+1] < se[i+2]) troughs.push(i);
  }
  const pivots = peaks.concat(troughs).sort((a,b)=>a-b);
  if (pivots.length < 4) return 42;
  const gaps = []; for (let i=1;i<pivots.length;i++) gaps.push(pivots[i]-pivots[i-1]);
  const medianGap = gaps.sort((a,b)=>a-b)[Math.floor(gaps.length/2)];
  const halfCycle = Math.max(15, Math.min(80, medianGap));
  return Math.round(halfCycle * 2); // peak-to-peak
}

function makeParamsFromCycle(cycleDays) {
  const fast = clamp(Math.round(0.30 * cycleDays) + randi(-3, 3), 7, 42);
  const slow = clamp(Math.round(0.90 * cycleDays) + randi(-5, 5), Math.max(2*fast, 30), 126);
  const zWin = clamp(Math.round(2.0 * slow) + randi(-10, 10), 63, 210);
  const zUp = 1.4 + rand() * 0.8; // 1.4~2.2
  const zDown = 1.4 + rand() * 0.8;
  const slopeMin = 0.010 + rand() * 0.010; // 0.010~0.020
  const lagUp = clamp(Math.round(0.15 * fast) + randi(0, 2), 2, 8);
  const lagDown = clamp(Math.round(0.20 * fast) + randi(0, 3), 3, 10);
  const leadOnWindow = clamp(Math.round(0.25 * fast) + randi(0, 2), 3, 10);
  const downGrace = clamp(Math.round(0.25 * fast) + randi(0, 2), 3, 10);
  const hazardWindow = clamp(Math.round(0.35 * fast) + randi(0, 3), 4, 14);
  const onFluxEase = 0.015 + rand() * 0.03; // 0.015~0.045
  const offFluxTighten = 0.02 + rand() * 0.04; // 0.02~0.06
  const confirmOnMin = randi(1, 3);
  const confirmOffMin = randi(1, 2);
  const onOverrideMargin = 0.005 + rand() * 0.02; // 0.005~0.025
  return { fast, slow, zWin, zUp, zDown, slopeMin, lagUp, lagDown, leadOnWindow, downGrace, hazardWindow, onFluxEase, offFluxTighten, confirmOnMin, confirmOffMin, onOverrideMargin, neutralLo:0.30, neutralHi:0.40 };
}

function scoreSet(ctx, MM, aligned, startDate, endDate, stabTune) {
  const alignedSlice = sliceAligned(aligned, startDate, endDate);
  const returns = MM.computeReturns(alignedSlice);
  ctx.PRICES = returns.priceSeries;
  vm.runInContext(`state.window=30; state.priceSeries = PRICES;`, ctx);
  const metrics = ctx.computeWindowMetrics(30, returns, alignedSlice);
  // inject candidate
  vm.runInContext(`RISK_CFG_FFL.stabTune = ${JSON.stringify(stabTune)};`, ctx);
  vm.runInContext(`state.riskMode='ffl_stab';`, ctx);
  const series = ctx.computeRiskSeriesFFL(metrics, metrics.records);
  if (!series || !Array.isArray(series.state)) return null;
  const pricesQQQ = returns.priceSeries['QQQ'];
  const baseRet = computeBaseReturns(pricesQQQ, 30);
  const executed = series.executedState.map((v,i)=> (i===0?0:series.executedState[i]));
  const neutralWeight = 0.33;
  const strat = executed.map((reg,i)=>{
    if (reg>0) return leveragedReturn(baseRet[i],3,1);
    if (reg<0) return 0;
    return leveragedReturn(baseRet[i],3,neutralWeight);
  });
  const eq = equityFromReturns(strat);
  const total = eq[eq.length-1]-1;
  const cagr = annualize(series.dates.length, total);
  const mdd = maxDrawdown(eq);
  const fwd1 = baseRet.slice(1); const sig1 = series.state.slice(0, series.state.length-1);
  const hr = hitRate(sig1, fwd1, true);
  // churn penalty: number of transitions per year
  let transitions = 0; for (let i=1;i<series.state.length;i++){ if (series.state[i]!==series.state[i-1]) transitions++; }
  const years = Math.max(series.dates.length/252, 0.1);
  const churnPerYear = transitions / years;
  return { total, cagr, mdd, hr, churnPerYear };
}

function blendedScore(res) {
  if (!res) return -1e9;
  // Higher is better. Balance CAGR vs MDD, add HR, penalize churn.
  const cagr = res.cagr || 0;
  const mdd = res.mdd || 1;
  const hr = res.hr || 0.45;
  const churn = res.churnPerYear || 20;
  const mddPenalty = Math.max(0, mdd - 0.45); // penalize above 45%
  return (3.0*cagr) + (0.5*hr) - (1.2*mddPenalty) - (0.01*churn);
}

async function main() {
  const MM = require(path.join(__dirname, '..', 'static_site', 'assets', 'metrics.js'));
  const raw = loadJSON(path.join(__dirname, '..', 'static_site', 'data', 'historical_prices.json'));
  const aligned = alignFromHistorical(MM, raw, SYMBOLS);
  const ctx = createAppContext(MM);
  // build full metrics to extract stability series for cycle detection
  const alignedFull = sliceAligned(aligned, START_FULL, END_FULL);
  const returnsFull = MM.computeReturns(alignedFull);
  ctx.PRICES = returnsFull.priceSeries;
  vm.runInContext(`state.window=30; state.priceSeries = PRICES;`, ctx);
  const metricsFull = ctx.computeWindowMetrics(30, returnsFull, alignedFull);
  const stabilitySeries = metricsFull.records.map(r=> Number.isFinite(r.stability)? r.stability : null);
  const cycleDays = detectCycleDays(stabilitySeries);
  console.log(`[info] estimated stability cycle ≈ ${cycleDays} trading days`);

  const splits = splitWindows();
  let best = null; let bestScore = -1e9; let iters = 0;
  const deadline = Date.now() + TUNE_TIME_MS;
  while (iters < TUNE_ITERS && Date.now() < deadline) {
    iters += 1;
    const params = makeParamsFromCycle(cycleDays);
    // evaluate across splits and average
    let totalScore = 0; let valid = true; const parts=[];
    for (const [s,e] of splits) {
      const res = scoreSet(ctx, MM, aligned, s, e, params);
      if (!res) { valid=false; break; }
      parts.push(res);
      totalScore += blendedScore(res);
    }
    if (!valid) continue;
    const avg = totalScore / parts.length;
    if (avg > bestScore) { bestScore = avg; best = { params, parts }; }
  }

  if (!best) { console.log('[warn] no result'); process.exit(1); }
  console.log('[best] score', bestScore.toFixed(4));
  console.log('[best] params', JSON.stringify(best.params, null, 2));
  best.parts.forEach((p,idx)=>{
    console.log(`[split${idx+1}] CAGR ${(p.cagr*100).toFixed(2)}% | MDD ${(p.mdd*100).toFixed(1)}% | Hit ${(p.hr*100).toFixed(1)}% | Churn/y ${p.churnPerYear.toFixed(1)}`);
  });

  // write suggestion file
  const outPath = path.join(__dirname, 'tune_ffl_stab_result.json');
  fs.writeFileSync(outPath, JSON.stringify({ cycleDays, bestScore, params: best.params }, null, 2));
  console.log(`[write] ${outPath}`);
}

main().catch((e)=>{ console.error(e); process.exit(1); });
