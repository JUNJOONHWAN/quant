#!/usr/bin/env node
/*
 * Random-search tuner for FFL+EXP fusion (Classic + Flux + Stability)
 * - Optimizes expTune parameters without changing the underlying formula.
 * - Multi-segment objective (2017–2019, 2020–2021, 2022, 2023–2025) to be robust across regimes.
 * - Time-bounded random search to avoid long runs.
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WINDOW = 30;
const SYMBOLS = ['QQQ', 'IWM', 'SPY', 'TLT', 'GLD', 'BTC-USD'];
const FULL_START = '2017-01-01';
const FULL_END = '2025-10-30';
const SEGMENTS = [
  ['2017-01-01', '2019-12-31'],
  ['2020-01-01', '2021-12-31'],
  ['2022-01-01', '2022-12-31'],
  ['2023-01-01', '2025-10-30'],
];

function dateToNum(s) { return Number(s.replace(/-/g, '')); }

function loadHistorical() {
  const p = path.join(__dirname, '..', 'static_site', 'data', 'historical_prices.json');
  if (!fs.existsSync(p)) throw new Error('Missing static_site/data/historical_prices.json');
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function createMM() { return require(path.join(__dirname, '..', 'static_site', 'assets', 'metrics.js')); }

function alignFromHistorical(MM, raw, symbols) {
  const list = raw.assets.filter((a) => symbols.includes(a.symbol))
    .map((a) => ({ symbol: a.symbol, category: a.category, dates: a.dates, prices: a.prices }));
  return MM.alignSeries(list);
}

function sliceAligned(aligned, start, end) {
  const s = dateToNum(start); const e = dateToNum(end);
  const dates = aligned.dates.filter((d) => { const n = dateToNum(d); return n >= s && n <= e; });
  const idx = new Map(aligned.dates.map((d, i) => [d, i]));
  const prices = {}; Object.entries(aligned.prices).forEach(([sym, arr]) => { prices[sym] = dates.map((d) => arr[idx.get(d)]); });
  return { dates, prices, categories: aligned.categories };
}

function createAppContext(MM) {
  const context = { window: { location: { hostname: 'localhost' }, MarketMetrics: MM }, document: { readyState: 'loading', addEventListener: () => {} }, console, setTimeout, clearTimeout };
  vm.createContext(context);
  const appPath = path.join(__dirname, '..', 'static_site', 'assets', 'app.js');
  const code = fs.readFileSync(appPath, 'utf8');
  vm.runInContext(code, context, { filename: 'app.js' });
  return context;
}

function buildMetrics(ctx, MM, aligned) {
  const returns = MM.computeReturns(aligned);
  vm.runInContext(`state.window=${WINDOW};`, ctx);
  ctx.PRICES = returns.priceSeries; vm.runInContext(`state.priceSeries = PRICES;`, ctx);
  const metrics = ctx.computeWindowMetrics(WINDOW, returns, aligned);
  return { returns, metrics };
}

function buildSeriesFFLExp(ctx, metrics, pricesQQQ, expTune) {
  ctx.CAND_EXP = expTune;
  vm.runInContext(`state.riskMode='ffl_exp'; if (CAND_EXP) { RISK_CFG_FFL.expTune = Object.assign({}, RISK_CFG_FFL.expTune, CAND_EXP); }`, ctx);
  const series = ctx.computeRiskSeriesFFL(metrics, metrics.records);
  const dates = series.dates;
  const windowOffset = WINDOW - 1;
  const baseRet = [];
  for (let i = 0; i < dates.length; i += 1) {
    const idx = windowOffset + i; const prev = pricesQQQ[idx - 1]; const cur = pricesQQQ[idx];
    baseRet.push(Number.isFinite(prev) && Number.isFinite(cur) && prev !== 0 ? (cur/prev - 1) : 0);
  }
  const executed = Array.isArray(series.executedState) && series.executedState.length === series.state.length
    ? series.executedState
    : series.state.map((v,i)=> i===0 ? 0 : series.state[i-1]||0);
  return { dates, state: series.state, executed, baseRet };
}

function statsFromReturns(ret) {
  let e = 1; const eq = [];
  for (const r of ret) { e *= 1 + (Number.isFinite(r) ? r : 0); eq.push(e); }
  const total = e - 1;
  const days = ret.length; const years = Math.max(days/252, 1e-9);
  const cagr = Math.pow(1 + total, 1/years) - 1;
  let peak = eq[0] || 1; let mdd = 0;
  for (const v of eq) { if (v > peak) peak = v; const dd = (peak - v) / peak; if (dd > mdd) mdd = dd; }
  return { total, cagr, mdd };
}

function hitRateOn(state, fwd) { let w=0,c=0; for (let i=0;i<Math.min(state.length-1, fwd.length); i+=1){ const s=state[i]; if (s<=0) continue; const r=fwd[i]; if (Number.isFinite(r)) { if (r>0) w+=1; c+=1; } } return c>0? w/c : 0; }

function scoreSegments(segmentStats) {
  // Geometric mean of (1+total) across segments for robustness
  let geo = 1; let k = 0; let mddSum = 0; let hrSum = 0;
  for (const s of segmentStats) { geo *= Math.max(1 + s.total, 1e-6); k += 1; mddSum += s.mdd; hrSum += s.hr; }
  const geoRet = Math.pow(geo, 1/Math.max(k,1)) - 1;
  const avgMDD = mddSum / Math.max(k,1);
  const avgHR = hrSum / Math.max(k,1);
  // objective: reward returns+hit, penalize drawdown
  return Math.log(1 + Math.max(geoRet, -0.8)) + 0.4*(avgHR - 0.5) - 0.4*avgMDD;
}

function drawCandidate() {
  const macroOn = 0.44 + Math.random()*0.08; // [0.44,0.52]
  const macroOff = 0.28 + Math.random()*0.10; // [0.28,0.38]
  const qOn = 0.65 + Math.random()*0.15; // [0.65,0.80]
  const qOff = 0.10 + Math.random()*0.15; // [0.10,0.25]
  const aS = 0.10 + Math.random()*0.25; // [0.10,0.35]
  const aJ = 0.05 + Math.random()*0.10; // [0.05,0.15]
  const bS = 0.10 + Math.random()*0.20; // [0.10,0.30]
  const bJ = 0.05 + Math.random()*0.10; // [0.05,0.15]
  const gSPos = 0.03 + Math.random()*0.06; // [0.03,0.09]
  const gSNeg = 0.03 + Math.random()*0.06; // [0.03,0.09]
  const confirmOn = Math.random() < 0.5 ? 1 : 2;
  const r = Math.random(); const confirmOff = r < 0.33 ? 1 : r < 0.66 ? 2 : 3;
  const hazardHigh = 0.48 + Math.random()*0.06; // [0.48,0.54]
  const hazardDrop = 0.02 + Math.random()*0.03; // [0.02,0.05]
  const hazardLookback = 3 + Math.floor(Math.random()*5); // 3..7
  // validity constraints
  if (macroOff >= macroOn - 0.04) return drawCandidate();
  if (qOff >= qOn - 0.08) return drawCandidate();
  return { macroOn: +macroOn.toFixed(3), macroOff: +macroOff.toFixed(3), qOn: +qOn.toFixed(3), qOff: +qOff.toFixed(3), aS:+aS.toFixed(3), aJ:+aJ.toFixed(3), bS:+bS.toFixed(3), bJ:+bJ.toFixed(3), gSPos:+gSPos.toFixed(3), gSNeg:+gSNeg.toFixed(3), confirmOn, confirmOff, hazardHigh:+hazardHigh.toFixed(3), hazardDrop:+hazardDrop.toFixed(3), hazardLookback };
}

async function main() {
  const MM = createMM();
  const raw = loadHistorical();
  const alignedFull = alignFromHistorical(MM, raw, SYMBOLS);
  const ctx = createAppContext(MM);

  // Pre-build segment metrics
  const segData = SEGMENTS.map(([s,e]) => {
    const aligned = sliceAligned(alignedFull, s, e);
    const { returns, metrics } = buildMetrics(ctx, MM, aligned);
    return { metrics, pricesQQQ: returns.priceSeries['QQQ'], label: `${s}~${e}` };
  });

  const startTime = Date.now();
  const timeBudgetMs = Number(process.env.TUNE_TIME_MS || 60000); // default 60s
  const maxIters = Number(process.env.TUNE_ITERS || 120);
  let best = null; let bestScore = -Infinity; let it = 0;

  while (it < maxIters && (Date.now() - startTime) < timeBudgetMs) {
    it += 1;
    const cand = drawCandidate();
    const segStats = [];
    let flipsTotal = 0; let daysTotal = 0;
    for (const sd of segData) {
      const series = buildSeriesFFLExp(ctx, sd.metrics, sd.pricesQQQ, cand);
      const fwd1 = series.baseRet.slice(1);
      const hr = hitRateOn(series.state, fwd1);
      const execRet = series.executed.map((reg,i)=> reg>0? Math.max(-0.99, 3*series.baseRet[i]) : reg<0 ? 0 : series.baseRet[i]);
      const st = statsFromReturns(execRet);
      segStats.push({ total: st.total, cagr: st.cagr, mdd: st.mdd, hr });
      for (let i=1;i<series.state.length;i+=1){ if(series.state[i]!==series.state[i-1]) flipsTotal += 1; }
      daysTotal += Math.max(0, series.state.length);
    }
    const segScore = scoreSegments(segStats);
    const flipRate = flipsTotal / Math.max(1, daysTotal);
    const score = segScore - 0.15*flipRate; // churn penalty
    if (score > bestScore) { bestScore = score; best = { params: cand, segStats, score, it }; }
  }

  console.log('Best (random search):', best);
}

main().catch((e)=>{ console.error(e); process.exit(1); });

