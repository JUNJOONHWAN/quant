#!/usr/bin/env node
/*
 * Compare FFL+STAB profiles: current app.js vs tuned parameters (JSON file)
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ARGS = process.argv.slice(2);
const START = ARGS[0] || '2017-01-01';
const END = ARGS[1] || '2025-10-30';
const PROFILE_PATH = ARGS[2] || path.join(__dirname, 'tune_stability_from_txt.json');
const WINDOW = 30;
const SYMBOLS = ['QQQ', 'IWM', 'SPY', 'TLT', 'GLD', 'BTC-USD'];

function dateToNum(s) { return Number(s.replace(/-/g, '')); }
function loadHistorical() { return JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'static_site', 'data', 'historical_prices.json'), 'utf8')); }
function alignFromHistorical(MM, raw, symbols) {
  const seriesList = raw.assets.filter((a) => symbols.includes(a.symbol)).map((a) => ({ symbol: a.symbol, category: a.category, dates: a.dates, prices: a.prices }));
  return MM.alignSeries(seriesList);
}
function sliceAligned(aligned, startDate, endDate) {
  const s = dateToNum(startDate); const e = dateToNum(endDate);
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

function createAppContext(MM) {
  const context = { window: { location: { hostname: 'localhost' }, MarketMetrics: MM }, document: { readyState: 'loading', addEventListener: () => {} }, console, setTimeout, clearTimeout };
  vm.createContext(context);
  const code = fs.readFileSync(path.join(__dirname, '..', 'static_site', 'assets', 'app.js'), 'utf8');
  vm.runInContext(code, context, { filename: 'app.js' });
  return context;
}

function buildMetrics(ctx, MM, aligned) {
  const returns = MM.computeReturns(aligned);
  ctx.PRICES = returns.priceSeries;
  vm.runInContext(`state.window=${WINDOW}; state.priceSeries = PRICES;`, ctx);
  const metrics = ctx.computeWindowMetrics(WINDOW, returns, aligned);
  return { returns, metrics };
}

function backtestSeries(series, pricesQQQ) {
  const windowOffset = Math.max(1, WINDOW - 1);
  const baseRet = [];
  for (let i = 0; i < series.dates.length; i += 1) {
    const idx = windowOffset + i;
    const prev = pricesQQQ[idx - 1]; const cur = pricesQQQ[idx];
    baseRet.push(Number.isFinite(prev) && Number.isFinite(cur) && prev !== 0 ? (cur / prev - 1) : 0);
  }
  const executed = series.executedState.map((v, i) => (i === 0 ? 0 : series.executedState[i]));
  const neutralWeight = 0.33;
  const strat = executed.map((reg, i) => {
    if (reg > 0) return leveragedReturn(baseRet[i], 3, 1);
    if (reg < 0) return 0;
    return leveragedReturn(baseRet[i], 3, neutralWeight);
  });
  const eq = equityFromReturns(strat);
  const total = eq[eq.length - 1] - 1;
  const cagr = annualize(series.dates.length, total);
  const mdd = maxDrawdown(eq);
  const fwd1 = baseRet.slice(1); const sig1 = series.state.slice(0, series.state.length - 1);
  const hr = hitRate(sig1, fwd1, true);
  return { total, cagr, mdd, hr };
}

async function main() {
  const MM = require(path.join(__dirname, '..', 'static_site', 'assets', 'metrics.js'));
  const raw = loadHistorical();
  const alignedFull = alignFromHistorical(MM, raw, SYMBOLS);
  const aligned = sliceAligned(alignedFull, START, END);
  const ctxA = createAppContext(MM);
  const ctxB = createAppContext(MM);
  const tuned = fs.existsSync(PROFILE_PATH) ? JSON.parse(fs.readFileSync(PROFILE_PATH, 'utf8')) : null;
  const { returns, metrics } = buildMetrics(ctxA, MM, aligned);
  const pricesQQQ = returns.priceSeries['QQQ'];

  // A) current app.js
  vm.runInContext(`state.riskMode='ffl_stab';`, ctxA);
  const a = ctxA.computeRiskSeriesFFL(metrics, metrics.records);
  const resA = backtestSeries(a, pricesQQQ);

  // B) tuned override
  ctxB.PRICES = returns.priceSeries;
  vm.runInContext(`state.window=${WINDOW}; state.priceSeries = PRICES;`, ctxB);
  if (tuned && tuned.stabTune) {
    vm.runInContext(`RISK_CFG_FFL.stabTune = ${JSON.stringify(tuned.stabTune)}; state.riskMode='ffl_stab';`, ctxB);
  } else {
    vm.runInContext(`state.riskMode='ffl_stab';`, ctxB);
  }
  const b = ctxB.computeRiskSeriesFFL(metrics, metrics.records);
  const resB = backtestSeries(b, pricesQQQ);

  function pct(x){ return `${(x*100).toFixed(1)}%`; }
  console.log(`Compare FFL+STAB profiles | Window ${WINDOW} | ${START} ~ ${END}`);
  console.log(`Current: Total ${pct(resA.total)} | CAGR ${pct(resA.cagr)} | MDD ${pct(resA.mdd)} | Hit ${pct(resA.hr)}`);
  console.log(`Tuned  : Total ${pct(resB.total)} | CAGR ${pct(resB.cagr)} | MDD ${pct(resB.mdd)} | Hit ${pct(resB.hr)}`);
}

main().catch((e)=>{ console.error(e); process.exit(1); });
