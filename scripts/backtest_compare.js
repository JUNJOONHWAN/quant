#!/usr/bin/env node
/*
 * Compare modes: FFL, FFL+EXP, Enhanced, Classic, and QQQ(BH)
 * Range: 2020-01-01 ~ 2025-10-30
 * Window: 30 days
 * Data: static_site/data/historical_prices.json (offline snapshot)
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ARGS = process.argv.slice(2);
const START = ARGS[0] || '2020-01-01';
const END = ARGS[1] || '2025-10-30';
const WINDOW = 30;
const SYMBOLS = ['QQQ', 'IWM', 'SPY', 'TLT', 'GLD', 'BTC-USD'];

function dateToNum(s) { return Number(s.replace(/-/g, '')); }

function loadHistorical() {
  const p = path.join(__dirname, '..', 'static_site', 'data', 'historical_prices.json');
  if (!fs.existsSync(p)) {
    throw new Error('historical_prices.json not found. Generate or provide the offline dataset.');
  }
  return JSON.parse(fs.readFileSync(p, 'utf8'));
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
  const prices = {};
  Object.entries(aligned.prices).forEach(([sym, arr]) => { prices[sym] = dates.map((d) => arr[idx.get(d)]); });
  return { dates, prices, categories: aligned.categories };
}

function leveragedReturn(r, lev = 3) { if (!Number.isFinite(r)) return 0; const x = lev * r; return Math.max(-0.99, x); }
function equityFromReturns(ret) { let e = 1; return ret.map((r) => { e *= 1 + (Number.isFinite(r) ? r : 0); return Number(e.toFixed(8)); }); }
function maxDrawdown(eq) { let p = eq[0] || 1; let m = 0; for (const v of eq) { if (v > p) p = v; const dd = (p - v) / p; if (dd > m) m = dd; } return m; }
function annualize(days, total) { const y = Math.max(days / 252, 1e-9); return Math.pow(1 + total, 1 / y) - 1; }
function hitRate(sig, fwd, skipNeutral = true) { let w = 0; let c = 0; for (let i = 0; i < Math.min(sig.length, fwd.length); i += 1) { const s = sig[i]; if (skipNeutral && s === 0) continue; const r = fwd[i]; if (Number.isFinite(r)) { if ((s > 0 && r > 0) || (s < 0 && r < 0)) w += 1; c += 1; } } return c > 0 ? (w / c) : 0; }

function createAppContext(MM) {
  const context = {
    window: { location: { hostname: 'localhost' }, MarketMetrics: MM },
    document: { readyState: 'loading', addEventListener: () => {} },
    console,
    setTimeout,
    clearTimeout,
  };
  vm.createContext(context);
  const appPath = path.join(__dirname, '..', 'static_site', 'assets', 'app.js');
  const code = fs.readFileSync(appPath, 'utf8');
  vm.runInContext(code, context, { filename: 'app.js' });
  return context;
}

function buildMetrics(ctx, MM, aligned) {
  const returns = MM.computeReturns(aligned);
  // expose prices to app state for helpers used in Enhanced/FFL
  vm.runInContext(`state.window=${WINDOW};`, ctx);
  ctx.PRICES = returns.priceSeries;
  vm.runInContext(`state.priceSeries = PRICES;`, ctx);
  const metrics = ctx.computeWindowMetrics(WINDOW, returns, aligned);
  return { returns, metrics };
}

function backtestSeries(series, pricesQQQ) {
  if (!series || !Array.isArray(series.state) || series.state.length === 0) return null;
  const windowOffset = Math.max(1, WINDOW - 1);
  const dates = series.dates;
  const baseRet = [];
  for (let i = 0; i < dates.length; i += 1) {
    const idx = windowOffset + i;
    const prev = pricesQQQ[idx - 1];
    const cur = pricesQQQ[idx];
    baseRet.push(Number.isFinite(prev) && Number.isFinite(cur) && prev !== 0 ? (cur / prev - 1) : 0);
  }
  const executed = Array.isArray(series.executedState) && series.executedState.length === series.state.length
    ? series.executedState
    : series.state.map((v, i) => (i === 0 ? 0 : series.state[i - 1] || 0));
  const stratRet = executed.map((reg, i) => (reg > 0 ? leveragedReturn(baseRet[i], 3) : reg < 0 ? 0 : baseRet[i]));
  const eqStrat = equityFromReturns(stratRet);
  const eqBH = equityFromReturns(baseRet);
  const totalDays = dates.length;
  const total = eqStrat[eqStrat.length - 1] - 1;
  const totalBH = eqBH[eqBH.length - 1] - 1;
  const cagr = annualize(totalDays, total);
  const mdd = maxDrawdown(eqStrat);
  const fwd1 = baseRet.slice(1);
  const sig1 = series.state.slice(0, series.state.length - 1);
  const hr1 = hitRate(sig1, fwd1, true);
  return { total, cagr, mdd, hr1, baseRet, dates };
}

function formatPct(x) { return `${(x * 100).toFixed(1)}%`; }

async function main() {
  const MM = require(path.join(__dirname, '..', 'static_site', 'assets', 'metrics.js'));
  const raw = loadHistorical();
  const alignedFull = alignFromHistorical(MM, raw, SYMBOLS);
  const aligned = sliceAligned(alignedFull, START, END);
  const ctx = createAppContext(MM);
  const { returns, metrics } = buildMetrics(ctx, MM, aligned);
  const pricesQQQ = returns.priceSeries['QQQ'];

  // Classic
  const classic = ctx.computeRiskSeriesClassic(metrics, metrics.records);
  // Enhanced
  const enhanced = ctx.computeRiskSeriesEnhanced(metrics, metrics.records);
  // FFL
  vm.runInContext(`state.riskMode='ffl';`, ctx);
  const ffl = ctx.computeRiskSeriesFFL(metrics, metrics.records);
  // FFL+EXP (our new majority rule)
  vm.runInContext(`state.riskMode='ffl_exp';`, ctx);
  const fflExp = ctx.computeRiskSeriesFFL(metrics, metrics.records);

  // Bench (QQQ BH)
  const windowOffset = WINDOW - 1;
  const baseRet = [];
  for (let i = 0; i < metrics.records.length; i += 1) {
    const idx = windowOffset + i; const prev = pricesQQQ[idx - 1]; const cur = pricesQQQ[idx];
    baseRet.push(Number.isFinite(prev) && Number.isFinite(cur) && prev !== 0 ? (cur / prev - 1) : 0);
  }
  const eqBH = equityFromReturns(baseRet);
  const totalBH = eqBH[eqBH.length - 1] - 1;
  const cagrBH = annualize(metrics.records.length, totalBH);
  const mddBH = maxDrawdown(eqBH);

  function summarize(tag, res) {
    if (!res) { console.log(`${tag}: N/A`); return; }
    console.log(`${tag}: Total ${formatPct(res.total)} | CAGR ${formatPct(res.cagr)} | MDD ${formatPct(res.mdd)} | Hit(1d) ${formatPct(res.hr1)}`);
  }

  console.log(`Compare (QQQ) | Window ${WINDOW} | ${START} ~ ${END}`);
  summarize('Classic', backtestSeries(classic, pricesQQQ));
  summarize('Enhanced', backtestSeries(enhanced, pricesQQQ));
  summarize('FFL', backtestSeries(ffl, pricesQQQ));
  summarize('FFL+EXP', backtestSeries(fflExp, pricesQQQ));
  console.log(`Bench QQQ: Total ${formatPct(totalBH)} | CAGR ${formatPct(cagrBH)} | MDD ${formatPct(mddBH)}`);
}

main().catch((err) => { console.error(err); process.exit(1); });
