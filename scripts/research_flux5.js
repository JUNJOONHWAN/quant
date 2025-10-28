#!/usr/bin/env node
// Research runner for Classic+Flux5 (10-pair Fick) from scratch.
// Reads static_site/data/historical_prices.json and prints summary & 2022 slice.

const fs = require('fs');
const path = require('path');
const MM = require('../static_site/assets/metrics');
const CF5 = require('../static_site/assets/engine/classic_flux5');

const DATA_PATH = path.join(__dirname, '..', 'static_site', 'data', 'historical_prices.json');
const WINDOW = Number(process.env.BACKTEST_WINDOW || 30);

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
  stocks: ['IWM', 'SPY'],
  safe: ['TLT', 'GLD'],
  risk: ['IWM', 'SPY', 'BTC-USD'],
};

function loadHistorical() {
  const raw = fs.readFileSync(DATA_PATH, 'utf8');
  const json = JSON.parse(raw);
  const bySymbol = new Map(json.assets.map((a) => [a.symbol, a]));
  const list = ASSETS.map(({symbol, category}) => ({ symbol, category, dates: bySymbol.get(symbol).dates, prices: bySymbol.get(symbol).prices }));
  const aligned = MM.alignSeries(list);
  const returns = MM.computeReturns(aligned);
  return { aligned, returns };
}

function equityFromState(dates, returns, baseSymbol, state, window) {
  const prices = returns.normalizedPrices; // not used
  const priceSeries = returns.priceSeries[baseSymbol];
  const offset = Math.max(1, window - 1);
  let s = 1; let b = 1;
  const eqS = []; const eqB = [];
  for (let i = 0; i < dates.length; i += 1) {
    const pi = offset + i; const pr = priceSeries[pi] / priceSeries[pi - 1] - 1;
    const rr = state[i] > 0 ? Math.max(-0.99, 3 * pr) : 0; // Neutral도 현금 취급(방어적 연구 설정)
    s *= (1 + rr); b *= (1 + pr);
    eqS.push(Number(s.toFixed(6))); eqB.push(Number(b.toFixed(6)));
  }
  return { eqS, eqB };
}

function sliceYear(dates, series, year) {
  const mask = dates.map((d) => d.startsWith(String(year)));
  const first = mask.indexOf(true); const last = mask.lastIndexOf(true);
  if (first < 1 || last < 1 || last <= first) return null;
  return (series[last] / series[first - 1]) - 1;
}

function runOnce(params) {
  const { aligned, returns } = loadHistorical();
  const symbols = Object.keys(aligned.prices);
  const categories = aligned.categories;
  const prices = returns.priceSeries;
  const pairGroups = { risk: SIGNAL.risk, safe: SIGNAL.safe, stocks: SIGNAL.stocks };
  const out = CF5.computeClassicFlux5({ aligned, returns, window: WINDOW, symbols, categories, prices, pairGroups, params });
  const { eqS, eqB } = equityFromState(out.dates, returns, 'QQQ', out.executedState, WINDOW);
  const years = [2020, 2021, 2022, 2023, 2024, 2025];
  const yearly = {
    strategy: Object.fromEntries(years.map((y) => [y, sliceYear(out.dates, eqS, y)])),
    benchmark: Object.fromEntries(years.map((y) => [y, sliceYear(out.dates, eqB, y)])),
  };
  return {
    window: WINDOW,
    range: { start: out.dates[0], end: out.dates[out.dates.length - 1] },
    equity: { strategy: eqS[eqS.length - 1], benchmark: eqB[eqB.length - 1] },
    yearly,
    diagnostics: {
      latest: {
        date: out.dates[out.dates.length - 1], mm: out.mm[out.mm.length - 1], flux5: out.flux5[out.flux5.length - 1], pcon: out.pcon[out.pcon.length - 1], score: out.score[out.score.length - 1],
      },
      drift: out.diagnostics.drift,
    },
  };
}

function gridSearch() {
  const grid = [];
  const vOns = [0.05, 0.10];
  const vOffs = [-0.05, 0.00];
  const pconOns = [0.55, 0.60, 0.65];
  const pconOffs = [0.40, 0.45, 0.50];
  const mmHis = [0.90, 0.92, 0.88];
  const downAlls = [0.60, 0.65, 0.70];
  const driftDays = [3, 5];
  for (const vOn of vOns) for (const vOff of vOffs)
  for (const pconOn of pconOns) for (const pconOff of pconOffs)
  for (const mmHi of mmHis) for (const downAll of downAlls)
  for (const dMin of driftDays) {
    grid.push({ vOn, vOff, pconOn, pconOff, mmHi, downAll, driftMinDays: dMin, corrConeDays: 5, offMinDays: 3 });
  }
  let best = null; let bestScore = -Infinity; let bestRes = null; let tried = 0;
  for (const params of grid) {
    const r = runOnce(params); tried++;
    const y = r.yearly.strategy;
    const f2022 = y[2022] ?? -Infinity; const f2023 = y[2023] ?? -Infinity; const f2024 = y[2024] ?? -Infinity;
    // Objective: 2022 ≥ -0.05, 2023/2024 positive, maximize (2023+2024) and minimize |2022|
    const ok = (f2022 >= -0.05);
    const score = (ok ? 1000 : 0) + (f2023 || 0) + (f2024 || 0) - Math.max(0, -f2022);
    if (score > bestScore) { bestScore = score; best = params; bestRes = r; }
  }
  return { best, bestRes, tried };
}

function main() {
  if (process.env.GRID === '1') {
    const { best, bestRes, tried } = gridSearch();
    console.log(JSON.stringify({ tried, best, result: bestRes }, null, 2));
  } else {
    const p = {};
    for (const k of ['jOn','jOff','pconOn','pconOff','mmOff','mmHi','downAll','corrConeDays','driftMinDays','driftCool','vOn','vOff','offMinDays']) {
      if (process.env[k] != null) p[k] = Number(process.env[k]);
    }
    const r = runOnce(p);
    console.log(JSON.stringify(r, null, 2));
  }
}

if (require.main === module) { try { main(); } catch (e) { console.error('[research_flux5] failed:', e?.stack || e); process.exit(1); } }
