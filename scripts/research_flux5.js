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
    const rr = state[i] > 0 ? Math.max(-0.99, 3 * pr) : state[i] < 0 ? 0 : pr; // 3x on On, cash on Off, 1x on Neutral
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

function main() {
  const { aligned, returns } = loadHistorical();
  const symbols = Object.keys(aligned.prices);
  const categories = aligned.categories;
  const prices = returns.priceSeries;
  const pairGroups = { risk: SIGNAL.risk, safe: SIGNAL.safe, stocks: SIGNAL.stocks };
  const out = CF5.computeClassicFlux5({ aligned, returns, window: WINDOW, symbols, categories, prices, pairGroups });
  const { eqS, eqB } = equityFromState(out.dates, returns, 'QQQ', out.executedState, WINDOW);
  const years = [2020, 2021, 2022, 2023, 2024, 2025];
  const yearly = {
    strategy: Object.fromEntries(years.map((y) => [y, sliceYear(out.dates, eqS, y)])),
    benchmark: Object.fromEntries(years.map((y) => [y, sliceYear(out.dates, eqB, y)])),
  };
  const result = {
    window: WINDOW,
    range: { start: out.dates[0], end: out.dates[out.dates.length - 1] },
    equity: { strategy: eqS[eqS.length - 1], benchmark: eqB[eqB.length - 1] },
    yearly,
    diagnostics: {
      latest: {
        date: out.dates[out.dates.length - 1], mm: out.mm[out.mm.length - 1], flux5: out.flux5[out.flux5.length - 1], pcon: out.pcon[out.pcon.length - 1], score: out.score[out.score.length - 1],
      },
    },
  };
  console.log(JSON.stringify(result, null, 2));
}

if (require.main === module) {
  try { main(); } catch (e) { console.error('[research_flux5] failed:', e?.stack || e); process.exit(1); }
}

