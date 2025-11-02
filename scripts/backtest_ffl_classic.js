#!/usr/bin/env node
/*
 * Headless backtest: Original FFL Classic (QQQ) — 2020-01-01 to 2025-10-30
 * Uses the minimal Classic+Flux5 engine wired to MarketMetrics and
 * the bundled offline dataset at static_site/data/historical_prices.json.
 */

const fs = require('fs');
const path = require('path');
const MM = require(path.join(__dirname, '..', 'static_site', 'assets', 'metrics.js'));
const { computeClassicFlux5 } = require(path.join(__dirname, '..', 'static_site', 'assets', 'engine', 'classic_flux5.js'));

function loadHistorical() {
  const p = path.join(__dirname, '..', 'static_site', 'data', 'historical_prices.json');
  const json = JSON.parse(fs.readFileSync(p, 'utf8'));
  return json;
}

function dateToNum(s) { return Number(s.replace(/-/g, '')); }

function alignFromHistorical(raw, symbols) {
  const seriesList = raw.assets
    .filter((a) => symbols.includes(a.symbol))
    .map((a) => ({ symbol: a.symbol, category: a.category, dates: a.dates, prices: a.prices }));
  return MM.alignSeries(seriesList);
}

function sliceAligned(aligned, startDate, endDate) {
  const s = dateToNum(startDate);
  const e = dateToNum(endDate);
  const dates = aligned.dates.filter((d) => {
    const n = dateToNum(d);
    return n >= s && n <= e;
  });
  const map = new Map(aligned.dates.map((d, i) => [d, i]));
  const prices = {};
  Object.entries(aligned.prices).forEach(([sym, arr]) => {
    prices[sym] = dates.map((d) => arr[map.get(d)]);
  });
  return { dates, prices, categories: aligned.categories };
}

function leveragedReturn(r, lev = 3, weight = 1) {
  if (!Number.isFinite(r)) return 0;
  const effWeight = Number.isFinite(weight) ? weight : 1;
  const x = lev * r * effWeight;
  return Math.max(-0.99, x);
}

function equityFromReturns(retArr) {
  let e = 1;
  const eq = [];
  for (let i = 0; i < retArr.length; i += 1) {
    const r = Number.isFinite(retArr[i]) ? retArr[i] : 0;
    e *= (1 + r);
    eq.push(Number(e.toFixed(8)));
  }
  return eq;
}

function maxDrawdown(equity) {
  let peak = equity[0] || 1;
  let mdd = 0;
  for (let i = 0; i < equity.length; i += 1) {
    const v = equity[i];
    if (v > peak) peak = v;
    const dd = (peak - v) / peak;
    if (dd > mdd) mdd = dd;
  }
  return mdd;
}

function annualize(days, totalReturn) {
  const years = Math.max(days / 252, 1e-9);
  return Math.pow(1 + totalReturn, 1 / years) - 1;
}

function hitRate(signals, fwdRet, skipNeutral = true) {
  let w = 0; let c = 0;
  for (let i = 0; i < Math.min(signals.length, fwdRet.length); i += 1) {
    const s = signals[i];
    if (skipNeutral && s === 0) continue;
    const r = fwdRet[i];
    if (Number.isFinite(r)) {
      if ((s > 0 && r > 0) || (s < 0 && r < 0)) w += 1;
      c += 1;
    }
  }
  return c > 0 ? (w / c) : 0;
}

async function main() {
  const START = '2020-01-01';
  const END = '2025-10-30';
  const WINDOW = 30;
  const raw = loadHistorical();
  const symbols = ['QQQ', 'SPY', 'TLT', 'GLD', 'BTC-USD'];
  const alignedFull = alignFromHistorical(raw, symbols);
  const aligned = sliceAligned(alignedFull, START, END);
  const returns = MM.computeReturns(aligned);

  const categories = aligned.categories; // symbol->category
  const pairGroups = {
    risk: ['QQQ', 'SPY', 'BTC-USD'],
    safe: ['TLT', 'GLD'],
    stocks: ['QQQ', 'SPY'],
  };

  const engineOut = computeClassicFlux5({
    aligned,
    returns,
    window: WINDOW,
    symbols,
    categories,
    prices: returns.priceSeries,
    pairGroups,
    params: { jOn: +0.10, jOff: -0.08, pconOn: 0.55, pconOff: 0.40, mmOff: 0.96, mmHi: 0.90 },
  });

  const dates = engineOut.dates;
  const base = returns.priceSeries['QQQ'];
  const priceOffset = WINDOW - 1;
  const baseRet = [];
  for (let i = 0; i < dates.length; i += 1) {
    const idx = priceOffset + i;
    const prev = base[idx - 1];
    const cur = base[idx];
    baseRet.push(Number.isFinite(prev) && Number.isFinite(cur) && prev !== 0 ? (cur / prev - 1) : 0);
  }

  const executed = engineOut.executedState;
  const neutralWeight = 0.33;
  const stratRet = executed.map((reg, i) => {
    if (reg > 0) return leveragedReturn(baseRet[i], 3, 1);
    if (reg < 0) return 0;
    return leveragedReturn(baseRet[i], 3, neutralWeight);
  });

  const eqStrat = equityFromReturns(stratRet);
  const eqBH = equityFromReturns(baseRet);
  const totalDays = dates.length;
  const totalStrat = eqStrat[eqStrat.length - 1] - 1;
  const totalBH = eqBH[eqBH.length - 1] - 1;
  const cagrStrat = annualize(totalDays, totalStrat);
  const cagrBH = annualize(totalDays, totalBH);
  const mddStrat = maxDrawdown(eqStrat);
  const mddBH = maxDrawdown(eqBH);

  const fwd1 = baseRet.slice(1);
  const sig1 = engineOut.state.slice(0, engineOut.state.length - 1);
  const hr1 = hitRate(sig1, fwd1, true);

  console.log('Backtest: Original FFL Classic (QQQ)');
  console.log(`Window: ${WINDOW} | Range: ${START} ~ ${END} | Samples: ${totalDays}`);
  console.log(`Strategy: Total ${(totalStrat*100).toFixed(1)}% | CAGR ${(cagrStrat*100).toFixed(1)}% | MDD ${(mddStrat*100).toFixed(1)}% | Hit(1d) ${(hr1*100).toFixed(1)}%`);
  console.log(`Buy&Hold QQQ: Total ${(totalBH*100).toFixed(1)}% | CAGR ${(cagrBH*100).toFixed(1)}% | MDD ${(mddBH*100).toFixed(1)}%`);
}

main().catch((err) => { console.error(err); process.exit(1); });
