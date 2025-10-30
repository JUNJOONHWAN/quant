#!/usr/bin/env node
/*
 * Hyperparameter tuning for FFL+EXP (Classic + Flux + Stability fusion)
 * - Does NOT change the formula; tunes:
 *   - Stability band: macroOn, macroOff
 *   - Flux quantiles: qOn, qOff (used to derive dyn flux thresholds from J_norm distribution)
 * - Train range: 2017-01-01 .. 2023-12-31
 * - Test range:  2024-01-01 .. 2025-10-30
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ARG = process.argv.slice(2);
const TRAIN_START = ARG[0] || '2017-01-01';
const TRAIN_END = ARG[1] || '2023-12-31';
const TEST_START = ARG[2] || '2024-01-01';
const TEST_END = ARG[3] || '2025-10-30';
const WINDOW = 30;
const SYMBOLS = ['QQQ', 'IWM', 'SPY', 'TLT', 'GLD', 'BTC-USD'];

function dateToNum(s) { return Number(s.replace(/-/g, '')); }

function loadHistorical() {
  const p = path.join(__dirname, '..', 'static_site', 'data', 'historical_prices.json');
  if (!fs.existsSync(p)) throw new Error('Missing static_site/data/historical_prices.json');
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function createMM() {
  return require(path.join(__dirname, '..', 'static_site', 'assets', 'metrics.js'));
}

function alignFromHistorical(MM, raw, symbols) {
  const seriesList = raw.assets
    .filter((a) => symbols.includes(a.symbol))
    .map((a) => ({ symbol: a.symbol, category: a.category, dates: a.dates, prices: a.prices }));
  return MM.alignSeries(seriesList);
}

function sliceAligned(aligned, start, end) {
  const s = dateToNum(start); const e = dateToNum(end);
  const dates = aligned.dates.filter((d) => { const n = dateToNum(d); return n >= s && n <= e; });
  const map = new Map(aligned.dates.map((d, i) => [d, i]));
  const prices = {}; Object.entries(aligned.prices).forEach(([sym, arr]) => { prices[sym] = dates.map((d) => arr[map.get(d)]); });
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

function quantile(values, q) {
  const xs = values.filter(Number.isFinite).slice().sort((a,b)=>a-b); if (xs.length === 0) return NaN;
  const pos = Math.min(Math.max(q,0),1) * (xs.length - 1); const lo = Math.floor(pos); const hi = Math.ceil(pos); if (lo===hi) return xs[lo];
  const t = pos - lo; return xs[lo]*(1-t) + xs[hi]*t;
}

function buildFFLExpSeries(ctx, metrics, pricesQQQ, expTune) {
  // apply candidate expTune into VM context and compute via app's EXP branch
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

function statsFromSeries(series) {
  const eq = []; let e = 1; for (const r of series) { e *= 1 + (Number.isFinite(r)?r:0); eq.push(Number(e.toFixed(8))); }
  const total = eq[eq.length - 1] - 1; const days = series.length; const years = Math.max(days/252,1e-9); const cagr = Math.pow(1+total,1/years)-1;
  let peak = eq[0] || 1; let mdd = 0; for (const v of eq){ if (v>peak) peak=v; const dd=(peak-v)/peak; if(dd>mdd) mdd=dd; }
  return { total, cagr, mdd };
}

function hitRateOn(state, fwd) {
  let w=0,c=0; for(let i=0;i<Math.min(state.length-1, fwd.length);i+=1){ const s=state[i]; if(s<=0) continue; const r=fwd[i]; if(Number.isFinite(r)){ if(r>0) w+=1; c+=1; } } return c>0? w/c : 0;
}

function objective({ cagr, mdd, hr1 }) {
  // Weighted objective: prioritize CAGR, then On-day hit rate, penalize MDD
  return (cagr) + 0.3*(hr1 - 0.5) - 0.4*(mdd);
}

async function main() {
  const raw = loadHistorical();
  const MM = createMM();
  const alignedFull = alignFromHistorical(MM, raw, SYMBOLS);
  const ctx = createAppContext(MM);

  function buildRange(start, end) {
    const aligned = sliceAligned(alignedFull, start, end);
    const { returns, metrics } = buildMetrics(ctx, MM, aligned);
    const pricesQQQ = returns.priceSeries['QQQ'];
    const classic = ctx.computeRiskSeriesClassic(metrics, metrics.records);
    const ffl = ctx.computeRiskSeriesFFL(metrics, metrics.records);
    return { returns, metrics, pricesQQQ, classic, ffl };
  }

  const train = buildRange(TRAIN_START, TRAIN_END);
  const test = buildRange(TEST_START, TEST_END);

  const grids = {
    macroOn: [0.45],
    macroOff: [0.35],
    qOn: [0.70, 0.72],
    qOff: [0.18],
    aS: [0.20, 0.25],
    aJ: [0.10],
    bS: [0.20],
    bJ: [0.10],
    gSPos: [0.06],
    gSNeg: [0.06],
    confirmOn: [2],
    confirmOff: [1, 2],
    hazardDrop: [0.03],
  };

  const combos = [];
  for (const mo of grids.macroOn) {
    for (const mf of grids.macroOff) {
      if (mf >= mo) continue;
      for (const qo of grids.qOn) {
        for (const qf of grids.qOff) {
          if (qf >= qo) continue;
          for (const aS of grids.aS) {
            for (const aJ of grids.aJ) {
              for (const bS of grids.bS) {
                for (const bJ of grids.bJ) {
                  for (const gSPos of grids.gSPos) {
                    for (const gSNeg of grids.gSNeg) {
                      for (const confirmOn of grids.confirmOn) {
                        for (const confirmOff of grids.confirmOff) {
                          for (const hazardDrop of grids.hazardDrop) {
                            combos.push({ macroOn: mo, macroOff: mf, qOn: qo, qOff: qf, aS, aJ, bS, bJ, gSPos, gSNeg, confirmOn, confirmOff, hazardDrop });
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  let best = null; let bestObj = -Infinity;
  for (const p of combos) {
    const seriesTrain = buildFFLExpSeries(ctx, train.metrics, train.pricesQQQ, p);
    const fwd1 = seriesTrain.baseRet.slice(1);
    const hr1 = hitRateOn(seriesTrain.state, fwd1);
    const stTrain = statsFromSeries(seriesTrain.executed.map((reg,i)=> reg>0 ? Math.max(-0.99, 3*seriesTrain.baseRet[i]) : reg<0 ? 0 : seriesTrain.baseRet[i]));
    let flips = 0; for (let i=1;i<seriesTrain.state.length;i+=1){ if(seriesTrain.state[i]!==seriesTrain.state[i-1]) flips+=1; }
    const flipRate = flips / Math.max(1, seriesTrain.state.length);
    const obj = objective({ cagr: stTrain.cagr, mdd: stTrain.mdd, hr1 }) - 0.2*flipRate;
    if (obj > bestObj) { bestObj = obj; best = { params: p, train: { hr1, ...stTrain } }; }
  }

  const p = best.params;
  const seriesTest = buildFFLExpSeries(ctx, test.metrics, test.pricesQQQ, p);
  const fwd1Test = seriesTest.baseRet.slice(1);
  const hr1Test = hitRateOn(seriesTest.state, fwd1Test);
  const stTest = statsFromSeries(seriesTest.executed.map((reg,i)=> reg>0 ? Math.max(-0.99, 3*seriesTest.baseRet[i]) : reg<0 ? 0 : seriesTest.baseRet[i]));

  console.log('Best params (train 2017-2023):', best.params);
  console.log('Train: Total %s, CAGR %s, MDD %s, HR_on(1d) %s',
    (best.train.total*100).toFixed(1)+'%', (best.train.cagr*100).toFixed(1)+'%', (best.train.mdd*100).toFixed(1)+'%', (best.train.hr1*100).toFixed(1)+'%');
  console.log('Test (2024-2025): Total %s, CAGR %s, MDD %s, HR_on(1d) %s',
    (stTest.total*100).toFixed(1)+'%', (stTest.cagr*100).toFixed(1)+'%', (stTest.mdd*100).toFixed(1)+'%', (hr1Test*100).toFixed(1)+'%');
}

main().catch((e)=>{ console.error(e); process.exit(1); });
