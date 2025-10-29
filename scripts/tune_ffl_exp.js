#!/usr/bin/env node
// Random-search tuner for FFL+EXP ratio parameters against 2020–2025 QQQ benchmark.
// Objective: Outperform QQQ in 2022 and 2023, while keeping regime accuracy decent.
// Usage: node scripts/tune_ffl_exp.js [--trials 80]

const { spawnSync } = require('node:child_process');

function runOnce(params) {
  const env = Object.assign({}, process.env, {
    MODE: 'ffl_exp',
    RON: String(params.rOn),
    ROFF: String(params.rOff),
    ELAM: String(params.lam),
    BMIN: '0',
    D0ANY: '0',
    D0BOTH: '0',
    ONK: String(params.onK),
    OFFK: String(params.offK),
    WIN: String(params.win),
    BACKTEST_START: '2017-01-01',
    BACKTEST_END: '2025-12-31',
    LEV_MODE: '1',
    LEV_R0: String(params.lev.r0),
    LEV_R1: String(params.lev.r1),
    LEV_MIN: String(params.lev.lmin),
    LEV_MAX: String(params.lev.lmax),
    LEV_DAMP: String(params.lev.damp),
    TRON: String(params.brk.on),
    TRMM: String(params.brk.mm),
    TRKAPPA: String(params.brk.kap),
  });
  const res = spawnSync(process.execPath, ['scripts/backtest_ffl.js'], { env, encoding: 'utf8' });
  if (res.status !== 0) {
    return null;
  }
  try { return JSON.parse(res.stdout); } catch { return null; }
}

function scoreRun(out) {
  if (!out) return -1e9;
  const t5 = out.timing?.k5 || {}; const t10 = out.timing?.k10 || {};
  const on5 = Number(t5.on ?? 0), off5 = Number(t5.off ?? 0);
  const on10 = Number(t10.on ?? 0), off10 = Number(t10.off ?? 0);
  // Timing-first objective: equal weight On/Off at 5d, secondary 10d
  let base = 0.5 * (on5 + off5) + 0.25 * (on10 + off10);
  // mild regularization: avoid degenerate always-off by rewarding some On capture
  const onCap = Number(out.regime?.onCapture ?? 0);
  base += 0.05 * (onCap - 0.55);
  // penalize deep drawdowns
  const mdd = Number(out.metrics?.mdd ?? 0);
  base -= 0.10 * Math.max(0, Math.abs(mdd) - 0.35);
  // softly reward equity vs BH
  const eqS = Number(out.equity?.strategy ?? 1); const eqB = Number(out.equity?.benchmark ?? 1);
  const rel = (eqS > 0 && eqB > 0) ? Math.log(eqS) / Math.log(eqB) : 0; // crude scale
  base += 0.10 * (rel - 0.3);
  return base;
}

function randIn(a, b) { return a + Math.random() * (b - a); }

function* candidates(n) {
  for (let i = 0; i < n; i += 1) {
    yield {
      rOn: +(randIn(0.5, 1.5)).toFixed(2),
      rOff: +(randIn(-0.4, 0.2)).toFixed(2),
      lam: +(randIn(0.6, 1.8)).toFixed(2),
      onK: +(randIn(0.5, 1.0)).toFixed(2),
      offK: +(randIn(0.3, 0.8)).toFixed(2),
      win: Math.round(randIn(40, 100)),
      // leverage + breaker
      lev: { r0: +(randIn(0.3, 0.8)).toFixed(2), r1: +(randIn(0.9, 1.3)).toFixed(2), lmin: 1, lmax: 3, damp: +(randIn(1.2, 3.0)).toFixed(2) },
      brk: { on: 1, mm: +(randIn(0.92, 0.98)).toFixed(2), kap: +(randIn(0.50, 0.70)).toFixed(2) }
    };
  }
}

function main() {
  const trials = Number(process.argv.includes('--trials') ? process.argv[process.argv.indexOf('--trials') + 1] : 60);
  const results = [];
  for (const p of candidates(trials)) {
    const out = runOnce(p);
    const score = scoreRun(out);
    results.push({ p, out, score });
  }
  results.sort((a, b) => b.score - a.score);
  const top = results.slice(0, 10);
  console.log(JSON.stringify(top.map(({ p, out, score }) => ({ p, score, yearly: out?.yearly, hitRate: out?.hitRate, regime: out?.regime, equity: out?.equity })), null, 2));
}

if (require.main === module) main();
