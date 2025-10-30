#!/usr/bin/env node
/*
 * Tune Stability-only parameters from 1.txt / 2.txt tables.
 * - Parse Stability/Smoothed columns
 * - Detect cycle via pivots on smoothed series
 * - Map to stabTune (fast/slow/zWin/thresholds/lags/windows)
 */
const fs = require('fs');
const path = require('path');

function readTxt(name) {
  const p = path.join(process.cwd(), name);
  if (!fs.existsSync(p)) return null;
  return fs.readFileSync(p, 'utf8');
}

function parseTable(txt) {
  if (!txt) return [];
  const lines = txt.split(/\r?\n/);
  const out = [];
  for (const line of lines) {
    // Expect: YYYY-MM-DD\t<stability>\t<smoothed>\t<delta>
    if (!/^\d{4}-\d{2}-\d{2}\t/.test(line)) continue;
    const cols = line.split('\t');
    if (cols.length < 2) continue;
    const date = cols[0].trim();
    const stability = Number(cols[1].replace(/[^0-9.+-]/g, ''));
    const smoothed = cols.length >= 3 ? Number(cols[2].replace(/[^0-9.+-]/g, '')) : NaN;
    if (!Number.isFinite(stability) && !Number.isFinite(smoothed)) continue;
    out.push({ date, stability, smoothed: Number.isFinite(smoothed) ? smoothed : stability });
  }
  return out;
}

function mergeByDate(a, b) {
  const map = new Map();
  for (const r of (a||[])) map.set(r.date, r);
  for (const r of (b||[])) map.set(r.date, r);
  return Array.from(map.values()).sort((x,y)=> x.date<y.date? -1 : x.date>y.date? 1 : 0);
}

function ema(arr, n) {
  const k = 2/(n+1);
  const out = new Array(arr.length).fill(null);
  let prev;
  for (let i=0;i<arr.length;i++) {
    const v = Number(arr[i]);
    if (!Number.isFinite(v)) { out[i] = prev==null? null : prev; continue; }
    prev = (prev==null)? v : prev + k*(v - prev);
    out[i] = prev;
  }
  return out;
}

function detectCycleDays(series) {
  // Use EMA(10) smoothing for pivot detection
  const sm = ema(series, 10);
  const pivots = [];
  for (let i=2;i<sm.length-2;i++) {
    const v=sm[i]; if(!Number.isFinite(v)) continue;
    const up = sm[i-2] < sm[i-1] && sm[i-1] < v && v > sm[i+1] && sm[i+1] > sm[i+2];
    const dn = sm[i-2] > sm[i-1] && sm[i-1] > v && v < sm[i+1] && sm[i+1] < sm[i+2];
    if (up || dn) pivots.push(i);
  }
  if (pivots.length < 4) return 30;
  const gaps = []; for (let i=1;i<pivots.length;i++) gaps.push(pivots[i]-pivots[i-1]);
  gaps.sort((a,b)=>a-b);
  const mid = gaps[Math.floor(gaps.length/2)];
  const half = Math.max(15, Math.min(90, mid));
  return Math.round(2*half);
}

function quantile(xs, q) {
  const arr = xs.filter(Number.isFinite).slice().sort((a,b)=>a-b);
  if (arr.length===0) return NaN;
  const pos = (arr.length-1)*q; const lo=Math.floor(pos), hi=Math.ceil(pos);
  if (lo===hi) return arr[lo];
  const w = pos-lo; return arr[lo]*(1-w)+arr[hi]*w;
}

function clamp(x, lo, hi){ return Math.max(lo, Math.min(hi, x)); }

function main() {
  const t1 = readTxt('1.txt');
  const t2 = readTxt('2.txt');
  const a = parseTable(t1); const b = parseTable(t2);
  if ((!a || a.length===0) && (!b || b.length===0)) {
    console.error('no stability rows parsed'); process.exit(1);
  }
  const rows = mergeByDate(a, b);
  const S = rows.map(r=> r.smoothed);
  const cycleDays = detectCycleDays(S);
  // Map cycle to windows
  const fast = clamp(Math.round(0.7 * cycleDays), 14, 42);
  const slow = clamp(Math.round(2.1 * cycleDays), Math.max(2*fast, 40), 126);
  const zWin = clamp(Math.round(2.0 * slow), 63, 252);

  // Build slope using provisional windows
  const ef = ema(S, fast);
  const es = ema(S, slow);
  const slope = ef.map((v,i)=> (Number.isFinite(v) && Number.isFinite(es[i])) ? (v - es[i]) : NaN);
  // Use rolling std proxy for z-calibration (global as proxy)
  const absSlope = slope.map(x=> Math.abs(x)).filter(Number.isFinite);
  const slopeMin = clamp(quantile(absSlope, 0.60), 0.008, 0.020);
  // approximate z thresholds from distribution
  const sMean = absSlope.reduce((p,c)=>p+c,0)/Math.max(absSlope.length,1);
  const sStd = Math.sqrt(absSlope.reduce((p,c)=>p+(c-sMean)*(c-sMean),0) / Math.max(absSlope.length-1,1));
  const zApprox = absSlope.map(x=> (sStd>0? x/sStd : 0));
  const zq = quantile(zApprox, 0.90);
  const zThr = clamp(zq || 1.8, 1.4, 2.5);

  // Lags & windows from cycle
  const lagUp = clamp(Math.ceil(0.10 * cycleDays), 2, 6);
  const lagDown = clamp(Math.ceil(0.13 * cycleDays), 3, 8);
  const leadOnWindow = clamp(Math.ceil(0.17 * cycleDays), 3, 8);
  const downGrace = clamp(Math.ceil(0.17 * cycleDays), 3, 8);
  const hazardWindow = clamp(Math.ceil(0.27 * cycleDays), 4, 12);

  const stabTune = {
    fast, slow, zWin,
    zUp: Number(zThr.toFixed(2)),
    zDown: Number(zThr.toFixed(2)),
    slopeMin: Number(slopeMin.toFixed(4)),
    neutralLo: 0.30, neutralHi: 0.40,
    lagUp, lagDown,
    onFluxEase: 0.02,
    confirmOnMin: 2,
    leadOnWindow,
    downGrace,
    hazardWindow,
    offFluxTighten: 0.03,
    confirmOffMin: 1,
    onOverrideMargin: 0.01,
  };
  const out = { cycleDays, stabTune };
  const outPath = path.join(__dirname, 'tune_stability_from_txt.json');
  fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
  console.log('[write]', outPath);
  console.log(JSON.stringify(out, null, 2));
}

main();

