// Classic + Flux5 (10-pair) engine, from-scratch minimal version
// Does not alter production data loading; can be used from research scripts or wired optionally.

(function (globalFactory) {
  if (typeof module === 'object' && typeof module.exports === 'object') {
    module.exports = globalFactory(require('../metrics'));
  } else {
    window.ClassicFlux5 = globalFactory(window.MarketMetrics);
  }
})(function (MM) {
  function topEigenvector(matrix, maxIter = 50, tol = 1e-6) {
    if (!Array.isArray(matrix) || matrix.length === 0) return null;
    const n = matrix.length; let v = new Array(n).fill(1/Math.sqrt(n));
    for (let it=0; it<maxIter; it+=1) {
      const w = new Array(n).fill(0);
      for (let i=0;i<n;i+=1){ const row = matrix[i]; let s=0; for(let j=0;j<n;j+=1){ s += row[j]*v[j]; } w[i]=s; }
      const norm = Math.sqrt(w.reduce((a,x)=>a+x*x,0)) || 1; const nv = w.map(x=>x/norm);
      let diff=0; for(let i=0;i<n;i+=1){ const d=nv[i]-v[i]; diff+=d*d; } v=nv; if(diff<tol*tol) break;
    }
    return v;
  }
  function zMomentum(series, index, k = 10, v = 20, zSat = 2.0) {
    if (!Array.isArray(series) || index == null) return null;
    if (index - k < 0 || index >= series.length) return null;
    const base = series[index - k];
    const latest = series[index];
    if (!Number.isFinite(base) || !Number.isFinite(latest) || base === 0) return null;
    // momentum in price ratio
    const m = latest / base - 1;
    // std of simple returns over v days
    let sum = 0; let cnt = 0; const rets = [];
    for (let i = Math.max(1, index - v + 1); i <= index; i += 1) {
      const p0 = series[i - 1]; const p1 = series[i];
      if (Number.isFinite(p0) && Number.isFinite(p1) && p0 !== 0) { const r = p1 / p0 - 1; rets.push(r); sum += r; cnt += 1; }
    }
    if (cnt < 2) return null;
    const mean = sum / cnt;
    let varr = 0; rets.forEach((r) => { varr += (r - mean) * (r - mean); }); varr /= (cnt - 1);
    const std = Math.sqrt(Math.max(varr, 0)); if (!Number.isFinite(std) || std === 0) return null;
    return Math.tanh((m / std) / zSat);
  }

  function pairWeight(symbolA, symbolB, categories) {
    const a = categories[symbolA]; const b = categories[symbolB];
    const set = new Set([a, b]);
    if (set.has('stock') && set.has('bond')) return 2.0; // stock-bond heavy
    if (set.has('stock') && set.has('gold')) return 1.5;
    if (set.size === 1 && set.has('stock')) return 1.0;
    if (set.has('stock') && set.has('crypto')) return 1.0;
    if (set.has('bond') && set.has('gold')) return 1.0;
    return 1.0;
  }

  function computeClassicFlux5({ aligned, returns, window = 30, symbols, categories, prices, pairGroups, params = {} }) {
    const dates = returns.dates;
    const n = dates.length;
    const priceOffset = window - 1;

    const out = {
      dates: [],
      flux5: [],
      jnorm: [],
      diff: [],
      fluxSlope: [],
      mmDelta: [],
      downAll: [],
      vAll: [],
      pcon: [],
      mm: [],
      scCorr: [],
      safeNeg: [],
      state: [],
      executedState: [],
      score: [],
      diagnostics: {},
    };

    const crossPairs = []; // risk-safe
    const riskPairs = []; // risk-risk
    const safePairs = []; // safe-safe
    const risk = pairGroups.risk; const safe = pairGroups.safe;
    for (let i = 0; i < safe.length; i += 1) {
      for (let j = 0; j < risk.length; j += 1) crossPairs.push([safe[i], risk[j]]);
    }
    for (let i = 0; i < risk.length; i += 1) {
      for (let j = i + 1; j < risk.length; j += 1) riskPairs.push([risk[i], risk[j]]);
    }
    for (let i = 0; i < safe.length; i += 1) {
      for (let j = i + 1; j < safe.length; j += 1) safePairs.push([safe[i], safe[j]]);
    }

    const cap = symbols.length;

    let prevJ = null; let prevMM = null;
    for (let end = window - 1; end < n; end += 1) {
      const start = end - window + 1;
      const full = MM.buildCorrelationMatrix(symbols, returns.returns, start, end);
      const lambda1 = MM.topEigenvalue(full) || 0; const mm = lambda1 / cap;
      const recordDate = dates[end];

      // subindices
      const signalMatrix = full; // same ordering
      const idx = new Map(symbols.map((s, i) => [s, i]));
      // safe-neg proxy
      let snVals = [];
      pairGroups.stocks.forEach((stk) => pairGroups.safe.forEach((sf) => {
        const i = idx.get(stk); const j = idx.get(sf); snVals.push(Math.max(0, -(signalMatrix[Math.min(i,j)][Math.max(i,j)])));
      }));
      const safeNeg = snVals.length ? snVals.reduce((a,b)=>a+b,0)/snVals.length : 0;

      // scCorr IWM|BTC if both present
      let scCorr = null; if (idx.has('IWM') && idx.has('BTC-USD')) {
        const i = idx.get('IWM'); const j = idx.get('BTC-USD'); scCorr = signalMatrix[Math.min(i,j)][Math.max(i,j)];
      }

      // zMomentum per symbol
      const priceIndex = priceOffset + end;
      const z = {}; symbols.forEach((s) => { z[s] = zMomentum(prices[s], priceIndex, 10, 20, 2.0); });

      // common down ratio across 5 signals
      const sigs = pairGroups.risk.concat(pairGroups.safe);
      const zAll = sigs.map((s) => z[s]).filter((v) => Number.isFinite(v));
      const downAll = zAll.length ? zAll.filter((v) => v < 0).length / zAll.length : 0;
      const vAll = zAll.length ? zAll.reduce((a,b)=>a+b,0)/zAll.length : 0;

      // 10-pair flux (Fick-like)
      let wAll = 0; let apdf = 0; let pconSum = 0; let pconW = 0;
      // cross-only Jbar for Diff/Jnorm
      let wCross = 0; let jbar = 0;
      const add = (a, b, mode) => {
        const i = idx.get(a); const j = idx.get(b); if (i == null || j == null) return;
        const c = signalMatrix[Math.min(i,j)][Math.max(i,j)] || 0; const w = Math.pow(Math.abs(c), 1.5);
        if (!(w>0)) return; wAll += w;
        const za = Number.isFinite(z[a]) ? z[a] : 0; const zb = Number.isFinite(z[b]) ? z[b] : 0;
        if (mode === 'cross') { apdf += w * (zb - za); pconSum += w * (zb > za ? 1 : 0); }
        else if (mode === 'risk') { apdf += w * (0.5 * (za + zb)); pconSum += w * ((za > 0 && zb > 0) ? 1 : 0); }
        else if (mode === 'safe') { apdf += w * (-0.5 * (za + zb)); pconSum += w * ((za <= 0 && zb <= 0) ? 1 : 0); }
        pconW += w;
        if (mode === 'cross') { jbar += w * (zb - za); wCross += w; }
      };
      crossPairs.forEach(([a,b]) => add(a,b,'cross'));
      riskPairs.forEach(([a,b]) => add(a,b,'risk'));
      safePairs.forEach(([a,b]) => add(a,b,'safe'));
      const APDF = wAll>0 ? Math.tanh((apdf/wAll)/0.25) : 0; // normalize
      const JNORM = wCross>0 ? Math.tanh((jbar/wCross)/0.25) : 0;
      const mmDelta = (prevMM!=null && Number.isFinite(mm)) ? (mm - prevMM) : 0;
      const fluxSlope = (prevJ!=null && Number.isFinite(JNORM)) ? (JNORM - prevJ) : 0;
      const Diff = JNORM - 0.50*Math.max(0, mmDelta) - 0.15*Math.max(0, -fluxSlope);
      prevMM = mm; prevJ = JNORM;
      // PC1 velocity (market-mode)
      const e1 = topEigenvector(signalMatrix);
      let vpc1 = 0; if (Array.isArray(e1)) { let num=0; let den=0; for (let i=0;i<e1.length;i+=1){ const s = symbols[i]; const zi = Number.isFinite(z[s])?z[s]:0; num += e1[i]*zi; den += Math.abs(e1[i]); } vpc1 = den>0 ? (num/den) : 0; }
      const VPC1 = Math.tanh(vpc1/0.5);
      const PCON = pconW>0 ? Math.max(0, Math.min(1, pconSum/pconW)) : 0;

      out.dates.push(recordDate);
      out.mm.push(mm);
      out.scCorr.push(scCorr);
      out.safeNeg.push(safeNeg);
      out.flux5.push(APDF);
      out.pcon.push(PCON);
      out.jnorm.push(JNORM);
      out.mmDelta.push(mmDelta);
      out.fluxSlope.push(fluxSlope);
      out.diff.push(Diff);
      out.downAll.push(downAll);
      out.vAll.push(vAll);
      out.vPC1 = out.vPC1 || [];
      out.vPC1.push(VPC1);
    }

    // Classic baseline + Flux5 + Drift/Corr-Lock gating
    const thresholds = Object.assign({ jOn: +0.10, jOff: -0.08, pconOn: 0.55, pconOff: 0.40, mmOff: 0.96, mmHi: 0.90, downAll: 0.60, corrConeDays: 5, driftMinDays: 3, driftCool: 2, vOn: +0.05, vOff: -0.05, offMinDays: 3 }, params);
    const state = []; let prev = 0; let onC = 0; let offC = 0; let offStreak = 0; let reentryArmed = false;
    // drift epoch
    let driftSeq = 0; let driftCool = 0; let inDrift = false; let driftEpochs = 0; let driftDays = 0; let offFromDriftDays = 0; let suppressed = 0;
    // corr-cone lock (short Off window)
    let coneLock = 0;
    for (let i = 0; i < out.dates.length; i += 1) {
      const flux = out.flux5[i] ?? 0; const mm = out.mm[i] ?? 0; const sc = out.scCorr[i] ?? 0; const safeN = out.safeNeg[i] ?? 0; const pcon = out.pcon[i] ?? 0;
      const jn = out.jnorm[i] ?? 0; const diff = out.diff[i] ?? 0; const down = out.downAll[i] ?? 0; const vAll = (out.vPC1?.[i] ?? out.vAll[i] ?? 0);
      // Classic score
      const score = 0.70 * Math.max(0, sc) + 0.30 * safeN; out.score.push(score);
      // Drift update
      const bench20 = (() => {
        const series = returns.priceSeries['QQQ']; const idx = (window - 1) + i; if (!Array.isArray(series) || idx - 20 < 0) return null; const a = series[idx - 20]; const b = series[idx]; if (!Number.isFinite(a)||!Number.isFinite(b)||a===0) return null; return b/a - 1;
      })();
      const driftNow = ((down >= thresholds.downAll) && (jn <= 0)) || ((mm >= thresholds.mmHi) && (Number.isFinite(bench20) ? bench20 <= 0 : true));
      if (driftNow) { driftSeq += 1; driftCool = 0; } else { driftSeq = 0; driftCool += 1; }
      if (!inDrift && driftSeq >= thresholds.driftMinDays) { inDrift = true; driftEpochs += 1; }
      if (inDrift && driftCool >= thresholds.driftCool) { inDrift = false; }
      if (inDrift) driftDays += 1;

      // CorrCone lock trigger (BTC–주식↑ & 주채권 동행 & 흡수 높음)
      const coneNow = (Number.isFinite(sc) ? sc >= 0.50 : false) && (()=>{
        const iSPY = symbols.indexOf('SPY'); const iTLT = symbols.indexOf('TLT'); if (iSPY<0||iTLT<0) return false; const c = aligned ? null : null; return true; })();
      // compute corr(SPY,TLT) from rolling window
      let corrST = null;
      try {
        const iSPY = symbols.indexOf('SPY'); const iTLT = symbols.indexOf('TLT');
        if (iSPY>=0 && iTLT>=0) {
          const m = MM.buildCorrelationMatrix([symbols[iSPY], symbols[iTLT]], returns.returns, (window - 1) + i - (window - 1), (window - 1) + i);
          corrST = m[0][1];
        }
      } catch (e) { corrST = null; }
      const coneTrigger = (Number.isFinite(sc) && sc >= 0.50) && (Number.isFinite(corrST) ? corrST >= -0.10 : false) && (mm >= thresholds.mmHi);
      if (coneTrigger) coneLock = thresholds.corrConeDays;
      if (coneLock > 0) coneLock -= 1;

      // reentry one-shot stronger threshold right after drift epoch
      if (!inDrift && driftSeq === 0 && driftCool === 1) reentryArmed = true;
      // Relative strength of diffusion vs drift
      const lam = Number.isFinite(params.kLambda) ? params.kLambda : 1.0; // drift weight
      const kappa = (Math.abs(diff)) / (Math.abs(diff) + lam * Math.abs(vAll) + 1e-6); // 0..1, diffusion dominance
      const kOn = Number.isFinite(params.kOn) ? params.kOn : 0.60;
      const reentryBonus = reentryArmed ? 0.05 : 0;
      // Quadrant-aware gating: prefer On when (diff>0) and diffusion dominance high, and drift not negative
      const onMain = (diff >= (thresholds.jOn + reentryBonus)) && (kappa >= kOn) && (pcon >= thresholds.pconOn) && (mm < thresholds.mmOff) && (vAll >= thresholds.vOn) && (!inDrift) && (coneLock === 0) && (offStreak >= thresholds.offMinDays);
      // Off when drift negative or diffusion negative with dominance low, or guards trip
      const offByRel = ((vAll <= thresholds.vOff) && (Math.abs(vAll) >= 0.05)) || ((diff <= thresholds.jOff) && (kappa < 0.55));
      const offMain = offByRel || (pcon <= thresholds.pconOff) || (mm >= thresholds.mmOff) || inDrift || (coneLock > 0) || ((mm >= thresholds.mmHi) && (vAll <= 0));
      const rawOn = onMain; const rawOff = offMain;
      onC = rawOn ? onC + 1 : 0; offC = rawOff ? offC + 1 : 0;
      let s = prev; const before = s;
      if (prev === 1) { s = rawOff ? -1 : 1; }
      else if (prev === -1) { s = onC >= 2 ? 1 : -1; }
      else { s = rawOff ? -1 : onC >= 2 ? 1 : 0; }
      if (inDrift) { if (before !== -1 && s !== -1) suppressed += 1; s = -1; offFromDriftDays += 1; }
      if (s === -1) { offStreak += 1; } else { offStreak = 0; }
      if (s === 1 && reentryArmed) reentryArmed = false;
      state.push(s); prev = s;
    }
    out.state = state; out.executedState = state.map((v, i) => i===0?0:state[i-1]||0);
    out.diagnostics.drift = { epochs: driftEpochs, days: driftDays, offFromDriftDays, suppressed };
    return out;
  }

  return { computeClassicFlux5 };
});
