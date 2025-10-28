// Classic + Flux5 (10-pair) engine, from-scratch minimal version
// Does not alter production data loading; can be used from research scripts or wired optionally.

(function (globalFactory) {
  if (typeof module === 'object' && typeof module.exports === 'object') {
    module.exports = globalFactory(require('../metrics'));
  } else {
    window.ClassicFlux5 = globalFactory(window.MarketMetrics);
  }
})(function (MM) {
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

  function computeClassicFlux5({ aligned, returns, window = 30, symbols, categories, prices, pairGroups }) {
    const dates = returns.dates;
    const n = dates.length;
    const priceOffset = window - 1;

    const out = {
      dates: [],
      flux5: [],
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

      // 10-pair flux (Fick-like)
      let wAll = 0; let apdf = 0; let pconSum = 0; let pconW = 0;
      const add = (a, b, mode) => {
        const i = idx.get(a); const j = idx.get(b); if (i == null || j == null) return;
        const c = signalMatrix[Math.min(i,j)][Math.max(i,j)] || 0; const w = Math.pow(Math.abs(c), 1.5);
        if (!(w>0)) return; wAll += w;
        const za = Number.isFinite(z[a]) ? z[a] : 0; const zb = Number.isFinite(z[b]) ? z[b] : 0;
        if (mode === 'cross') { apdf += w * (zb - za); pconSum += w * (zb > za ? 1 : 0); }
        else if (mode === 'risk') { apdf += w * (0.5 * (za + zb)); pconSum += w * ((za > 0 && zb > 0) ? 1 : 0); }
        else if (mode === 'safe') { apdf += w * (-0.5 * (za + zb)); pconSum += w * ((za <= 0 && zb <= 0) ? 1 : 0); }
        pconW += w;
      };
      crossPairs.forEach(([a,b]) => add(a,b,'cross'));
      riskPairs.forEach(([a,b]) => add(a,b,'risk'));
      safePairs.forEach(([a,b]) => add(a,b,'safe'));
      const APDF = wAll>0 ? Math.tanh((apdf/wAll)/0.25) : 0; // normalize
      const PCON = pconW>0 ? Math.max(0, Math.min(1, pconSum/pconW)) : 0;

      out.dates.push(recordDate); out.mm.push(mm); out.scCorr.push(scCorr); out.safeNeg.push(safeNeg); out.flux5.push(APDF); out.pcon.push(PCON);
    }

    // Classic baseline + Flux5 gating (minimal)
    const thresholds = { jOn: +0.10, jOff: -0.08, pconOn: 0.55, pconOff: 0.40, mmOff: 0.96 };
    const state = []; let prev = 0; let onC = 0; let offC = 0;
    for (let i = 0; i < out.dates.length; i += 1) {
      const flux = out.flux5[i] ?? 0; const mm = out.mm[i] ?? 0; const sc = out.scCorr[i] ?? 0; const safeN = out.safeNeg[i] ?? 0; const pcon = out.pcon[i] ?? 0;
      // Classic score
      const score = 0.70 * Math.max(0, sc) + 0.30 * safeN; out.score.push(score);
      const onMain = (flux >= thresholds.jOn) && (pcon >= thresholds.pconOn) && (mm < thresholds.mmOff);
      const offMain = (flux <= thresholds.jOff) || (pcon <= thresholds.pconOff) || (mm >= thresholds.mmOff);
      const rawOn = onMain; const rawOff = offMain;
      onC = rawOn ? onC + 1 : 0; offC = rawOff ? offC + 1 : 0;
      let s = prev;
      if (prev === 1) { s = rawOff ? -1 : 1; }
      else if (prev === -1) { s = onC >= 2 ? 1 : -1; }
      else { s = rawOff ? -1 : onC >= 2 ? 1 : 0; }
      state.push(s); prev = s;
    }
    out.state = state; out.executedState = state.map((v, i) => i===0?0:state[i-1]||0);
    return out;
  }

  return { computeClassicFlux5 };
});

