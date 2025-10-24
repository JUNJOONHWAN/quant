/*
 * Shared numerical utilities for the market stability dashboard.
 * The module is intentionally framework-agnostic so it can run both
 * in the browser (exposed on `window.MarketMetrics`) and in Node.js
 * test environments (via `module.exports`).
 */
(function (global, factory) {
  if (typeof module === 'object' && typeof module.exports === 'object') {
    module.exports = factory();
  } else {
    global.MarketMetrics = factory();
  }
})(typeof self !== 'undefined' ? self : this, () => {
  function alignSeries(seriesList) {
    const dateCounts = new Map();
    seriesList.forEach((series) => {
      series.dates.forEach((date) => {
        dateCounts.set(date, (dateCounts.get(date) || 0) + 1);
      });
    });

    const targetCount = seriesList.length;
    const commonDates = Array.from(dateCounts.entries())
      .filter(([, count]) => count === targetCount)
      .map(([date]) => date)
      .sort();

    const priceBySymbol = {};
    seriesList.forEach((series) => {
      const map = new Map();
      series.dates.forEach((date, index) => {
        map.set(date, series.prices[index]);
      });
      priceBySymbol[series.symbol] = commonDates.map((date) => map.get(date));
    });

    return {
      dates: commonDates,
      prices: priceBySymbol,
      categories: Object.fromEntries(
        seriesList.map((series) => [series.symbol, series.category]),
      ),
    };
  }

  function computeReturns(aligned) {
    const analysisDates = aligned.dates.slice(1);
    const normalizedPrices = {};
    const returns = {};

    Object.entries(aligned.prices).forEach(([symbol, prices]) => {
      if (prices.length < 2) {
        throw new Error(`${symbol}의 데이터가 충분하지 않습니다.`);
      }
      const assetReturns = [];
      const normalized = [];
      const base = prices[1];
      for (let i = 1; i < prices.length; i += 1) {
        const current = prices[i];
        const previous = prices[i - 1];
        assetReturns.push(Math.log(current / previous));
        normalized.push(current / base);
      }
      returns[symbol] = assetReturns;
      normalizedPrices[symbol] = normalized;
    });

    return {
      dates: analysisDates,
      returns,
      normalizedPrices,
    };
  }

  function buildCorrelationMatrix(symbols, returnsBySymbol, startIndex, endIndex) {
    const matrix = [];
    symbols.forEach((symbolA, i) => {
      const row = [];
      symbols.forEach((symbolB, j) => {
        if (i === j) {
          row.push(1);
        } else if (j < i) {
          row.push(matrix[j][i]);
        } else {
          const seriesA = returnsBySymbol[symbolA].slice(startIndex, endIndex + 1);
          const seriesB = returnsBySymbol[symbolB].slice(startIndex, endIndex + 1);
          row.push(correlation(seriesA, seriesB));
        }
      });
      matrix.push(row);
    });
    return matrix;
  }

  function computeStability(matrix, symbols, categories) {
    let weighted = 0;
    let totalWeight = 0;
    for (let i = 0; i < symbols.length; i += 1) {
      for (let j = i + 1; j < symbols.length; j += 1) {
        const weight = pairWeight(symbols[i], symbols[j], categories);
        weighted += weight * Math.abs(matrix[i][j]);
        totalWeight += weight;
      }
    }
    return totalWeight > 0 ? weighted / totalWeight : 0;
  }

  function computeSubIndices(matrix, symbols, categories) {
    const stockSymbols = symbols.filter((symbol) => categories[symbol] === 'stock');
    const cryptoSymbols = symbols.filter((symbol) => categories[symbol] === 'crypto');
    const traditionalSymbols = symbols.filter((symbol) => ['stock', 'bond', 'gold'].includes(categories[symbol]));
    const safeTargets = symbols.filter((symbol) => ['bond', 'gold'].includes(categories[symbol]));

    const stockCrypto = averagePairs(matrix, symbols, stockSymbols, cryptoSymbols, Math.abs);
    const traditional = averagePairs(matrix, symbols, traditionalSymbols, traditionalSymbols, Math.abs, true);
    const safeNegative = averagePairs(matrix, symbols, stockSymbols, safeTargets, (value) => Math.max(0, -value));

    return { stockCrypto, traditional, safeNegative };
  }

  function averagePairs(matrix, symbols, groupA, groupB, transform, skipSame = false) {
    const index = new Map(symbols.map((symbol, idx) => [symbol, idx]));
    const values = [];
    groupA.forEach((symbolA) => {
      groupB.forEach((symbolB) => {
        if (skipSame && symbolA === symbolB) return;
        if (symbolA === symbolB) return;
        const i = index.get(symbolA);
        const j = index.get(symbolB);
        if (i == null || j == null) return;
        const value = transform(matrix[Math.min(i, j)][Math.max(i, j)]);
        values.push(value);
      });
    });
    return values.length > 0 ? mean(values) : 0;
  }

  function ema(values, period) {
    if (values.length === 0) return [];
    const result = [];
    const k = 2 / (period + 1);
    let prev = values[0];
    result.push(prev);
    for (let i = 1; i < values.length; i += 1) {
      const current = values[i] * k + prev * (1 - k);
      result.push(current);
      prev = current;
    }
    return result;
  }

  function correlation(seriesA, seriesB) {
    const n = Math.min(seriesA.length, seriesB.length);
    if (n === 0) return 0;
    let meanA = 0;
    let meanB = 0;
    for (let i = 0; i < n; i += 1) {
      meanA += seriesA[i];
      meanB += seriesB[i];
    }
    meanA /= n;
    meanB /= n;
    let numerator = 0;
    let varA = 0;
    let varB = 0;
    for (let i = 0; i < n; i += 1) {
      const da = seriesA[i] - meanA;
      const db = seriesB[i] - meanB;
      numerator += da * db;
      varA += da * da;
      varB += db * db;
    }
    if (varA === 0 || varB === 0) return 0;
    return numerator / Math.sqrt(varA * varB);
  }

  function mean(values) {
    if (!values || values.length === 0) return 0;
    const sum = values.reduce((acc, value) => acc + value, 0);
    return sum / values.length;
  }

  function pairWeight(symbolA, symbolB, categories) {
    const categoryA = categories[symbolA];
    const categoryB = categories[symbolB];
    const pair = new Set([categoryA, categoryB]);

    if (pair.has('stock') && pair.has('bond')) return 2.0;
    if (pair.has('stock') && pair.has('gold')) return 1.5;
    if (pair.size === 1 && pair.has('stock')) return 1.0;
    if (pair.has('stock') && pair.has('crypto')) return 1.0;
    if (pair.has('bond') && pair.has('gold')) return 1.0;
    return 1.0;
  }

  return {
    alignSeries,
    computeReturns,
    buildCorrelationMatrix,
    computeStability,
    computeSubIndices,
    averagePairs,
    ema,
    correlation,
    mean,
    pairWeight,
  };
});
