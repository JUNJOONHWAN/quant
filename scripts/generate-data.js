#!/usr/bin/env node
/*
 * Generate precomputed market stability metrics using Financial Modeling Prep data.
 * The script writes a JSON blob that the GitHub Pages frontend consumes.
 */
const fs = require('fs/promises');
const path = require('path');

function flagEnabled(value) {
  if (value == null) {
    return false;
  }
  const normalized = String(value).trim().toLowerCase();
  return normalized === '1' || normalized === 'true' || normalized === 'yes';
}

const {
  alignSeries,
  computeReturns,
  buildCorrelationMatrix,
  computeStability,
  computeSubIndices,
  ema,
  mean,
} = require('../static_site/assets/metrics');

const API_KEY = process.env.FMP_API_KEY;
const USE_HISTORY_ONLY = flagEnabled(
  process.env.USE_FMP_HISTORY_ONLY
  || process.env.GENERATE_USE_FMP_ONLY
  || process.env.GENERATE_USE_HISTORY_ONLY
);
const DATA_DIR = path.join(__dirname, '..', 'static_site', 'data');
const PRECOMPUTED_PATH = path.join(DATA_DIR, 'precomputed.json');
const HISTORICAL_PATH = path.join(DATA_DIR, 'historical_prices.json');

const DATA_START_DATE = '2017-01-01';
const ASSETS = [
  { symbol: 'QQQ', label: 'QQQ (NASDAQ 100 ETF)', category: 'stock', fmpSymbol: 'QQQ' },
  { symbol: 'IWM', label: 'IWM (Russell 2000 ETF)', category: 'stock', fmpSymbol: 'IWM' },
  { symbol: 'SPY', label: 'SPY (S&P 500 ETF)', category: 'stock', fmpSymbol: 'SPY' },
  { symbol: 'TLT', label: 'TLT (미국 장기채)', category: 'bond', fmpSymbol: 'TLT' },
  { symbol: 'GLD', label: 'GLD (금 ETF)', category: 'gold', fmpSymbol: 'GLD' },
  { symbol: 'BTC-USD', label: 'BTC-USD (비트코인)', category: 'crypto', fmpSymbol: 'BTCUSD' },
];

const SIGNAL = {
  symbols: ['IWM', 'SPY', 'TLT', 'GLD', 'BTC-USD'],
};

const WINDOWS = [20, 30, 60];
const MINIMUM_CUTOFF_DATE = DATA_START_DATE;

async function main() {
  if (typeof fetch !== 'function') {
    await emitNoDataWithReason('global fetch is not available. emitting no_data placeholder.');
    return;
  }

  const hasApiKey = Boolean(API_KEY);
  const historical = await readHistoricalSeries();
  const cutoffDate = computeCutoffDate();
  const endDate = computeEndDate();
  const useHistoryOnly = USE_HISTORY_ONLY || !hasApiKey;

  try {
    let mergedSeries;
    if (useHistoryOnly) {
      if (!historical || !(historical.map instanceof Map) || historical.map.size === 0) {
        throw new Error('Historical dataset is unavailable; cannot build precomputed payload without FMP data.');
      }
      if (!hasApiKey) {
        console.warn('[generate-data] FMP_API_KEY missing; using cached dataset only.');
      } else if (USE_HISTORY_ONLY) {
        console.log('[generate-data] USE_FMP_HISTORY_ONLY/GENERATE_USE_FMP_ONLY enabled; skipping live fetch.');
      }
      mergedSeries = buildSeriesFromHistory(historical, cutoffDate);
    } else {
      const assetSeries = [];
      for (let index = 0; index < ASSETS.length; index += 1) {
        const asset = ASSETS[index];
        console.log(`Fetching ${asset.symbol} via Financial Modeling Prep`);
        const series = await fetchFmpSeries(asset, cutoffDate, endDate);
        assetSeries.push(series);
        console.log(`[${asset.symbol}] rows after cutoff = ${series.dates.length}`);
      }
      mergedSeries = mergeSeriesWithHistory(assetSeries, historical, cutoffDate);
    }

    const aligned = alignSeries(mergedSeries);
    const returns = computeReturns(aligned);
    const output = buildOutput(aligned, returns);
    await writeOutput(output);
  } catch (error) {
    console.error(error);
    await emitNoDataWithReason('Failed to generate dataset from available sources. emitting no_data placeholder.', error);
  }
}

function computeEndDate() {
  return new Date().toISOString().slice(0, 10);
}

async function fetchFmpSeries(asset, startDate, endDate) {
  const fmpSymbol = asset.fmpSymbol || sanitizeFmpSymbol(asset.symbol);
  const url = buildFmpUrl(fmpSymbol, startDate, endDate);
  const json = await fetchJson(url);
  const rows = Array.isArray(json?.historical) ? json.historical : [];
  if (rows.length === 0) {
    throw buildFmpError(json, asset.symbol);
  }

  const filtered = rows
    .filter((row) => typeof row?.date === 'string')
    .filter((row) => (!startDate || row.date >= startDate) && (!endDate || row.date <= endDate))
    .map((row) => ({
      date: row.date,
      price: safeNumber(row.close ?? row.adjClose ?? row.price),
    }))
    .filter((row) => row.price != null)
    .sort((a, b) => a.date.localeCompare(b.date));

  if (filtered.length < 2) {
    throw new Error(`${asset.symbol}: insufficient samples from FMP after filtering.`);
  }

  return {
    symbol: asset.symbol,
    label: asset.label,
    category: asset.category,
    dates: filtered.map((row) => row.date),
    prices: filtered.map((row) => row.price),
  };
}

function buildFmpUrl(symbol, startDate, endDate) {
  const url = new URL(`https://financialmodelingprep.com/api/v3/historical-price-full/${encodeURIComponent(symbol)}`);
  url.searchParams.set('apikey', API_KEY || '');
  url.searchParams.set('serietype', 'line');
  if (startDate) {
    url.searchParams.set('from', startDate);
  }
  if (endDate) {
    url.searchParams.set('to', endDate);
  }
  return url;
}

function sanitizeFmpSymbol(symbol) {
  return symbol.replace(/[^A-Za-z0-9]/g, '');
}

function safeNumber(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: { 'User-Agent': 'market-stability-dashboard/1.0 (+https://github.com/)' },
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url.toString()}: ${response.status}`);
  }
  const json = await response.json();
  return json;
}

function buildFmpError(payload, symbol) {
  if (payload && typeof payload.Note === 'string') {
    return new Error(`${symbol}: FMP returned a notice - ${payload.Note}`);
  }
  if (payload && typeof payload['Error Message'] === 'string') {
    return new Error(`${symbol}: FMP error - ${payload['Error Message']}`);
  }
  return new Error(`${symbol}: FMP response did not contain historical data.`);
}

function buildOutput(aligned, returns) {
  const windows = {};
  WINDOWS.forEach((window) => {
    windows[window] = computeWindowMetrics(window, returns, aligned);
  });

  return {
    status: 'ok',
    generatedAt: new Date().toISOString(),
    analysisDates: returns.dates,
    normalizedPrices: returns.normalizedPrices,
    priceSeries: returns.priceSeries,
    priceSeriesSource: 'actual',
    assets: ASSETS.map(({ symbol, label, category }) => ({ symbol, label, category })),
    windows,
  };
}

async function writeOutput(output) {
  await fs.mkdir(DATA_DIR, { recursive: true });
  await fs.writeFile(PRECOMPUTED_PATH, JSON.stringify(output));
  console.log(`Wrote ${PRECOMPUTED_PATH}`);
}

function computeWindowMetrics(window, returns, aligned) {
  const allSymbols = ASSETS.map((asset) => asset.symbol);
  const signalSymbols = SIGNAL.symbols;
  const categories = aligned.categories;
  const records = [];
  const stabilityValues = [];

  for (let endIndex = window - 1; endIndex < returns.dates.length; endIndex += 1) {
    const startIndex = endIndex - window + 1;
    const fullMatrix = buildCorrelationMatrix(allSymbols, returns.returns, startIndex, endIndex);
    const signalMatrix = buildCorrelationMatrix(signalSymbols, returns.returns, startIndex, endIndex);
    const stability = computeStability(signalMatrix, signalSymbols, categories);
    const sub = computeSubIndices(signalMatrix, signalSymbols, categories);

    records.push({
      date: returns.dates[endIndex],
      stability,
      sub,
      matrix: signalMatrix,
      fullMatrix,
      smoothed: stability,
      delta: 0,
    });
    stabilityValues.push(stability);
  }

  if (records.length === 0) {
    throw new Error(`Window ${window}: insufficient samples`);
  }

  const smoothedSeries = ema(stabilityValues, 10);
  const shortEma = ema(stabilityValues, 3);
  const longEma = ema(stabilityValues, 10);

  records.forEach((record, index) => {
    record.smoothed = smoothedSeries[index];
    record.delta = shortEma[index] - longEma[index];
  });

  const pairs = buildPairSeries(
    records,
    window,
    aligned.prices,
    allSymbols,
  );

  return {
    records,
    average180: mean(stabilityValues.slice(-180)),
    latest: records[records.length - 1],
    pairs,
  };
}

function buildPairSeries(records, window, alignedPrices, symbols) {
  const pairs = {};
  const priceOffset = window - 1;
  for (let i = 0; i < symbols.length; i += 1) {
    for (let j = i + 1; j < symbols.length; j += 1) {
      const key = `${symbols[i]}|${symbols[j]}`;
      pairs[key] = { dates: [], correlation: [], priceA: [], priceB: [] };
    }
  }

  records.forEach((record, idx) => {
    const matrix = Array.isArray(record.fullMatrix) ? record.fullMatrix : record.matrix;
    const returnsIndex = priceOffset + idx;
    const alignedIndex = returnsIndex + 1;
    for (let i = 0; i < symbols.length; i += 1) {
      for (let j = i + 1; j < symbols.length; j += 1) {
        const key = `${symbols[i]}|${symbols[j]}`;
        const pair = pairs[key];
        pair.dates.push(record.date);
        const corrValue = Array.isArray(matrix) && matrix[i] ? matrix[i][j] : null;
        pair.correlation.push(Number.isFinite(corrValue) ? corrValue : null);
        const seriesA = Array.isArray(alignedPrices?.[symbols[i]])
          ? alignedPrices[symbols[i]]
          : [];
        const seriesB = Array.isArray(alignedPrices?.[symbols[j]])
          ? alignedPrices[symbols[j]]
          : [];

        const valueA = seriesA?.[alignedIndex];
        const valueB = seriesB?.[alignedIndex];

        pair.priceA.push(Number.isFinite(valueA) ? valueA : null);
        pair.priceB.push(Number.isFinite(valueB) ? valueB : null);
      }
    }
  });

  return pairs;
}

function computeCutoffDate() {
  return MINIMUM_CUTOFF_DATE;
}

async function readHistoricalSeries() {
  try {
    const raw = await fs.readFile(HISTORICAL_PATH, 'utf8');
    const json = JSON.parse(raw);
    const assets = Array.isArray(json?.assets) ? json.assets : [];
    const map = new Map();
    assets.forEach((asset) => {
      if (!asset || typeof asset.symbol !== 'string') {
        return;
      }
      const dates = Array.isArray(asset.dates) ? asset.dates : [];
      const prices = Array.isArray(asset.prices) ? asset.prices : [];
      if (dates.length === 0 || dates.length !== prices.length) {
        return;
      }
      map.set(asset.symbol, {
        symbol: asset.symbol,
        label: asset.label,
        category: asset.category,
        dates,
        prices,
      });
    });

    if (map.size === 0) {
      console.warn(`[generate-data] ${HISTORICAL_PATH} contained no usable assets.`);
      return null;
    }

    console.log(`[generate-data] loaded historical cache (${map.size} assets) from ${HISTORICAL_PATH}`);
    return {
      map,
      startDate: json.startDate,
      endDate: json.endDate,
      generatedAt: json.generatedAt,
    };
  } catch (error) {
    if (error?.code === 'ENOENT') {
      console.warn(`[generate-data] ${HISTORICAL_PATH} not found; continuing without cached history.`);
      return null;
    }
    console.warn(`[generate-data] failed to read ${HISTORICAL_PATH}: ${error?.message || error}`);
    return null;
  }
}

function mergeSeriesWithHistory(latestSeries, historical, cutoffDate) {
  if (!historical || !(historical.map instanceof Map)) {
    return latestSeries;
  }

  const cutoff = cutoffDate || DATA_START_DATE;
  const merged = latestSeries.map((series) => {
    const history = historical.map.get(series.symbol);
    if (!history) {
      const filtered = trimSeries(series, (date) => !cutoff || date >= cutoff);
      if (filtered.dates.length === series.dates.length) {
        return series;
      }
      return {
        ...series,
        dates: filtered.dates,
        prices: filtered.prices,
      };
    }
    const after = trimSeries(series, (date) => !cutoff || date >= cutoff);
    const firstAfterDate = after.dates[0];
    const before = trimSeries(history, (date) => {
      if (firstAfterDate) {
        return date < firstAfterDate;
      }
      return !cutoff || date < cutoff;
    });
    if (before.dates.length === 0) {
      return { ...series, dates: after.dates, prices: after.prices };
    }
    if (after.dates.length === 0) {
      return { ...series, dates: before.dates, prices: before.prices };
    }
    const boundary = firstAfterDate || cutoff || 'live-data';
    console.log(`[generate-data] ${series.symbol}: prepended ${before.dates.length} cached rows before ${boundary}`);
    return {
      ...series,
      dates: before.dates.concat(after.dates),
      prices: before.prices.concat(after.prices),
    };
  });

  historical.map.forEach((history, symbol) => {
    if (merged.some((entry) => entry.symbol === symbol)) {
      return;
    }
    console.warn(`[generate-data] missing live data for ${symbol}; using cached history only.`);
    merged.push({
      symbol,
      label: history.label || findAssetLabel(symbol),
      category: history.category || findAssetCategory(symbol),
      dates: history.dates.slice(),
      prices: history.prices.slice(),
    });
  });

  return merged;
}

function buildSeriesFromHistory(historical, cutoffDate) {
  const merged = [];
  const cutoff = cutoffDate || DATA_START_DATE;
  ASSETS.forEach((asset) => {
    const cached = historical.map.get(asset.symbol);
    if (!cached) {
      throw new Error(`Historical dataset missing ${asset.symbol}; cannot build payload.`);
    }
    const trimmed = trimSeries(cached, (date) => !cutoff || date >= cutoff);
    if (trimmed.dates.length < 2) {
      throw new Error(`${asset.symbol}: insufficient historical samples after cutoff ${cutoff}.`);
    }
    merged.push({
      symbol: asset.symbol,
      label: asset.label,
      category: asset.category,
      dates: trimmed.dates,
      prices: trimmed.prices,
    });
  });
  return merged;
}

function trimSeries(series, predicate) {
  const dates = [];
  const prices = [];
  if (!series) {
    return { dates, prices };
  }

  const length = Math.min(
    Array.isArray(series.dates) ? series.dates.length : 0,
    Array.isArray(series.prices) ? series.prices.length : 0,
  );

  for (let index = 0; index < length; index += 1) {
    const date = series.dates[index];
    const price = series.prices[index];
    if (typeof date !== 'string' || !Number.isFinite(price)) {
      continue;
    }
    if (!predicate(date)) {
      continue;
    }
    dates.push(date);
    prices.push(price);
  }

  return { dates, prices };
}

function findAssetLabel(symbol) {
  return ASSETS.find((asset) => asset.symbol === symbol)?.label || symbol;
}

function findAssetCategory(symbol) {
  return ASSETS.find((asset) => asset.symbol === symbol)?.category || 'unknown';
}

async function emitNoDataPlaceholder() {
  await fs.mkdir(DATA_DIR, { recursive: true });
  const payload = {
    status: 'no_data',
    generatedAt: new Date().toISOString(),
  };
  await fs.writeFile(PRECOMPUTED_PATH, `${JSON.stringify(payload, null, 2)}\n`);
  console.warn(`[generate-data] wrote no_data placeholder to ${PRECOMPUTED_PATH}`);
}

async function emitNoDataWithReason(message, error) {
  console.warn(`[generate-data] ${message}`);
  if (error) {
    const detail = error instanceof Error ? error.message : String(error);
    console.warn(`[generate-data] detail: ${detail}`);
  }
  await emitNoDataPlaceholder();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
