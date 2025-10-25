#!/usr/bin/env node
/*
 * Generate precomputed market stability metrics using Alpha Vantage data.
 * The script writes a JSON blob that the GitHub Pages frontend consumes.
 */
const fs = require('fs/promises');
const path = require('path');

const {
  alignSeries,
  computeReturns,
  buildCorrelationMatrix,
  computeStability,
  computeSubIndices,
  ema,
  mean,
} = require('../static_site/assets/metrics');

const API_KEY = process.env.ALPHAVANTAGE_API_KEY;
const DATA_DIR = path.join(__dirname, '..', 'static_site', 'data');
const PRECOMPUTED_PATH = path.join(DATA_DIR, 'precomputed.json');

const ASSETS = [
  { symbol: 'QQQ', label: 'QQQ (NASDAQ 100 ETF)', category: 'stock', source: 'TIME_SERIES_DAILY' },
  { symbol: 'SPY', label: 'SPY (S&P 500 ETF)', category: 'stock', source: 'TIME_SERIES_DAILY' },
  { symbol: 'TLT', label: 'TLT (미국 장기채)', category: 'bond', source: 'TIME_SERIES_DAILY' },
  { symbol: 'GLD', label: 'GLD (금 ETF)', category: 'gold', source: 'TIME_SERIES_DAILY' },
  { symbol: 'BTC-USD', label: 'BTC-USD (비트코인)', category: 'crypto', source: 'DIGITAL_CURRENCY_DAILY' },
];

const WINDOWS = [20, 30, 60];
const RANGE_YEARS = 5;
const MINIMUM_CUTOFF_DATE = '2020-01-01';
const REQUEST_DELAY_MS = 15000;

async function main() {
  if (typeof fetch !== 'function') {
    await emitNoDataWithReason('global fetch is not available. emitting no_data placeholder.');
    return;
  }

  if (!API_KEY) {
    console.warn('[generate-data] no API key -> emit no_data placeholder');
    await emitNoDataPlaceholder();
    return;
  }

  try {
    const assetSeries = [];
    const cutoffDate = computeCutoffDate(RANGE_YEARS);

    for (let index = 0; index < ASSETS.length; index += 1) {
      const asset = ASSETS[index];
      console.log(`Fetching ${asset.symbol} via Alpha Vantage (${asset.source})`);
      const series = await fetchAlphaSeries(asset, cutoffDate);
      assetSeries.push(series);
      console.log(`[${asset.symbol}] rows after cutoff = ${series.dates.length}`);

      if (index < ASSETS.length - 1) {
        console.log('Waiting to respect Alpha Vantage rate limits...');
        await delay(REQUEST_DELAY_MS);
      }
    }

    const aligned = alignSeries(assetSeries);
    const returns = computeReturns(aligned);
    const output = buildOutput(aligned, returns);
    await writeOutput(output);
  } catch (error) {
    console.error(error);
    await emitNoDataWithReason('Failed to generate dataset from Alpha Vantage. emitting no_data placeholder.', error);
  }
}

function computeCutoffDate() {
  return MINIMUM_CUTOFF_DATE;
}

async function fetchAlphaSeries(asset, cutoffDate) {
  if (asset.source === 'DIGITAL_CURRENCY_DAILY') {
    return fetchDigital(asset, cutoffDate);
  }
  return fetchEquity(asset, cutoffDate);
}

async function fetchEquity(asset, cutoffDate) {
  const url = buildAlphaUrl({
    function: 'TIME_SERIES_DAILY',
    symbol: asset.symbol,
    outputsize: 'full',
  });

  const json = await fetchJson(url);
  const series = extractEquitySeries(json);

  if (!series) {
    throw buildAlphaError(json, asset.symbol);
  }

  return normalizeSeries(asset, series, cutoffDate, (value) => Number(value?.['4. close']));
}

async function fetchDigital(asset, cutoffDate) {
  const { base, market } = parseCryptoSymbol(asset.symbol);
  const url = buildAlphaUrl({
    function: 'DIGITAL_CURRENCY_DAILY',
    symbol: base,
    market,
  });

  const json = await fetchJson(url);
  const series = json['Time Series (Digital Currency Daily)'];
  if (!series) {
    throw buildAlphaError(json, asset.symbol);
  }

  const primaryCloseKey = `4a. close (${market})`;
  const secondaryCloseKey = `4b. close (${market})`;
  const legacyCloseKey = '4. close';
  const dates = Object.keys(series)
    .filter((date) => date >= cutoffDate)
    .sort();
  if (dates.length === 0) {
    throw new Error(`${asset.symbol}: no samples after ${cutoffDate}`);
  }

  const filteredDates = [];
  const prices = [];
  dates.forEach((date) => {
    const value = series[date];
    const close = Number(
      value?.[primaryCloseKey]
        ?? value?.[secondaryCloseKey]
        ?? value?.[legacyCloseKey]
    );
    if (Number.isFinite(close)) {
      filteredDates.push(date);
      prices.push(close);
    }
  });

  if (filteredDates.length < 2) {
    console.warn(`${asset.symbol}: short series ${filteredDates.length}d after cutoff ${cutoffDate}`);
  }

  return {
    symbol: asset.symbol,
    label: asset.label,
    category: asset.category,
    dates: filteredDates,
    prices,
  };
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

function normalizeSeries(asset, rawSeries, cutoffDate, extractClose) {
  const dates = Object.keys(rawSeries)
    .filter((date) => date >= cutoffDate)
    .sort();
  if (dates.length === 0) {
    throw new Error(`${asset.symbol}: no samples after ${cutoffDate}`);
  }

  const filteredDates = [];
  const prices = [];
  dates.forEach((date) => {
    const close = extractClose(rawSeries[date]);
    if (Number.isFinite(close)) {
      filteredDates.push(date);
      prices.push(close);
    }
  });

  if (filteredDates.length < 2) {
    throw new Error(`${asset.symbol}: insufficient samples after filtering`);
  }

  return {
    symbol: asset.symbol,
    label: asset.label,
    category: asset.category,
    dates: filteredDates,
    prices,
  };
}

function buildAlphaUrl(params) {
  const url = new URL('https://www.alphavantage.co/query');
  Object.entries(params).forEach(([key, value]) => {
    url.searchParams.set(key, value);
  });
  url.searchParams.set('apikey', API_KEY);
  return url;
}

function extractEquitySeries(payload) {
  return payload?.['Time Series (Daily)'];
}

function parseCryptoSymbol(symbol) {
  const [base, quote = 'USD'] = symbol.split('-');
  return { base, market: quote };
}

function buildOutput(aligned, returns) {
  const windows = {};
  WINDOWS.forEach((window) => {
    windows[window] = computeWindowMetrics(window, returns, aligned);
  });

  return {
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
  const symbols = ASSETS.map((asset) => asset.symbol);
  const categories = aligned.categories;
  const records = [];
  const stabilityValues = [];

  for (let endIndex = window - 1; endIndex < returns.dates.length; endIndex += 1) {
    const startIndex = endIndex - window + 1;
    const matrix = buildCorrelationMatrix(symbols, returns.returns, startIndex, endIndex);
    const stability = computeStability(matrix, symbols, categories);
    const sub = computeSubIndices(matrix, symbols, categories);

    records.push({
      date: returns.dates[endIndex],
      stability,
      sub,
      matrix,
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
    symbols,
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
    const returnsIndex = priceOffset + idx;
    const alignedIndex = returnsIndex + 1;
    for (let i = 0; i < symbols.length; i += 1) {
      for (let j = i + 1; j < symbols.length; j += 1) {
        const key = `${symbols[i]}|${symbols[j]}`;
        const pair = pairs[key];
        pair.dates.push(record.date);
        pair.correlation.push(record.matrix[i][j]);
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

function buildAlphaError(payload, symbol) {
  const message = payload?.Note || payload?.['Error Message'] || JSON.stringify(payload);
  return new Error(`${symbol}: Alpha Vantage returned an error: ${message}`);
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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
