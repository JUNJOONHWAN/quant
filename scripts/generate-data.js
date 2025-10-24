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
if (!API_KEY) {
  console.error('error: ALPHAVANTAGE_API_KEY environment variable is not set.');
  process.exit(1);
}

if (typeof fetch !== 'function') {
  console.error('error: global fetch is not available. Please run the generator with Node.js 18 or newer.');
  process.exit(1);
}

const ASSETS = [
  { symbol: 'QQQ', label: 'QQQ (NASDAQ 100 ETF)', category: 'stock', source: 'TIME_SERIES_DAILY_ADJUSTED' },
  { symbol: 'SPY', label: 'SPY (S&P 500 ETF)', category: 'stock', source: 'TIME_SERIES_DAILY_ADJUSTED' },
  { symbol: 'TLT', label: 'TLT (미국 장기채)', category: 'bond', source: 'TIME_SERIES_DAILY_ADJUSTED' },
  { symbol: 'GLD', label: 'GLD (금 ETF)', category: 'gold', source: 'TIME_SERIES_DAILY_ADJUSTED' },
  { symbol: 'BTC-USD', label: 'BTC-USD (비트코인)', category: 'crypto', source: 'DIGITAL_CURRENCY_DAILY' },
];

const WINDOWS = [20, 30, 60];
const RANGE_YEARS = 5;
const ONE_MINUTE = 60 * 1000;

async function main() {
  const assetSeries = [];
  const cutoffDate = computeCutoffDate(RANGE_YEARS);

  for (let index = 0; index < ASSETS.length; index += 1) {
    const asset = ASSETS[index];
    console.log(`Fetching ${asset.symbol} via Alpha Vantage (${asset.source})`);
    const series = await fetchAlphaSeries(asset, cutoffDate);
    assetSeries.push(series);

    if (index < ASSETS.length - 1) {
      console.log('Waiting to respect Alpha Vantage rate limits...');
      await delay(ONE_MINUTE / 5 + 1000); // ~13 seconds between calls (5/min limit)
    }
  }

  const aligned = alignSeries(assetSeries);
  const returns = computeReturns(aligned);
  const output = buildOutput(aligned, returns);

  const targetDir = path.join(__dirname, '..', 'static_site', 'data');
  await fs.mkdir(targetDir, { recursive: true });
  const outputPath = path.join(targetDir, 'precomputed.json');
  await fs.writeFile(outputPath, JSON.stringify(output));
  console.log(`Wrote ${outputPath}`);
}

function computeCutoffDate(years) {
  const cutoff = new Date();
  cutoff.setUTCDate(cutoff.getUTCDate() + 1); // ensure today included
  cutoff.setUTCFullYear(cutoff.getUTCFullYear() - years);
  return cutoff.toISOString().slice(0, 10);
}

async function fetchAlphaSeries(asset, cutoffDate) {
  if (asset.source === 'DIGITAL_CURRENCY_DAILY') {
    return fetchDigital(asset, cutoffDate);
  }
  return fetchEquity(asset, cutoffDate);
}

async function fetchEquity(asset, cutoffDate) {
  const url = new URL('https://www.alphavantage.co/query');
  url.searchParams.set('function', 'TIME_SERIES_DAILY_ADJUSTED');
  url.searchParams.set('symbol', asset.symbol);
  url.searchParams.set('outputsize', 'full');
  url.searchParams.set('apikey', API_KEY);

  const json = await fetchJson(url);
  const series = json['Time Series (Daily)'];
  if (!series) {
    throw buildAlphaError(json, asset.symbol);
  }
  return normalizeSeries(asset, series, cutoffDate, (value) => Number(value['5. adjusted close'] || value['4. close']));
}

async function fetchDigital(asset, cutoffDate) {
  const url = new URL('https://www.alphavantage.co/query');
  url.searchParams.set('function', 'DIGITAL_CURRENCY_DAILY');
  url.searchParams.set('symbol', asset.symbol.split('-')[0]);
  url.searchParams.set('market', 'USD');
  url.searchParams.set('apikey', API_KEY);

  const json = await fetchJson(url);
  const series = json['Time Series (Digital Currency Daily)'];
  if (!series) {
    throw buildAlphaError(json, asset.symbol);
  }
  return normalizeSeries(asset, series, cutoffDate, (value) => Number(value['4a. close (USD)'] || value['4b. close (USD)']));
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

function buildOutput(aligned, returns) {
  const windows = {};
  WINDOWS.forEach((window) => {
    windows[window] = computeWindowMetrics(window, returns, aligned);
  });

  return {
    generatedAt: new Date().toISOString(),
    analysisDates: returns.dates,
    normalizedPrices: returns.normalizedPrices,
    assets: ASSETS.map(({ symbol, label, category }) => ({ symbol, label, category })),
    windows,
  };
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

  const pairs = buildPairSeries(records, window, returns.normalizedPrices, symbols);

  return {
    records,
    average180: mean(stabilityValues.slice(-180)),
    latest: records[records.length - 1],
    pairs,
  };
}

function buildPairSeries(records, window, normalizedPrices, symbols) {
  const pairs = {};
  const priceOffset = window - 1;
  for (let i = 0; i < symbols.length; i += 1) {
    for (let j = i + 1; j < symbols.length; j += 1) {
      const key = `${symbols[i]}|${symbols[j]}`;
      pairs[key] = { dates: [], correlation: [], priceA: [], priceB: [] };
    }
  }

  records.forEach((record, idx) => {
    const priceIndex = priceOffset + idx;
    for (let i = 0; i < symbols.length; i += 1) {
      for (let j = i + 1; j < symbols.length; j += 1) {
        const key = `${symbols[i]}|${symbols[j]}`;
        const pair = pairs[key];
        pair.dates.push(record.date);
        pair.correlation.push(record.matrix[i][j]);
        pair.priceA.push(normalizedPrices[symbols[i]][priceIndex]);
        pair.priceB.push(normalizedPrices[symbols[j]][priceIndex]);
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

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
