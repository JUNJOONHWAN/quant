#!/usr/bin/env node
/*
 * Export daily backtest rows for the FLL-Fusion regime (delay1 open execution)
 * using the exact same logic as the frontend dashboard.
 *
 * Usage:
 *   node scripts/export_fll_fusion_csv.js \
 *     --start 2022-01-03 --end 2025-10-31 \
 *     --out docs/backtest_fll_fusion_open_delay1.csv
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const args = process.argv.slice(2);
function argValue(flag, fallback) {
  const idx = args.indexOf(flag);
  return idx >= 0 && args[idx + 1] ? args[idx + 1] : fallback;
}

const START = argValue('--start', '2022-01-03');
const END = argValue('--end', '2025-10-31');
const OUTPUT = argValue('--out', path.join('docs', 'backtest_fll_fusion_open_delay1.csv'));

const metrics = require(path.join(__dirname, '..', 'static_site', 'assets', 'metrics.js'));

const context = {
  window: { location: { hostname: 'localhost' }, MarketMetrics: metrics },
  document: {
    readyState: 'loading',
    addEventListener: () => {},
    getElementById: () => null,
    querySelector: () => null,
  },
  console,
  setTimeout,
  clearTimeout,
  URL: {
    createObjectURL: () => 'blob:stub',
    revokeObjectURL: () => {},
  },
  Blob: function () {},
};
context.window.document = context.document;
vm.createContext(context);

const appCode = fs.readFileSync(path.join(__dirname, '..', 'static_site', 'assets', 'app.js'), 'utf8');
vm.runInContext(appCode, context);
vm.runInContext(`state.window = 30;`, context);
vm.runInContext(`state.customRange = { start: '${START}', end: '${END}', valid: true };`, context);
vm.runInContext('state.range = 180;', context);
vm.runInContext("state.riskMode = 'fll_fusion';", context);

const precomputedPath = path.join(__dirname, '..', 'static_site', 'data', 'precomputed.json');
const precomputed = JSON.parse(fs.readFileSync(precomputedPath, 'utf8'));
context.hydrateFromPrecomputed(precomputed);

const metrics30 = vm.runInContext('state.metrics[state.window]', context);
const filtered = context.getFilteredRecords(metrics30);
if (!filtered || filtered.empty) {
  throw new Error('Filtered records unavailable; check start/end dates.');
}

vm.runInContext(`state.customRange = { start: '${START}', end: '${END}', valid: true };`, context);

const payload = vm.runInContext('buildBacktestCsv()', context);
if (!payload || typeof payload.csv !== 'string') {
  throw new Error('Failed to build CSV payload via app.js helpers.');
}

fs.writeFileSync(OUTPUT, payload.csv);
const lines = payload.csv.trim().split(/\r?\n/);
const rowCount = Math.max(0, lines.length - 1);
console.log(`Exported ${rowCount} rows to ${OUTPUT}`);
