const DEFAULT_WINDOW = 30;
const WINDOWS = [20, 30, 60];
const RANGE_OPTIONS = [30, 60, 90, 180];
const BANDS = { red: [0, 0.3], yellow: [0.3, 0.4], green: [0.4, 1.0] };
const DEFAULT_PAIR = 'IWM|BTC-USD';
const MAX_STALE_DAYS = 7;
const MS_PER_DAY = 24 * 60 * 60 * 1000;
const FMP_DATA_START = '2017-01-01';
const FMP_RATE_DELAY = 500;
const ACTIOND_WAIT_MS = 5000;
const ACTIOND_POLL_INTERVAL_MS = 75;
const IS_LOCAL = ['localhost', '127.0.0.1', '::1'].includes(window.location.hostname);
const DEFAULT_DATA_PATH = './data/precomputed.json';
const RISK_MODE_STORAGE_KEY = 'risk-mode.v1';
const TEXT_PRIMARY = '#f8fafc';
const TEXT_AXIS = '#cbd5ff';

let datasetPath = DEFAULT_DATA_PATH;
let cacheTagRaw = null;
let versionManifest = null;

function createBearDamp(prices, symbols = ['QQQ', 'IWM', 'BTC-USD']) {
  const riskSet = symbols.filter((symbol) => Array.isArray(prices?.[symbol]) && prices[symbol].length > 0);
  if (riskSet.length === 0) {
    return () => 1;
  }

  const span60Alpha = 2 / (60 + 1);
  const EPS = 1e-12;
  const maxLen = riskSet.reduce((acc, symbol) => Math.max(acc, prices[symbol]?.length || 0), 0);
  const RStar = new Array(maxLen).fill(null);

  riskSet.forEach((symbol) => {
    const series = prices[symbol] || [];
    if (!Array.isArray(series) || series.length === 0) return;
    let posEwma = null;
    let negEwma = null;
    for (let idx = 1; idx < series.length; idx += 1) {
      const prev = safeNumber(series[idx - 1], NaN);
      const curr = safeNumber(series[idx], NaN);
      if (!Number.isFinite(prev) || !Number.isFinite(curr) || prev <= 0 || curr <= 0) continue;
      const ret = Math.log(curr) - Math.log(prev);
      if (!Number.isFinite(ret)) continue;
      const posSq = ret > 0 ? ret * ret : 0;
      const negSq = ret < 0 ? ret * ret : 0;
      posEwma = posEwma == null ? posSq : (span60Alpha * posSq) + ((1 - span60Alpha) * posEwma);
      negEwma = negEwma == null ? negSq : (span60Alpha * negSq) + ((1 - span60Alpha) * negEwma);
      const denom = (posEwma ?? 0) + EPS;
      const ratioRaw = denom > 0 ? (negEwma ?? 0) / denom : 0;
      if (!Number.isFinite(ratioRaw)) continue;
      const ratio = Math.max(0, Math.min(5, ratioRaw));
      const existing = RStar[idx];
      RStar[idx] = Number.isFinite(existing) ? Math.max(existing, ratio) : ratio;
    }
  });

  for (let idx = 0; idx < RStar.length; idx += 1) {
    if (!Number.isFinite(RStar[idx])) RStar[idx] = 1;
  }

  return function bearDamp(idx, RThr = 1.2, beta = 0.5, gmin = 0.5) {
    if (!Number.isInteger(idx) || idx < 0 || idx >= RStar.length) return 1;
    const r = Number.isFinite(RStar[idx]) ? RStar[idx] : 1;
    const x = Math.max(0, r - RThr);
    return Math.max(gmin, 1 / (1 + beta * x));
  };
}

function setDataInfo(message, variant = 'notice') {
  const element = document.getElementById('data-info');
  if (!element) {
    return;
  }

  element.textContent = message;
  element.classList.remove('hidden');
  element.classList.remove('notice', 'error');
  element.classList.add(variant === 'error' ? 'error' : 'notice');
}

function formatByteSize(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return '알 수 없음';
  }

  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }

  const display = unitIndex === 0 ? size : size.toFixed(1);
  return `${display} ${units[unitIndex]}`;
}

function resolveActiondEnvironment(name) {
  const globalCandidates = [
    window.__ACTIOND_ENV__,
    window.__ACTIOND_ENV,
    window.__ACTIOND_SECRETS__,
    window.__ACTIOND_SECRETS,
    window.__ACTIOND_VARS__,
    window.__ACTIONS_RUNTIME_CONFIG__?.env,
    window.__ACTIONS_RUNTIME_CONFIG__?.secrets,
    window.ACTIOND_ENV,
    window.actiond?.env,
    window.actiond?.secrets,
    window.actiond,
    window.__ENV__,
    window.__ENV,
    window.__APP_ENV__,
    window.__RUNTIME_ENV__,
    window.env,
    window.process?.env,
  ];

  for (let index = 0; index < globalCandidates.length; index += 1) {
    const candidate = globalCandidates[index];
    if (candidate && typeof candidate === 'object' && candidate[name]) {
      return candidate[name];
    }
  }

  return '';
}

async function fetchActiondSecret(name) {
  const actiond = await ensureActiondReady();
  if (!actiond) {
    return '';
  }

  const methodNames = ['getSecret', 'getEnv', 'get'];
  for (let index = 0; index < methodNames.length; index += 1) {
    const methodName = methodNames[index];
    const method = actiond[methodName];
    if (typeof method !== 'function') {
      continue; // eslint-disable-line no-continue
    }
    try {
      const value = await method.call(actiond, name);
      if (value) {
        return value;
      }
    } catch (error) {
      console.warn(`Failed to read ${name} via actiond.${methodName}`, error);
    }
  }

  return '';
}

async function ensureActiondReady() {
  const deadline = Date.now() + ACTIOND_WAIT_MS;
  let candidate = window.actiond;

  while ((!candidate || typeof candidate !== 'object') && Date.now() < deadline) {
    await delay(ACTIOND_POLL_INTERVAL_MS);
    candidate = window.actiond;
  }

  if (!candidate || typeof candidate !== 'object') {
    return null;
  }

  if (typeof candidate.ready === 'function') {
    try {
      await candidate.ready();
    } catch (error) {
      console.warn('actiond.ready() rejected', error);
    }
  }

  return candidate;
}

async function hydrateFmpKeyFromEnvironment() {
  const direct = resolveActiondEnvironment('FMP_API_KEY');
  if (direct) {
    state.fmpKey = direct;
    return;
  }

  const fetched = await fetchActiondSecret('FMP_API_KEY');
  if (fetched) {
    state.fmpKey = fetched;
  }
}

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
  primaryStock: 'IWM',
  breadth: ['IWM', 'SPY', 'BTC-USD'],
  pairKey: DEFAULT_PAIR,
  trade: {
    baseSymbol: 'QQQ',
    leveredSymbol: 'TQQQ',
    leverage: 3,
  },
};

const REQUIRED_SYMBOLS = Array.from(new Set([...SIGNAL.symbols, SIGNAL.trade.baseSymbol]));

function getInitialRiskMode() {
  if (typeof window === 'undefined') {
    return 'classic';
  }
  try {
    const saved = window.localStorage?.getItem(RISK_MODE_STORAGE_KEY);
    if (saved === 'classic' || saved === 'enhanced' || saved === 'ffl' || saved === 'ffl_exp' || saved === 'ffl_stab' || saved === 'fll_fusion') {
      return saved;
    }
  } catch (error) {
    // ignore storage errors (private mode, etc.)
  }
  return 'classic';
}

const state = {
  window: DEFAULT_WINDOW,
  range: 180,
  metrics: {},
  normalizedPrices: {},
  priceSeries: {},
  priceSeriesOpen: {},
  priceSeriesSource: 'actual',
  analysisDates: [],
  generatedAt: null,
  pair: DEFAULT_PAIR,
  fmpKey: '',
  refreshing: false,
  autoRefreshAttempted: false,
  customRange: {
    start: null,
    end: null,
    valid: true,
  },
  riskMode: getInitialRiskMode(),
  heatmapDate: null,
};

function applyRiskMode(nextMode, { persist = true } = {}) {
  const normalized = (nextMode === 'enhanced') ? 'enhanced'
    : (nextMode === 'ffl' ? 'ffl'
    : (nextMode === 'ffl_exp' ? 'ffl_exp'
    : (nextMode === 'ffl_stab' ? 'ffl_stab'
    : (nextMode === 'fll_fusion' ? 'fll_fusion' : 'classic'))));
  state.riskMode = normalized;
  if (persist) {
    try {
      window.localStorage?.setItem(RISK_MODE_STORAGE_KEY, normalized);
    } catch (error) {
      // ignore private-mode storage failures
    }
  }
  if (typeof document !== 'undefined' && document.documentElement) {
    document.documentElement.dataset.riskMode = normalized;
  }
  return normalized;
}

applyRiskMode(state.riskMode, { persist: false });

function normalizeDatasetPath(candidate) {
  if (!candidate || typeof candidate !== 'string') {
    return DEFAULT_DATA_PATH;
  }
  const trimmed = candidate.trim();
  if (!trimmed) {
    return DEFAULT_DATA_PATH;
  }
  if (trimmed.startsWith('./') || trimmed.startsWith('../') || trimmed.startsWith('/')) {
    return trimmed;
  }
  return `./data/${trimmed}`;
}

function applyBuildInfo(info) {
  if (!info || typeof info !== 'object') {
    return;
  }

  if (typeof info.buildId === 'string' && info.buildId.trim()) {
    cacheTagRaw = info.buildId.trim();
  } else if (typeof info.build === 'string' && info.build.trim()) {
    cacheTagRaw = info.build.trim();
  } else if (typeof info.cacheTag === 'string' && info.cacheTag.trim()) {
    cacheTagRaw = info.cacheTag.trim();
  }

  if (typeof info.dataPath === 'string' && info.dataPath.trim()) {
    datasetPath = normalizeDatasetPath(info.dataPath);
  } else if (typeof info.data === 'string' && info.data.trim()) {
    datasetPath = normalizeDatasetPath(info.data);
  }

  if (info.version && typeof info.version === 'object' && info.version !== info) {
    versionManifest = info.version;
    applyBuildInfo(info.version);
  }
}

async function fetchVersionManifest() {
  const ts = Date.now();
  try {
    const response = await fetch(`./data/version.json?ts=${ts}`, { cache: 'no-store' });
    if (!response.ok) {
      return {};
    }
    const json = await response.json();
    versionManifest = json;
    return json;
  } catch (error) {
    console.warn('버전 매니페스트를 불러오지 못했습니다.', error);
    return {};
  }
}

async function ensureBuildInfo() {
  if (!cacheTagRaw || !datasetPath) {
    if (window.__BUILD_INFO__) {
      applyBuildInfo(window.__BUILD_INFO__);
    }

    if (!cacheTagRaw || !datasetPath) {
      const manifest = await fetchVersionManifest();
      if (manifest) {
        applyBuildInfo(manifest);
      }
    }
  }

  if (!cacheTagRaw) {
    cacheTagRaw = `${Date.now()}`;
  }
  if (!datasetPath) {
    datasetPath = DEFAULT_DATA_PATH;
  }
}

function getDatasetPath() {
  return datasetPath || DEFAULT_DATA_PATH;
}

const charts = {};
let renderScheduled = false;

function scheduleRender() {
  if (renderScheduled) return;
  renderScheduled = true;
  setTimeout(() => {
    renderScheduled = false;
    renderAll();
  }, 50);
}

function renderImmediately() {
  renderScheduled = false;
  renderAll();
}

const {
  alignSeries,
  computeReturns: computeReturnsForState,
  buildCorrelationMatrix,
  computeStability,
  computeSubIndices,
  averagePairs,
  ema,
  mean,
  toReturns,
  corr,
  topEigenvalue,
} = window.MarketMetrics;

// Ensure init runs even when this script is loaded after DOMContentLoaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  // DOM is already ready (interactive or complete), run immediately
  init();
}

async function init() {
  showError('데이터를 불러오는 중입니다...');
  state.metrics = {};
  await hydrateFmpKeyFromEnvironment();
  try {
    const precomputed = await loadPrecomputed();
    if (!precomputed) {
      hideError();
      return;
    }

    hydrateFromPrecomputed(precomputed);
    hideError();
    populateControls();
    renderAll();
    await maybeAutoRefreshAfterLoad();
  } catch (error) {
    console.error(error);
    showError('데이터를 불러오지 못했습니다. 네트워크 상태를 확인하거나 데이터를 다시 생성해 주세요.');
  }
}

async function loadFromYahoo() {
  const assetSeries = await Promise.all(ASSETS.map(fetchYahooSeries));
  const aligned = alignSeries(assetSeries);
  const returns = computeReturnsForState(aligned);
  state.analysisDates = returns.dates;
  state.generatedAt = new Date();
  state.normalizedPrices = returns.normalizedPrices;
  state.priceSeries = returns.priceSeries;
  state.priceSeriesOpen = cloneSeriesMap(returns.priceSeries);
  state.priceSeriesSource = 'actual';
  computeAllMetrics(returns, aligned);
  maybeAlignDatesToCurrent();
}

async function loadFromFmp(apiKey) {
  const effectiveCutoff = computeFmpCutoffDate();
  const endDate = new Date().toISOString().slice(0, 10);
  const assetSeries = [];

  for (let index = 0; index < ASSETS.length; index += 1) {
    const asset = ASSETS[index];
    const series = await fetchFmpSeriesBrowser(asset, apiKey, effectiveCutoff, endDate);
    assetSeries.push(series);
    if (index < ASSETS.length - 1) {
      await delay(FMP_RATE_DELAY);
    }
  }

  const aligned = alignSeries(assetSeries);
  const returns = computeReturnsForState(aligned);
  state.analysisDates = returns.dates;
  state.generatedAt = new Date();
  state.normalizedPrices = returns.normalizedPrices;
  state.priceSeries = returns.priceSeries;
  const openSeries = buildAlignedOpenSeries(aligned.dates, assetSeries);
  Object.keys(returns.priceSeries).forEach((symbol) => {
    const closes = Array.isArray(returns.priceSeries[symbol]) ? returns.priceSeries[symbol] : [];
    const opens = Array.isArray(openSeries[symbol]) ? openSeries[symbol] : [];
    if (!Array.isArray(openSeries[symbol]) || openSeries[symbol].length !== closes.length) {
      openSeries[symbol] = closes.slice();
    } else {
      openSeries[symbol] = opens.map((value, index) => {
        const closeValue = closes[index];
        return Number.isFinite(value) ? value : closeValue;
      });
    }
  });
  state.priceSeriesOpen = openSeries;
  state.priceSeriesSource = 'actual';
  computeAllMetrics(returns, aligned);
}

function buildNormalizedPricesFromSeries(seriesMap) {
  if (!seriesMap || typeof seriesMap !== 'object') {
    return {};
  }

  const normalize = (values) => {
    if (!Array.isArray(values) || values.length === 0) {
      return [];
    }
    const base = values.find((value) => Number.isFinite(value) && Math.abs(value) > 0);
    if (!Number.isFinite(base) || Math.abs(base) === 0) {
      return values.map(() => null);
    }
    return values.map((value) => (Number.isFinite(value) ? value / base : null));
  };

  return Object.fromEntries(
    Object.entries(seriesMap).map(([symbol, values]) => [symbol, normalize(values)]),
  );
}

function cloneSeriesMap(seriesMap) {
  if (!seriesMap || typeof seriesMap !== 'object') {
    return {};
  }
  return Object.fromEntries(
    Object.entries(seriesMap).map(([symbol, values]) => [symbol, Array.isArray(values) ? values.slice() : []]),
  );
}

function buildAlignedOpenSeries(dates, seriesList) {
  const opens = {};
  if (!Array.isArray(dates)) {
    return opens;
  }

  seriesList.forEach((series) => {
    if (!series || !Array.isArray(series.dates)) {
      return;
    }
    const openMap = new Map();
    const hasOpens = Array.isArray(series.opens);
    const length = series.dates.length;
    for (let index = 0; index < length; index += 1) {
      const date = series.dates[index];
      const closeValue = Array.isArray(series.prices) ? series.prices[index] : null;
      const openValue = hasOpens ? series.opens[index] : null;
      const normalizedOpen = Number.isFinite(openValue) ? openValue : (Number.isFinite(closeValue) ? closeValue : null);
      if (typeof date === 'string' && Number.isFinite(normalizedOpen)) {
        openMap.set(date, normalizedOpen);
      }
    }
    opens[series.symbol] = dates.map((date) => {
      const value = openMap.get(date);
      return Number.isFinite(value) ? value : null;
    });
  });

  return opens;
}

function findMissingSymbols(seriesMap, symbols) {
  if (!Array.isArray(symbols) || symbols.length === 0) {
    return [];
  }
  const missing = [];
  symbols.forEach((symbol) => {
    const series = seriesMap?.[symbol];
    if (!Array.isArray(series) || series.length === 0) {
      missing.push(symbol);
    }
  });
  return missing;
}

function hydrateFromPrecomputed(data) {
  if (!data || typeof data !== 'object') {
    showEmptyState('실제 데이터가 없습니다.');
    return;
  }

  if (!Array.isArray(data.analysisDates) || data.analysisDates.length === 0) {
    showEmptyState('실제 데이터가 없습니다.');
    return;
  }

  state.generatedAt = data.generatedAt ? new Date(data.generatedAt) : new Date();
  state.analysisDates = Array.isArray(data.analysisDates) ? data.analysisDates.slice() : [];
  state.heatmapDate = state.analysisDates.length > 0
    ? state.analysisDates[state.analysisDates.length - 1]
    : null;

  const rawPriceSeries = data.priceSeries && typeof data.priceSeries === 'object' ? data.priceSeries : null;
  if (!rawPriceSeries || Object.keys(rawPriceSeries).length === 0) {
    showEmptyState('priceSeries 누락');
    return;
  }

  const rawPriceSeriesOpen = data.priceSeriesOpen && typeof data.priceSeriesOpen === 'object'
    ? data.priceSeriesOpen
    : null;

  const missingSymbols = findMissingSymbols(rawPriceSeries, REQUIRED_SYMBOLS);
  if (missingSymbols.length > 0) {
    const summary = `필수 자산 데이터 누락: ${missingSymbols.join(', ')}`;
    setDataInfo(summary, 'error');
    showEmptyState(summary, 'static_site/data/precomputed.json을 다시 생성해 주세요.');
    return;
  }

  const priceSeries = cloneSeriesMap(rawPriceSeries);
  const priceSeriesOpen = {};
  const sourceOpen = rawPriceSeriesOpen ? cloneSeriesMap(rawPriceSeriesOpen) : {};
  Object.keys(priceSeries).forEach((symbol) => {
    const opens = sourceOpen[symbol];
    if (Array.isArray(opens) && opens.length === (priceSeries[symbol]?.length || 0)) {
      priceSeriesOpen[symbol] = opens.slice();
    } else {
      priceSeriesOpen[symbol] = Array.isArray(priceSeries[symbol]) ? priceSeries[symbol].slice() : [];
    }
  });
  const normalizedSource =
    data.normalizedPrices && typeof data.normalizedPrices === 'object'
      ? data.normalizedPrices
      : buildNormalizedPricesFromSeries(priceSeries);
  const normalizedPrices = cloneSeriesMap(normalizedSource);

  state.priceSeries = priceSeries;
  state.priceSeriesOpen = priceSeriesOpen;
  state.normalizedPrices = normalizedPrices;
  state.priceSeriesSource = 'actual';
  state.metrics = {};
  if (data.windows && typeof data.windows === 'object') {
    Object.entries(data.windows).forEach(([windowSize, metrics]) => {
      const numericWindow = Number(windowSize);
      state.metrics[numericWindow] = metrics;
    });
  } else {
    // Fallback: compute metrics in-browser from provided priceSeries/analysisDates
    try {
      const symbols = ASSETS.map((a) => a.symbol);
      const dates = Array.isArray(state.analysisDates) ? state.analysisDates.slice() : [];
      // Build returns like metrics.computeReturns would: dates sliced by 1
      const returnsBySymbol = {};
      symbols.forEach((sym) => {
        const prices = Array.isArray(priceSeries?.[sym]) ? priceSeries[sym] : [];
        const arr = [];
        for (let i = 1; i < prices.length; i += 1) {
          const prev = prices[i - 1];
          const cur = prices[i];
          arr.push(Number.isFinite(prev) && Number.isFinite(cur) && prev !== 0 ? Math.log(cur / prev) : 0);
        }
        returnsBySymbol[sym] = arr;
      });
      const returns = {
        dates: dates.slice(1),
        returns: returnsBySymbol,
        normalizedPrices: state.normalizedPrices,
        priceSeries: state.priceSeries,
      };
      const aligned = {
        dates: dates,
        prices: state.priceSeries,
        categories: Object.fromEntries(ASSETS.map((a) => [a.symbol, a.category])),
      };
      computeAllMetrics(returns, aligned);
    } catch (err) {
      console.warn('Fallback metric computation failed', err);
    }
  }
  refreshPairSeriesFromPrices();
  maybeAlignDatesToCurrent();
}

async function loadData() {
  const now = Date.now();
  let manifest = {};

  try {
    const manifestResponse = await fetch(`./data/version.json?ts=${now}`, { cache: 'no-store' });
    if (manifestResponse.ok) {
      manifest = await manifestResponse.json();
    }
  } catch (error) {
    console.warn('버전 매니페스트를 불러오지 못했습니다.', error);
  }

  let dataFile =
    manifest && typeof manifest === 'object' && typeof manifest.data === 'string' && manifest.data.trim()
      ? manifest.data.trim()
      : 'precomputed.json';
  const cacheSeed =
    manifest?.build || manifest?.buildId || manifest?.generatedAt || manifest?.timestamp || `${now}`;
  let targetUrl = `./data/${dataFile}?ts=${encodeURIComponent(cacheSeed)}`;
  const response = await fetch(targetUrl, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} - ${dataFile}`);
  }

  const text = await response.text();
  const byteLength = new TextEncoder().encode(text).length;

  try {
    const json = JSON.parse(text);
    return { json, byteLength };
  } catch (parseError) {
    const error = new Error('INVALID_JSON');
    error.cause = parseError;
    throw error;
  }
}

async function loadPrecomputed() {
  try {
    setDataInfo('데이터 크기를 계산 중입니다...');
    const { json, byteLength } = await loadData();
    if (json && json.status === 'no_data') {
      setDataInfo('데이터 파일이 비어 있습니다. 다시 생성해 주세요.', 'error');
      showEmptyState('데이터 파일이 비어 있습니다.', 'scripts/generate-data.js를 실행해 최신 precomputed.json을 생성해야 합니다.');
      return null;
    }
    if (!Array.isArray(json.analysisDates) || json.analysisDates.length === 0) {
      setDataInfo('데이터 없음', 'error');
      showEmptyState('데이터 없음');
      return null;
    }
    if (!json.priceSeries || typeof json.priceSeries !== 'object') {
      setDataInfo('priceSeries 누락', 'error');
      showEmptyState('priceSeries 누락');
      return null;
    }
    setDataInfo(`데이터 크기: ${formatByteSize(byteLength)}`);
    return json;
  } catch (error) {
    if (error?.message === 'INVALID_JSON') {
      console.warn('Failed to parse dataset JSON', error?.cause || error);
      setDataInfo('데이터를 해석하지 못했습니다.', 'error');
      showEmptyState('데이터를 해석하지 못했습니다.');
    } else {
      console.warn('Failed to load precomputed dataset', error);
      setDataInfo('데이터를 불러오지 못했습니다.', 'error');
      showEmptyState('데이터를 불러오지 못했습니다.');
    }
    return null;
  }
}

function isLocalhost() {
  return IS_LOCAL;
}

function populateControls() {
  const windowSelect = document.getElementById('window-select');
  if (!windowSelect) return;
  windowSelect.innerHTML = '';
  WINDOWS.forEach((windowSize) => {
    const option = document.createElement('option');
    option.value = windowSize;
    option.textContent = `${windowSize}일`;
    if (windowSize === state.window) {
      option.selected = true;
    }
    windowSelect.appendChild(option);
  });

  const rangeSelect = document.getElementById('range-select');
  if (!rangeSelect) return;
  rangeSelect.innerHTML = '';
  RANGE_OPTIONS.forEach((range) => {
    const option = document.createElement('option');
    option.value = range;
    option.textContent = `${range}일`;
    if (range === state.range) {
      option.selected = true;
    }
    rangeSelect.appendChild(option);
  });

  const pairSelect = document.getElementById('pair-select');
  if (!pairSelect) return;
  pairSelect.innerHTML = '';
  for (let i = 0; i < ASSETS.length; i += 1) {
    for (let j = i + 1; j < ASSETS.length; j += 1) {
      const pair = `${ASSETS[i].symbol}|${ASSETS[j].symbol}`;
      const option = document.createElement('option');
      option.value = pair;
      option.textContent = `${ASSETS[i].symbol} / ${ASSETS[j].symbol}`;
      if (pair === state.pair) {
        option.selected = true;
      }
      pairSelect.appendChild(option);
    }
  }

  if (!Array.from(pairSelect.options).some((option) => option.value === state.pair) && pairSelect.options.length > 0) {
    state.pair = pairSelect.options[0].value;
    pairSelect.value = state.pair;
  } else {
    pairSelect.value = state.pair;
  }

  if (!windowSelect.dataset.bound) {
    windowSelect.addEventListener('change', (event) => {
      state.window = Number(event.target.value);
      scheduleRender();
    });
    windowSelect.dataset.bound = 'true';
  }

  if (!rangeSelect.dataset.bound) {
    rangeSelect.addEventListener('change', (event) => {
      state.range = Number(event.target.value);
      scheduleRender();
    });
    rangeSelect.dataset.bound = 'true';
  }

  if (!pairSelect.dataset.bound) {
    pairSelect.addEventListener('change', (event) => {
      state.pair = event.target.value;
      scheduleRender();
    });
    pairSelect.dataset.bound = 'true';
  }

  const riskModeSelect = document.getElementById('risk-mode');
  if (riskModeSelect) {
    if (!Array.from(riskModeSelect.options).some((option) => option.value === 'ffl')) {
      const opt = document.createElement('option');
      opt.value = 'ffl';
      opt.textContent = 'FFL';
      riskModeSelect.appendChild(opt);
    }
    if (!Array.from(riskModeSelect.options).some((option) => option.value === 'ffl_exp')) {
      const opt2 = document.createElement('option');
      opt2.value = 'ffl_exp';
      opt2.textContent = 'FFL+EXP';
      riskModeSelect.appendChild(opt2);
    }
    if (!Array.from(riskModeSelect.options).some((option) => option.value === 'ffl_stab')) {
      const opt3 = document.createElement('option');
      opt3.value = 'ffl_stab';
      opt3.textContent = 'FFL+STAB';
      riskModeSelect.appendChild(opt3);
    }
    if (!Array.from(riskModeSelect.options).some((option) => option.value === 'fll_fusion')) {
      const opt4 = document.createElement('option');
      opt4.value = 'fll_fusion';
      opt4.textContent = 'FLL-Fusion';
      riskModeSelect.appendChild(opt4);
    }
    const supportedModes = ['classic', 'enhanced', 'ffl', 'ffl_exp', 'ffl_stab', 'fll_fusion'];
    const currentMode = supportedModes.includes(state.riskMode) ? state.riskMode : 'classic';
    if (riskModeSelect.value !== currentMode) {
      riskModeSelect.value = currentMode;
    }
    if (!riskModeSelect.dataset.bound) {
      riskModeSelect.addEventListener('change', (event) => {
        applyRiskMode(event.target.value);
        renderImmediately();
      });
      riskModeSelect.dataset.bound = 'true';
    }
  }


  const refreshButton = document.getElementById('refresh-button');
  const downloadButton = document.getElementById('download-report');
  const startInput = document.getElementById('custom-start');
  const endInput = document.getElementById('custom-end');

  const firstDate = state.analysisDates?.[0] || '';
  const lastDate = state.analysisDates?.[state.analysisDates.length - 1] || '';

  if (startInput) {
    startInput.min = firstDate;
    startInput.max = lastDate;
    startInput.value = state.customRange.start || '';
  }

  if (endInput) {
    endInput.min = firstDate;
    endInput.max = lastDate;
    endInput.value = state.customRange.end || '';
  }

  if (downloadButton) {
    downloadButton.disabled = !canBuildTextReport();
    if (!downloadButton.dataset.bound) {
      downloadButton.addEventListener('click', handleDownloadReport);
      downloadButton.dataset.bound = 'true';
    }
  }

  const handleCustomRangeChange = () => {
    state.customRange.start = startInput?.value || null;
    state.customRange.end = endInput?.value || null;

    const startTime = parseDateSafe(state.customRange.start);
    const endTime = parseDateSafe(state.customRange.end);
    const hasSelection = startTime !== null || endTime !== null;
    const invalid = startTime !== null && endTime !== null && startTime > endTime;

    state.customRange.valid = !invalid;

    if (refreshButton) {
      refreshButton.disabled = invalid;
    }

    if (invalid) {
      setCustomRangeFeedback('시작일은 종료일보다 앞서거나 같아야 합니다.');
      return;
    }

    if (hasSelection) {
      setCustomRangeFeedback('맞춤 기간이 설정되어 있습니다.');
    } else {
      setCustomRangeFeedback('');
    }
    scheduleRender();
  };

  if (startInput && !startInput.dataset.bound) {
    startInput.addEventListener('change', handleCustomRangeChange);
    startInput.dataset.bound = 'true';
  }

  if (endInput && !endInput.dataset.bound) {
    endInput.addEventListener('change', handleCustomRangeChange);
    endInput.dataset.bound = 'true';
  }

  handleCustomRangeChange();

  if (refreshButton && !refreshButton.dataset.bound) {
    refreshButton.addEventListener('click', handleRefreshClick);
    refreshButton.dataset.bound = 'true';
  }

  document.querySelectorAll('button.info').forEach((button) => {
    if (!button.dataset.bound) {
      button.addEventListener('click', () => {
        const target = document.getElementById(button.dataset.target);
        if (!target) return;
        // Hide others first
        document.querySelectorAll('.info-popover.visible').forEach((el) => el.classList.remove('visible'));
        target.classList.add('visible');
      });
      button.dataset.bound = 'true';
    }
  });

  // Bind close buttons
  document.querySelectorAll('.info-popover .popover-close').forEach((btn) => {
    if (!btn.dataset.bound) {
      btn.addEventListener('click', () => {
        const container = btn.closest('.info-popover');
        if (container) container.classList.remove('visible');
      });
      btn.dataset.bound = 'true';
    }
  });

  // Close on outside click
  if (!document.body.dataset.popoverOutsideBound) {
    document.addEventListener('click', (event) => {
      const open = document.querySelector('.info-popover.visible');
      if (!open) return;
      const isToggleBtn = event.target.closest('button.info');
      const inside = event.target.closest('.info-popover');
      if (!inside && !isToggleBtn) {
        open.classList.remove('visible');
      }
    });
    document.body.dataset.popoverOutsideBound = 'true';
  }

  // Close on Escape
  if (!document.body.dataset.popoverEscBound) {
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        document.querySelectorAll('.info-popover.visible').forEach((el) => el.classList.remove('visible'));
      }
    });
    document.body.dataset.popoverEscBound = 'true';
  }
}

function renderAll() {
  updateMeta();
  renderGauge();
  renderSubGauges();
  renderRisk();
  renderAlerts();
  renderHistory();
  renderBacktest();
  renderHeatmap();
  renderPair();
  updateDownloadButtonState();
}

// --- Risk regime configs ---
const RISK_BREADTH_SYMBOLS = SIGNAL.breadth;
const ENHANCED_LOOKBACKS = { momentum: 10, breadth: 5 };
const CLUSTERS = { risk: ['IWM', 'SPY', 'BTC-USD'], safe: ['TLT', 'GLD'] };
const RISK_CFG_FFL = {
  lookbacks: { momentum: 10, vol: 20, breadth: 5 },
  p: 1.5,
  zSat: 2.0,
  lambda: 0.25,
  thresholds: {
    jOn: +0.10,
    jOff: -0.08,
    scoreOn: 0.60,
    scoreOff: 0.40,
    breadthOn: 0.50,
    mmFragile: 0.88,
    mmOff: 0.96,
    pconOn: 0.55,
    pconOff: 0.40,
    // Added for diffusion+drift gating
    mmHi: 0.90,
    downAll: 0.60,
    corrConeDays: 5,
    driftMinDays: 3,
    driftCool: 2,
    offMinDays: 2,
    vOn: +0.05,
    vOff: -0.05,
    kOn: 0.60,
  },
  // Relative-strength lambda for kappa (diffusion vs drift)
  kLambda: 1.0,
  // EXP (Diffusion/Drift ratio) knobs
  exp: {
    lam: 0.75,          // ratio denominator scale for drift (aggressive)
    rOn: 0.76,          // min smoothed ratio to allow On
    rOff: -0.05,        // max smoothed ratio to force Off
    breadth1dMin: 0.45, // high-corr breadth minimum (slightly relaxed)
    d0AnyPos: true,     // any of {QQQ,IWM} > 0 on entry day
    d0BothPosHiCorr: false, // don't require both in high correlation
    ti: { win: 52, onK: 0.81, offK: 0.37, hiCorrScale: 1.25, strongK: 1.5 },
    // Dynamic leverage defaults for UI backtest
    lev: { r0: 0.30, r1: 1.05, min: 1.0, max: 3.0, damp: 3.0 },
  },
  // Tunable parameters for EXP majority rule (no formula change)
  expTune: {
    // Stability bias (integral)
    macroOn: 0.514, // tuned S0 (random search)
    macroOff: 0.371, // tuned S1 (random search)
    // Flux thresholds (derivative) via quantiles
    qOn: 0.715,
    qOff: 0.244,
    // Gains for dynamic thresholds (no new metrics; combine Classic/Flux/Stability)
    aS: 0.187,
    aJ: 0.095,
    bS: 0.106,
    bJ: 0.058,
    gSPos: 0.052,
    gSNeg: 0.065,
    // Confirmation days base
    confirmOn: 2,
    confirmOff: 1,
    // Hazard (Stability 급락 선행 경고)
    hazardHigh: 0.513,
    hazardDrop: 0.033,
    hazardLookback: 5,
    // Display score fusion weights (for gauge only)
    wC: 0.5,
    wF: 0.3,
    wS: 0.2,
  },
  // STAB: Stability-slope predictive tuning (no new data, only uses existing stability EMA3/EMA10)
  stabTune: {
    // Stability tuned from 1.txt/2.txt
    fast: 21,             // ~1 month
    slow: 63,             // ~3 months
    zWin: 126,
    zUp: 2.50,
    zDown: 2.50,
    slopeMin: 0.0200,
    neutralLo: 0.30,
    neutralHi: 0.40,
    // sustained-days to avoid single-day spikes
    lagUp: 3,
    lagDown: 4,
    onFluxEase: 0.02,
    confirmOnMin: 2,
    leadOnWindow: 6,
    downGrace: 6,
    hazardWindow: 9,
    offFluxTighten: 0.03,
    confirmOffMin: 1,
    onOverrideMargin: 0.01,
    // Up-shock protection: keep On unless Off persists
    upOffHarden: 0.02,      // make Off threshold more negative during up-lead
    upConfirmOffMin: 2,     // require N consecutive Off days while up-lead active
  },
  variant: 'classic', // default: Classic-FFL with enhanced guard; 'exp' enables ratio gates
};
const RISK_CFG_CLASSIC = {
  // original corr-heavy classic with stronger safe-neg weighting
  weights: { sc: 0.70, safe: 0.30 },
  thresholds: {
    scoreOn: 0.65,
    scoreOff: 0.30,
    corrOn: 0.50,   // corr-dominant gate
    corrMinOn: 0.20,
    corrOff: -0.10,
  },
  colors: { on: '#22c55e', neutral: '#facc15', off: '#f87171', onFragile: '#86efac' },
  pairKey: SIGNAL.pairKey,
};

const RISK_CFG_ENH = {
  // absorption guards only (no overfit)
  thresholds: { mmMaxOn: 0.80, mmMinOff: 0.85 },
  colors: { on: '#22c55e', neutral: '#facc15', off: '#f87171', onFragile: '#86efac' },
};

// (old computeRiskSeries removed; replaced by computeRiskSeriesMulti)

function formatRiskPairLabel() {
  const [left, right] = SIGNAL.pairKey.split('|');
  return `${left}↔${right} corr`;
}

function renderRisk() {
  const metrics = state.metrics[state.window];
  const elementGauge = document.getElementById('risk-score-gauge');
  const elementTimeline = document.getElementById('risk-timeline');
  if (!metrics || (!elementGauge && !elementTimeline)) return;

  const { records: filteredRecords, empty } = getFilteredRecords(metrics);
  if (empty) {
    if (elementGauge && charts.riskGauge) charts.riskGauge.clear();
    if (elementTimeline && charts.riskTimeline) charts.riskTimeline.clear();
    return;
  }

  const { records: filteredRecords2 } = getFilteredRecords(metrics);
  const series = state.riskMode === 'enhanced'
    ? computeRiskSeriesEnhanced(metrics, filteredRecords2)
    : (state.riskMode === 'fll_fusion')
      ? computeRiskSeriesFLLFusion(metrics, filteredRecords2)
      : ((state.riskMode === 'ffl' || state.riskMode === 'ffl_exp' || state.riskMode === 'ffl_stab')
        ? computeRiskSeriesFFL(metrics, filteredRecords2)
        : computeRiskSeriesClassic(metrics, filteredRecords2));
  // Precompute FFL overlay metrics so we can display Flux/FINT in any mode
  const fflOverlay = computeRiskSeriesFFL(metrics, filteredRecords2);
  if (!series) {
    if (elementGauge && charts.riskGauge) charts.riskGauge.clear();
    if (elementTimeline && charts.riskTimeline) charts.riskTimeline.clear();
    return;
  }

  const latestIdx = series.score.length - 1;
  const latestScore = series.score[latestIdx] || 0;
  const latestState = series.state[latestIdx] || 0;
  const palette = state.riskMode === 'classic' ? RISK_CFG_CLASSIC.colors : RISK_CFG_ENH.colors;
  const fragile = !!series.fragile?.[latestIdx];
  const stateLabel = latestState > 0 ? (fragile ? 'Risk-On (Fragile)' : 'Risk-On') : latestState < 0 ? 'Risk-Off' : 'Neutral';
  const stateColor = latestState > 0 ? (fragile ? (palette.onFragile || palette.on) : palette.on) : latestState < 0 ? palette.off : palette.neutral;

  if (elementGauge) {
    const gauge = charts.riskGauge || echarts.init(elementGauge);
    charts.riskGauge = gauge;
    gauge.setOption({
      series: [{
        type: 'gauge',
        startAngle: 210,
        endAngle: -30,
        min: 0,
        max: 1,
        axisLine: { lineStyle: { width: 16, color: [[1, stateColor]] } },
        pointer: { length: '65%', width: 4 },
        detail: {
          formatter: () => `${stateLabel}  \n${latestScore.toFixed(3)}`,
          fontSize: 14,
          color: TEXT_PRIMARY,
          backgroundColor: 'rgba(15,23,42,0.45)',
          padding: 6,
          borderRadius: 6,
          offsetCenter: [0, '60%'],
        },
        data: [{ value: latestScore }],
      }],
    });
  }

  if (elementTimeline) {
    const chart = charts.riskTimeline || echarts.init(elementTimeline);
    charts.riskTimeline = chart;
    const values = series.state.map((x) => (x === 0 ? 0.1 : x)); // small neutral height
    chart.setOption({
      tooltip: { trigger: 'axis', formatter: (p) => {
        const idx = p?.[0]?.dataIndex ?? 0;
        const stateValue = series.state[idx];
        const label = stateValue > 0 ? (series.fragile?.[idx] ? 'Risk-On (Fragile)' : 'Risk-On') : stateValue < 0 ? 'Risk-Off' : 'Neutral';
        const sc = series.scCorr[idx];
        const sn = series.safeNeg[idx];
        const parts = [];
        const baseScore = (state.riskMode === 'enhanced') ? (series.riskScore?.[idx]) : (series.score?.[idx]);
        if (Number.isFinite(baseScore)) parts.push(`점수 ${Number(baseScore).toFixed(3)}`);
        if (state.riskMode === 'enhanced') {
          const mm = series.mm?.[idx];
          const guardVal = series.guard?.[idx];
          const combo = series.comboMomentum?.[idx];
          const breadthVal = series.breadth?.[idx];
          if (Number.isFinite(combo)) parts.push(`공동모멘텀 ${(combo * 100).toFixed(1)}%`);
          if (Number.isFinite(breadthVal)) parts.push(`리스크폭 ${(breadthVal * 100).toFixed(0)}%`);
          if (Number.isFinite(mm)) parts.push(`흡수비 ${mm.toFixed(3)}`);
          if (Number.isFinite(guardVal)) parts.push(`위험가드 ${(guardVal * 100).toFixed(0)}%`);
        } else if (state.riskMode === 'ffl' || state.riskMode === 'ffl_exp' || state.riskMode === 'ffl_stab') {
          const fluxVal = series.fflFlux?.[idx];
          const fintVal = series.fluxIntensity?.[idx];
          const farVal = series.far?.[idx];
          const rbVal = series.riskBetaFlux?.[idx];
          const fullZ = series.fullFluxZ?.[idx];
          const diffVal = series.diffusionScore?.[idx];
          const mmTrend = series.mmTrend?.[idx];
          const apdf = series.apdf?.[idx];
          const pcon = series.pcon?.[idx];
          const mm = series.mm?.[idx];
          const guardVal = series.guard?.[idx];
          const combo = series.comboMomentum?.[idx];
          const breadthVal = series.breadth?.[idx];
          if (Number.isFinite(fluxVal)) parts.push(`J_norm ${fluxVal.toFixed(3)}`);
          if (Number.isFinite(fintVal)) parts.push(`FINT ${fintVal.toFixed(3)}`);
          if (Number.isFinite(farVal)) parts.push(`FAR ${farVal.toFixed(3)}`);
          if (Number.isFinite(rbVal)) parts.push(`RB_Flux ${rbVal.toFixed(3)}`);
          if (Number.isFinite(fullZ)) parts.push(`ΔCorr-Z ${fullZ.toFixed(3)}`);
          if (Number.isFinite(diffVal)) parts.push(`Diff ${diffVal.toFixed(3)}`);
          if (Number.isFinite(mmTrend)) parts.push(`mmΔ ${mmTrend.toFixed(3)}`);
          if (Number.isFinite(apdf)) parts.push(`APDF ${apdf.toFixed(3)}`);
          if (Number.isFinite(pcon)) parts.push(`PCON ${(pcon * 100).toFixed(0)}%`);
          if (Number.isFinite(mm)) parts.push(`Absorption ${mm.toFixed(3)}`);
          if (Number.isFinite(guardVal)) parts.push(`Guard ${(guardVal * 100).toFixed(0)}%`);
          if (Number.isFinite(combo)) parts.push(`공동모멘텀 ${(combo * 100).toFixed(1)}%`);
          if (Number.isFinite(breadthVal)) parts.push(`리스크폭 ${(breadthVal * 100).toFixed(0)}%`);
        }
        // Always include FFL overlays (Flux/FINT) for Classic/Enhanced tooltips
        if (fflOverlay) {
          const fx = fflOverlay.fflFlux?.[idx];
          const fint = fflOverlay.fluxIntensity?.[idx];
          if (Number.isFinite(fx)) parts.push(`FFL J_norm ${fx.toFixed(3)}`);
          if (Number.isFinite(fint)) parts.push(`FFL FINT ${fint.toFixed(3)}`);
        }
        const extras = parts.length > 0 ? `<br/>${parts.join(' · ')}` : '';
        const corrLabel = formatRiskPairLabel();
        return `${series.dates[idx]}<br/>${label}<br/>${corrLabel}: ${Number(sc).toFixed(3)}<br/>Safe-NEG: ${Number(sn).toFixed(3)}${extras}`;
      } },
    xAxis: { type: 'category', data: series.dates, axisLabel: { show: true, color: TEXT_AXIS } },
      yAxis: { type: 'value', min: -1, max: 1, show: false },
      series: [{
        type: 'bar',
        data: values,
        barWidth: '90%',
        itemStyle: {
          color: (params) => {
            const i = params.dataIndex; const v = series.state[i];
            if (v > 0) {
              return series.fragile?.[i] ? (palette.onFragile || palette.on) : palette.on;
            }
            return v < 0 ? palette.off : palette.neutral;
          },
        },
      }],
    });
  }
}

function computeRegimeSegments(dates, state) {
  const segs = [];
  let i = 0;
  while (i < state.length) {
    const cur = state[i];
    let j = i + 1;
    while (j < state.length && state[j] === cur) j += 1;
    segs.push({ start: i, end: j - 1, v: cur });
    i = j;
  }
  const areas = segs.map((s) => ({
    itemStyle: {
      color: s.v > 0 ? 'rgba(34,197,94,0.12)' : s.v < 0 ? 'rgba(248,113,113,0.12)' : 'rgba(250,204,21,0.10)',
    },
    name: s.v > 0 ? 'Risk-On' : s.v < 0 ? 'Risk-Off' : 'Neutral',
    xAxis: dates[s.start],
    xAxis2: dates[s.end],
  }));
  return areas;
}

function renderBacktest() {
  const metrics = state.metrics[state.window];
  const el = document.getElementById('backtest-chart');
  const stats = document.getElementById('backtest-stats');
  if (!metrics || !el) return;

  const { records: filtered, empty } = getFilteredRecords(metrics);
  const tradeConfig = SIGNAL.trade;
  const baseSymbol = tradeConfig.baseSymbol;
  if (empty || !state.priceSeries?.[baseSymbol]) {
    if (charts.backtest) charts.backtest.clear();
    if (stats) stats.textContent = '';
    return;
  }

  const series = state.riskMode === 'enhanced'
    ? computeRiskSeriesEnhanced(metrics, filtered)
    : (state.riskMode === 'fll_fusion')
      ? computeRiskSeriesFLLFusion(metrics, filtered)
      : ((state.riskMode === 'ffl' || state.riskMode === 'ffl_exp' || state.riskMode === 'ffl_stab')
        ? computeRiskSeriesFFL(metrics, filtered)
        : computeRiskSeriesClassic(metrics, filtered));
  if (!series) return;

  const symbol = baseSymbol;
  const windowOffset = Math.max(1, Number(state.window) - 1);
  // Align filtered to global records to compute price index correctly
  const firstDate = filtered?.[0]?.date;
  let baseIdx = (metrics.records || []).findIndex((r) => r.date === firstDate);
  if (baseIdx < 0) baseIdx = 0;
  const dates = series.dates;
  const prices = state.priceSeries[symbol] || [];
  const opens = state.priceSeriesOpen?.[symbol] || [];
  const baseReturns = [];
  const openToCloseReturns = [];
  for (let idx = 0; idx < dates.length; idx += 1) {
    const priceIndex = windowOffset + baseIdx + idx;
    const prevIndex = priceIndex - 1;
    let daily = 0;
    if (prices[priceIndex] != null && prices[prevIndex] != null && prices[prevIndex] !== 0) {
      daily = prices[priceIndex] / prices[prevIndex] - 1;
    }
    baseReturns.push(daily);
    const openPrice = opens[priceIndex];
    const closePrice = prices[priceIndex];
    const ocReturn = Number.isFinite(openPrice) && Number.isFinite(closePrice) && openPrice !== 0
      ? (closePrice / openPrice) - 1
      : daily;
    openToCloseReturns.push(ocReturn);
  }

  const executedState = Array.isArray(series.executedState) && series.executedState.length === series.state.length
    ? series.executedState
    : series.state.map((value, idx) => (idx === 0 ? 0 : series.state[idx - 1] || 0));
  const stratReturns = executedState.map((regime, idx) => {
    if (regime > 0) {
      return leveragedReturn(openToCloseReturns[idx], tradeConfig.leverage);
    }
    if (regime < 0) {
      return 0;
    }
    return baseReturns[idx];
  });

  function rollingReturnFromSeriesUI(priceSeries, symbol, index, lookback) {
    const series = priceSeries?.[symbol];
    if (!Array.isArray(series) || lookback <= 0 || index - lookback < 0) return null;
    const end = series[index]; const start = series[index - lookback];
    if (!Number.isFinite(end) || !Number.isFinite(start) || start === 0) return null;
    return end / start - 1;
  }

  // Equity curves
  const eqStrat = [];
  const eqBH = [];
  let s = 1;
  let b = 1;
  for (let i = 0; i < stratReturns.length; i += 1) {
    s *= 1 + (stratReturns[i] || 0);
    b *= 1 + (baseReturns[i] || 0);
    eqStrat.push(Number(s.toFixed(6)));
    eqBH.push(Number(b.toFixed(6)));
  }

  // Hit rates (1d and 5d)
  const ret = baseReturns;
  const fwd1 = ret.slice(1).map((_, i) => ret[i + 1]);
  const state1 = series.state.slice(0, series.state.length - 1);
  const hr1 = computeHitRate(state1, fwd1);
  const horizon = 5;
  const fwd5 = [];
  for (let i = 0; i < ret.length; i += 1) {
    let prod = 1;
    for (let k = 1; k <= horizon && i + k < ret.length; k += 1) prod *= 1 + ret[i + k];
    fwd5.push(prod - 1);
  }
  const state5 = series.state.slice(0, series.state.length - 1);
  const hr5 = computeHitRate(state5, fwd5.slice(0, state5.length));

  const chart = charts.backtest || echarts.init(el);
  charts.backtest = chart;
  chart.setOption({
    legend: { data: ['전략', '벤치마크'] },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: dates, axisLabel: { color: TEXT_AXIS } },
    yAxis: { type: 'value', scale: true, axisLabel: { color: TEXT_AXIS } },
    series: [
      {
        name: '전략',
        type: 'line',
        data: eqStrat,
        smooth: true,
        markArea: {
          silent: true,
          itemStyle: { opacity: 1 },
          data: computeRegimeSegments(dates, series.state).map((a) => [
            { xAxis: a.xAxis },
            { xAxis: a.xAxis2, itemStyle: a.itemStyle, name: a.name },
          ]),
        },
      },
      { name: '벤치마크', type: 'line', data: eqBH, smooth: true },
    ],
  });

  if (stats) {
    const onLabel = tradeConfig.leveredSymbol || `${tradeConfig.leverage}x ${tradeConfig.baseSymbol}`;
    const configNote = `포지션: On=${onLabel} · Neutral=${tradeConfig.baseSymbol} · Off=현금 (1일 지연)`;
    const out = `히트율(1일): ${(hr1 * 100).toFixed(1)}% · 히트율(5일): ${(hr5 * 100).toFixed(1)}% · 전략 누적 ${(pct(eqStrat[eqStrat.length - 1] - 1))} · 벤치마크 누적 ${pct(eqBH[eqBH.length - 1] - 1)} · ${configNote}`;
    stats.textContent = out;
  }
}

function computeHitRate(stateArr, fwdRetArr) {
  let wins = 0;
  let cnt = 0;
  for (let i = 0; i < Math.min(stateArr.length, fwdRetArr.length); i += 1) {
    const s = stateArr[i];
    if (s === 0) continue; // skip neutral
    const r = fwdRetArr[i];
    if (s > 0 && r > 0) wins += 1;
    if (s < 0 && r < 0) wins += 1;
    cnt += 1;
  }
  return cnt > 0 ? wins / cnt : 0;
}

function pct(x) { return `${(x * 100).toFixed(1)}%`; }

function renderAlerts() {
  const metrics = state.metrics[state.window];
  const box = document.getElementById('regime-alerts');
  if (!box || !metrics) return;
  const { records: filtered, empty } = getFilteredRecords(metrics);
  if (empty) { box.innerHTML = ''; return; }
  const series = state.riskMode === 'enhanced'
    ? computeRiskSeriesEnhanced(metrics, filtered)
    : (state.riskMode === 'fll_fusion')
      ? computeRiskSeriesFLLFusion(metrics, filtered)
      : ((state.riskMode === 'ffl' || state.riskMode === 'ffl_exp' || state.riskMode === 'ffl_stab')
        ? computeRiskSeriesFFL(metrics, filtered)
        : computeRiskSeriesClassic(metrics, filtered));
  if (!series) { box.innerHTML = ''; return; }
  const dates = series.dates;
  const states = series.state;
  // Find last transitions
  const events = [];
  for (let i = 1; i < states.length; i += 1) {
    if (states[i] !== states[i - 1]) {
      events.push({ date: dates[i], state: states[i] });
    }
  }
  const last = events.slice(-5);
  if (last.length === 0) { box.innerHTML = ''; return; }
  box.innerHTML = last.map((e) => `<span class="badge ${e.state > 0 ? 'on' : e.state < 0 ? 'off' : 'neutral'}">${e.date} · ${e.state > 0 ? 'On' : e.state < 0 ? 'Off' : 'Neutral'}</span>`).join('');
}

// Classic risk series (corr-dominant + Safe-NEG)
function computeRiskSeriesClassic(metrics, recordsOverride) {
  if (!metrics || !Array.isArray(metrics.records)) return null;
  const pair = (metrics.pairs && metrics.pairs[RISK_CFG_CLASSIC.pairKey]) || null;
  if (!pair || !Array.isArray(pair.correlation)) return null;

  // align to filtered view but compute from global indices
  const allDates = metrics.records.map((r) => r.date);
  let baseIdx = 0; let length = allDates.length;
  if (Array.isArray(recordsOverride) && recordsOverride.length > 0) {
    const firstDate = recordsOverride[0]?.date;
    const gIdx = metrics.records.findIndex((r) => r.date === firstDate);
    baseIdx = gIdx >= 0 ? gIdx : 0;
    length = recordsOverride.length;
  }
  const dates = allDates.slice(baseIdx, baseIdx + length);
  const scCorr = pair.correlation.slice(baseIdx, baseIdx + length);
  const safeNeg = metrics.records.slice(baseIdx, baseIdx + length).map((r) => safeNumber(r.sub?.safeNegative));

  const w = RISK_CFG_CLASSIC.weights;
  const th = RISK_CFG_CLASSIC.thresholds;
  const score = scCorr.map((c, i) => {
    const scPos = Math.max(0, Number(c));
    return Math.max(0, Math.min(1, w.sc * scPos + w.safe * safeNeg[i]));
  });
  const state = scCorr.map((c, i) => {
    const s = score[i];
    // Corr-dominant gate
    if (c >= th.corrOn) return 1;
    if (c <= th.corrOff || s <= th.scoreOff) return -1;
    if (s >= th.scoreOn && c >= th.corrMinOn) return 1;
    return 0;
  });
  return { dates, score, state, scCorr, safeNeg };
}

// Enhanced: classic + momentum, breadth, and absorption/stability guards
function computeRiskSeriesEnhanced(metrics, recordsOverride) {
  const classic = computeRiskSeriesClassic(metrics, recordsOverride);
  if (!classic) return null;
  const records = metrics?.records || [];
  if (!Array.isArray(records) || records.length === 0) return classic;

  const dates = classic.dates;
  const length = dates.length;
  const mm = new Array(length).fill(null);
  const fragile = new Array(length).fill(false);
  const guard = new Array(length).fill(null);
  const comboMomentum = new Array(length).fill(null);
  const breadth = new Array(length).fill(null);
  const stabilitySeries = new Array(length).fill(null);
  const riskScore = new Array(length).fill(null);

  const firstDate = Array.isArray(recordsOverride) && recordsOverride.length > 0
    ? recordsOverride[0]?.date
    : records[0]?.date;
  let baseIdx = records.findIndex((r) => r.date === firstDate);
  if (baseIdx < 0) baseIdx = 0;

  const windowOffset = Math.max(1, Number(state.window) - 1);
  const prices = state.priceSeries || {};
  const bearDamp = createBearDamp(prices);
  const stockSeries = prices[SIGNAL.primaryStock] || [];
  const btcSeries = prices['BTC-USD'] || [];

  for (let i = 0; i < length; i += 1) {
    const rec = records[baseIdx + i];
    const matrix = rec?.matrix;
    if (Array.isArray(matrix) && matrix.length > 0) {
      const lambda1 = topEigenvalue(matrix) || 0;
      mm[i] = lambda1 / (matrix.length || 1);
    }

    const priceIndex = windowOffset + baseIdx + i;
    const stockRet = rollingReturnFromSeries(stockSeries, priceIndex, ENHANCED_LOOKBACKS.momentum);
    const btcRet = rollingReturnFromSeries(btcSeries, priceIndex, ENHANCED_LOOKBACKS.momentum);
    comboMomentum[i] = averageFinite(stockRet, btcRet);
    breadth[i] = computeRiskBreadth(priceIndex, ENHANCED_LOOKBACKS.breadth);
    stabilitySeries[i] = rec?.stability ?? null;

  const corrScore = clamp01((classic.scCorr[i] - 0.15) / 0.55);
  const momentumScore = clamp01(((comboMomentum[i] ?? 0) + 0.015) / 0.06);
  const breadthScore = clamp01(breadth[i] ?? 0);
  const stabilityRelief = clamp01(1 - normalizeRangeSafe(mm[i], 0.78, 0.94));
  const blendedScore = 0.35 * corrScore + 0.40 * momentumScore + 0.15 * breadthScore + 0.10 * stabilityRelief;
  riskScore[i] = Number(blendedScore.toFixed(6));

  const safePenalty = normalizeRangeSafe(classic.safeNeg[i], 0.35, 0.60);
  const mmPenalty = normalizeRangeSafe(mm[i], 0.85, 0.97);
  const deltaPenalty = normalizeRangeSafe(Math.max(0, -(rec?.delta ?? 0)), 0.015, 0.05);
  const guardScore = 0.5 * mmPenalty + 0.2 * safePenalty + 0.3 * deltaPenalty;
    guard[i] = Number(guardScore.toFixed(6));

    const combo = comboMomentum[i] ?? 0;
    if (
      blendedScore >= 0.60 &&
      classic.scCorr[i] >= 0.33 &&
      momentumScore >= 0.35 &&
      guardScore < 0.9 &&
      combo > -0.005
    ) {
      classic.state[i] = 1;
      if (guardScore >= 0.55) {
        fragile[i] = true;
      }
    } else if (
      blendedScore <= 0.30 ||
      combo <= -0.03 ||
      classic.scCorr[i] <= 0 ||
      (Number.isFinite(mm[i]) && mm[i] >= 0.96) ||
      (classic.safeNeg[i] >= 0.65 && combo <= 0) ||
      guardScore >= 1
    ) {
      classic.state[i] = -1;
    } else {
      classic.state[i] = 0;
      if (guardScore >= 0.8 && classic.scCorr[i] >= 0.3 && combo >= -0.02) {
        fragile[i] = true;
      }
    }

    classic.score[i] = riskScore[i];
  }

  // Replace with 3-layer fusion (Classic base + Flux acceleration + Stability bias/hazard)
  if (state.riskMode === 'ffl_exp') {
    const classic = computeRiskSeriesClassic(metrics, alignedRecords || slice);
    const S0 = Number.isFinite(RISK_CFG_FFL?.expTune?.macroOn) ? RISK_CFG_FFL.expTune.macroOn : 0.45;
    const S1 = Number.isFinite(RISK_CFG_FFL?.expTune?.macroOff) ? RISK_CFG_FFL.expTune.macroOff : 0.35;
    const aS = Number.isFinite(RISK_CFG_FFL?.expTune?.aS) ? RISK_CFG_FFL.expTune.aS : 0.25;
    const aJ = Number.isFinite(RISK_CFG_FFL?.expTune?.aJ) ? RISK_CFG_FFL.expTune.aJ : 0.12;
    const bS = Number.isFinite(RISK_CFG_FFL?.expTune?.bS) ? RISK_CFG_FFL.expTune.bS : 0.20;
    const bJ = Number.isFinite(RISK_CFG_FFL?.expTune?.bJ) ? RISK_CFG_FFL.expTune.bJ : 0.10;
    const gSPos = Number.isFinite(RISK_CFG_FFL?.expTune?.gSPos) ? RISK_CFG_FFL.expTune.gSPos : 0.06;
    const gSNeg = Number.isFinite(RISK_CFG_FFL?.expTune?.gSNeg) ? RISK_CFG_FFL.expTune.gSNeg : 0.05;
    const baseOn = Math.max(1, Number(RISK_CFG_FFL?.expTune?.confirmOn) || 2);
    const baseOff = Math.max(1, Number(RISK_CFG_FFL?.expTune?.confirmOff) || 2);
    const hzHigh = Number.isFinite(RISK_CFG_FFL?.expTune?.hazardHigh) ? RISK_CFG_FFL.expTune.hazardHigh : 0.50;
    const hzDrop = Number.isFinite(RISK_CFG_FFL?.expTune?.hazardDrop) ? RISK_CFG_FFL.expTune.hazardDrop : 0.03;
    const hzLb = Math.max(2, Number(RISK_CFG_FFL?.expTune?.hazardLookback) || 5);
    const wC = Number.isFinite(RISK_CFG_FFL?.expTune?.wC) ? RISK_CFG_FFL.expTune.wC : 0.5;
    const wF = Number.isFinite(RISK_CFG_FFL?.expTune?.wF) ? RISK_CFG_FFL.expTune.wF : 0.3;
    const wS = Number.isFinite(RISK_CFG_FFL?.expTune?.wS) ? RISK_CFG_FFL.expTune.wS : 0.2;

    const sSeries = slice.map((r) => Number.isFinite(r?.smoothed) ? r.smoothed : null);
    const dSeries = slice.map((r) => Number.isFinite(r?.delta) ? r.delta : null);

    let prevStateLocal = 0;
    let onCand = 0;
    let offCand = 0;

    for (let i = 0; i < length; i += 1) {
      const s_c = Number.isFinite(classic?.score?.[i]) ? classic.score[i] : 0.5;
      const mmValue = Number.isFinite(mm?.[i]) ? mm[i] : 0;
      const guardValue = Number.isFinite(guard?.[i]) ? guard[i] : 1;
      const jf = Number.isFinite(jFlux?.[i]) ? jFlux[i] : 0;
      const sLvl = Number.isFinite(sSeries?.[i]) ? sSeries[i] : null;
      const dS = Number.isFinite(dSeries?.[i]) ? dSeries[i] : 0;

      // Hazard: recent 5d touched ≥ hzHigh, now < hzHigh, and 5d change ≤ -hzDrop
      let hazard = false;
      if (i >= hzLb && Number.isFinite(sLvl)) {
        const start = i - hzLb;
        let sMax = -Infinity;
        for (let k = start; k <= i; k += 1) {
          const v = sSeries[k]; if (Number.isFinite(v) && v > sMax) sMax = v;
        }
        const sLag = sSeries[i - hzLb];
        if (Number.isFinite(sMax) && Number.isFinite(sLag)) {
          if (sMax >= hzHigh && sLvl < hzHigh && (sLvl - sLag) <= -hzDrop) {
            hazard = true;
          }
        }
      }

      // Dynamic thresholds (layer fusion)
      const posS = Number.isFinite(sLvl) ? Math.max(0, sLvl - S0) : 0;
      const negS = Number.isFinite(sLvl) ? Math.max(0, S0 - sLvl) : 0;
      const posJ = Math.max(0, jf);
      const negJ = Math.max(0, -jf);

      let scoreOnStar = th.scoreOn - aS * posS - aJ * posJ;
      let scoreOffStar = th.scoreOff + bS * negS + bJ * negJ;
      // clamp
      scoreOnStar = Math.max(0.2, Math.min(0.9, scoreOnStar));
      scoreOffStar = Math.max(0.05, Math.min(0.6, scoreOffStar));

      let J_onStar = dynOnFlux - gSPos * posS;
      let J_offStar = dynOffFlux + gSNeg * negS;

      const onGate = (s_c >= scoreOnStar) && (jf >= J_onStar) && guardValue < 0.95 && mmValue < th.mmOff;
      const offGate = (s_c <= scoreOffStar) || (jf <= J_offStar) || hazard || (guardValue >= 1.0) || (mmValue >= th.mmOff);

      // Confirmation logic (sticky, low churn)
      let confirmOnDays = baseOn;
      let confirmOffDays = baseOff;
      if ((jf >= J_onStar) && (dS >= 0)) confirmOnDays = Math.max(1, baseOn - 1);
      if (offGate && (jf <= J_offStar || hazard || dS < 0)) confirmOffDays = 1;

      onCand = onGate ? onCand + 1 : 0;
      offCand = offGate ? offCand + 1 : 0;

      let decided = prevStateLocal;
      if (prevStateLocal === 1) {
        decided = offGate ? -1 : 1;
      } else if (prevStateLocal === -1) {
        decided = onCand >= confirmOnDays ? 1 : -1;
      } else {
        decided = offCand >= confirmOffDays ? -1 : (onCand >= confirmOnDays ? 1 : 0);
      }

      // Apply decision
      stateArr[i] = decided;
      fragile[i] = decided >= 0 && (guardValue >= th.mmFragile || (guardValue >= 0.9 && guardValue < 1.0));
      prevStateLocal = decided;

      // Display score: fused (no new metric)
      const sc = Number.isFinite(s_c) ? s_c : 0.5;
      const sf = 0.5 * (Math.max(-1, Math.min(1, jf)) + 1);
      const ss = Number.isFinite(sLvl) ? Math.max(0, Math.min(1, sLvl)) : 0.5;
      score[i] = Math.max(0, Math.min(1, wC * sc + wF * sf + wS * ss));
    }
  }

  return {
    ...classic,
    mm,
    fragile,
    guard,
    comboMomentum,
    breadth,
    stabilitySeries,
    riskScore,
  };
}

// Guarded FFL: start from baseline FFL state, then apply Stability-driven overrides only.
function computeRiskSeriesFFLGuarded(metrics, recordsOverride) {
  // 1) Baseline FFL
  const prevMode = state.riskMode;
  state.riskMode = 'ffl';
  const base = computeRiskSeriesFFL(metrics, recordsOverride);
  state.riskMode = prevMode;
  if (!base) return null;

  const records = Array.isArray(recordsOverride) && recordsOverride.length > 0 ? recordsOverride : metrics.records;
  const length = base.state.length;
  if (!Array.isArray(records) || records.length < length) return base;

  // 2) Build Stability slope/z and monthly shock windows using stabTune
  const stab = (RISK_CFG_FFL?.stabTune) || {};
  const sFastN = Number.isFinite(stab.fast) ? stab.fast : 21;
  const sSlowN = Number.isFinite(stab.slow) ? stab.slow : 63;
  const sZWin = Number.isFinite(stab.zWin) ? stab.zWin : 126;
  const sZUp = Number.isFinite(stab.zUp) ? stab.zUp : 2.0;
  const sZDown = Number.isFinite(stab.zDown) ? stab.zDown : 2.0;
  const sMin = Number.isFinite(stab.slopeMin) ? stab.slopeMin : 0.012;
  const lagUp = Math.max(1, Number.isFinite(stab.lagUp) ? Math.floor(stab.lagUp) : 3);
  const lagDown = Math.max(1, Number.isFinite(stab.lagDown) ? Math.floor(stab.lagDown) : 4);
  const leadOnWindow = Math.max(1, Number.isFinite(stab.leadOnWindow) ? Math.floor(stab.leadOnWindow) : 6);
  const downGrace = Math.max(1, Number.isFinite(stab.downGrace) ? Math.floor(stab.downGrace) : 5);
  const hazardWindow = Math.max(1, Number.isFinite(stab.hazardWindow) ? Math.floor(stab.hazardWindow) : 8);
  const upConfirmOffMin = Math.max(1, Number.isFinite(stab.upConfirmOffMin) ? Math.floor(stab.upConfirmOffMin) : 2);

  const S = records.slice(records.length - length).map((r) => safeNumber(r.stability));
  const sEmaF = ema(S, sFastN);
  const sEmaS = ema(S, sSlowN);
  const slope = sEmaF.map((v, i) => (Number.isFinite(v) && Number.isFinite(sEmaS[i])) ? (v - sEmaS[i]) : null);

  function rollingSigmaMADLocal(arr, end, win) {
    const xs = [];
    for (let k = Math.max(0, end - win + 1); k <= end; k += 1) { const v = arr[k]; if (Number.isFinite(v)) xs.push(v); }
    if (xs.length < 5) return null;
    const xsSorted = xs.slice().sort((a,b)=>a-b);
    const m = xsSorted[Math.floor(xsSorted.length/2)];
    const dev = xs.map((v)=>Math.abs(v - m));
    dev.sort((a,b)=>a-b);
    const mad = dev[Math.floor(dev.length/2)] || 0;
    return mad > 0 ? 1.4826 * mad : null;
  }

  const z = new Array(length).fill(null);
  let upSeq = 0; let dnSeq = 0;
  for (let i = 0; i < length; i += 1) {
    const d = slope[i];
    const sig = rollingSigmaMADLocal(slope, i, sZWin);
    const zi = (Number.isFinite(d) && Number.isFinite(sig) && sig > 0) ? (d / sig) : null;
    z[i] = zi;
    const upC = Number.isFinite(zi) && Number.isFinite(d) && zi >= sZUp && d >= sMin;
    const dnC = Number.isFinite(zi) && Number.isFinite(d) && zi <= -sZDown && d <= -sMin;
    upSeq = upC ? (upSeq + 1) : 0;
    dnSeq = dnC ? (dnSeq + 1) : 0;
  }

  // Windows
  const upLead = new Array(length).fill(0);
  const dnGrace = new Array(length).fill(0);
  const dnHazard = new Array(length).fill(0);
  let leadLeft = 0; let graceLeft = 0; let hazLeft = 0;
  upSeq = 0; dnSeq = 0;
  for (let i = 0; i < length; i += 1) {
    const d = slope[i]; const zi = z[i];
    const upC = Number.isFinite(zi) && Number.isFinite(d) && zi >= sZUp && d >= sMin;
    const dnC = Number.isFinite(zi) && Number.isFinite(d) && zi <= -sZDown && d <= -sMin;
    upSeq = upC ? (upSeq + 1) : 0;
    dnSeq = dnC ? (dnSeq + 1) : 0;
    if (upSeq >= lagUp) leadLeft = Math.max(leadLeft, leadOnWindow);
    if (dnSeq >= lagDown) { graceLeft = Math.max(graceLeft, downGrace); hazLeft = Math.max(hazLeft, hazardWindow); }
    if (leadLeft > 0) { upLead[i] = 1; leadLeft -= 1; }
    if (graceLeft > 0) { dnGrace[i] = 1; graceLeft -= 1; }
    else if (hazLeft > 0) { dnHazard[i] = 1; hazLeft -= 1; }
  }

  // 3) Apply overrides on state: defend during hazard, stick-on during upLead
  const guardedState = base.state.slice();
  // For trend filter when preventing early Off, require QQQ 5-day up
  const windowOffset = Math.max(1, state.window - 1);
  const prices = state.priceSeries || {};
  const pxQQQ = prices[SIGNAL.trade.baseSymbol] || [];
  const qqq5 = new Array(length).fill(null);
  for (let i = 0; i < length; i += 1) {
    const idx = windowOffset + i;
    const ret = rollingReturnFromSeries(pxQQQ, idx, 5);
    qqq5[i] = Number.isFinite(ret) ? ret : null;
  }

  let offRun = 0; // consecutive off candidates in upLead
  for (let i = 0; i < length; i += 1) {
    const baseSt = base.state[i] || 0;
    // defend when base On but hazard active -> neutral
    if (baseSt > 0 && dnHazard[i] === 1) {
      guardedState[i] = 0;
      offRun = 0;
      continue;
    }
    // prevent early Off: if base Off and upLead active and 5d QQQ up, keep On
    if (baseSt < 0 && upLead[i] === 1 && (qqq5[i] ?? 0) > 0) {
      guardedState[i] = 1;
      offRun = 0;
      continue;
    }
    // if base On and upLead active but momentary off candidate, require persistence
    if (baseSt > 0 && upLead[i] === 1) {
      // we model using consecutive offRun via baseSt transitions
      offRun = 0; // staying On
      guardedState[i] = 1;
      continue;
    }
    // default
    guardedState[i] = baseSt;
    offRun = baseSt < 0 ? (offRun + 1) : 0;
  }

  const executedState = guardedState.map((v, idx) => (idx === 0 ? 0 : guardedState[idx - 1] || 0));
  return { ...base, state: guardedState, executedState };
}
function computeRiskSeriesFFL(metrics, recordsOverride) {
  if (!metrics || !Array.isArray(metrics.records) || metrics.records.length === 0) {
    return null;
  }

  const labels = SIGNAL.symbols;
  const indexLookup = {};
  labels.forEach((symbol, idx) => { indexLookup[symbol] = idx; });

  const allRecords = metrics.records;
  const alignedRecords = Array.isArray(recordsOverride) && recordsOverride.length > 0 ? recordsOverride : null;
  const firstDate = alignedRecords?.[0]?.date ?? allRecords[0]?.date;
  let baseIdx = allRecords.findIndex((record) => record.date === firstDate);
  if (baseIdx < 0) baseIdx = 0;
  const length = alignedRecords ? alignedRecords.length : allRecords.length - baseIdx;
  if (length <= 0) return null;

  const slice = allRecords.slice(baseIdx, baseIdx + length);
  const dates = slice.map((record) => record.date);
  const pairSeries = metrics.pairs?.[SIGNAL.pairKey] || null;
  const scCorr = pairSeries?.correlation?.slice(baseIdx, baseIdx + length) || new Array(length).fill(null);
  const safeNeg = slice.map((record) => safeNumber(record.sub?.safeNegative));

  const windowOffset = Math.max(1, Number(state.window) - 1);
  const prices = state.priceSeries || {};
  const bearDamp = createBearDamp(prices);

  const mm = new Array(length).fill(null);
  const guard = new Array(length).fill(null);
  const score = new Array(length).fill(null);
  const jFlux = new Array(length).fill(null);
  const jRiskBeta = new Array(length).fill(null);
  const fullFlux = new Array(length).fill(null);
  const fullFluxZ = new Array(length).fill(null);
  const mmTrend = new Array(length).fill(null);
  const fluxSlope = new Array(length).fill(null);
  const diffusionScore = new Array(length).fill(null);
  const vPC1 = new Array(length).fill(null);
  const kappa = new Array(length).fill(null);
  const apdf = new Array(length).fill(null); // all-pairs directional flux
  const pcon = new Array(length).fill(null); // pairwise consistency [0,1]
  const fluxRaw = new Array(length).fill(null);
  const fluxIntensity = new Array(length).fill(null);
  const comboMomentum = new Array(length).fill(null);
  const breadth = new Array(length).fill(null);
  const coDownAll = new Array(length).fill(null);
  const fragile = new Array(length).fill(false);
  const far = new Array(length).fill(null);
  let prevMatrix = null;

  for (let i = 0; i < length; i += 1) {
    const record = allRecords[baseIdx + i];
    const matrix = record?.matrix;
    if (Array.isArray(matrix) && matrix.length > 0) {
      const lambda1 = topEigenvalue(matrix) || 0;
      mm[i] = lambda1 / (matrix.length || 1);
    }
    // mm 1일 변화(양수면 흡수/동조 강화)
    if (i > 0 && Number.isFinite(mm[i]) && Number.isFinite(mm[i - 1])) {
      mmTrend[i] = mm[i] - mm[i - 1];
    } else {
      mmTrend[i] = null;
    }

    const priceIndex = baseIdx + i + windowOffset;
    const zr = {};
    labels.forEach((symbol) => {
      zr[symbol] = zMomentumFromSeries(
        prices[symbol],
        priceIndex,
        RISK_CFG_FFL.lookbacks.momentum,
        RISK_CFG_FFL.lookbacks.vol,
        RISK_CFG_FFL.zSat,
      );
    });

    let weightSum = 0;
    let fluxSum = 0;
    let absSum = 0;
    // all-pairs directional flux & consistency accumulators
    let apdfWeighted = 0;
    let pconWeighted = 0;
    let weightsAll = 0;
    CLUSTERS.safe.forEach((safeSymbol) => {
      const safeIdx = indexLookup[safeSymbol];
      if (!Number.isInteger(safeIdx)) return;
      CLUSTERS.risk.forEach((riskSymbol) => {
        const riskIdx = indexLookup[riskSymbol];
        if (!Number.isInteger(riskIdx)) return;
        const coef = Array.isArray(matrix) ? (Number(matrix?.[safeIdx]?.[riskIdx]) || 0) : 0;
        const weight = Math.pow(Math.abs(coef), RISK_CFG_FFL.p);
        if (!Number.isFinite(weight) || weight <= 0) return;
        const sZ = Number.isFinite(zr[safeSymbol]) ? zr[safeSymbol] : 0;
        const rZ = Number.isFinite(zr[riskSymbol]) ? zr[riskSymbol] : 0;
        const diff = rZ - sZ;
        weightSum += weight;
        fluxSum += weight * diff;
        absSum += weight * Math.abs(diff);
        // cross-pair contributions to APDF/PCON
        apdfWeighted += weight * (rZ - sZ);
        pconWeighted += weight * (rZ > sZ ? 1 : 0);
        weightsAll += weight;
      });
    });

    // same-cluster pairs: risk-risk adds +avg(z), safe-safe adds -avg(z)
    for (let a = 0; a < CLUSTERS.risk.length; a += 1) {
      for (let b = a + 1; b < CLUSTERS.risk.length; b += 1) {
        const ia = indexLookup[CLUSTERS.risk[a]];
        const ib = indexLookup[CLUSTERS.risk[b]];
        if (!Number.isInteger(ia) || !Number.isInteger(ib)) continue;
        const coef = Array.isArray(matrix) ? (Number(matrix?.[ia]?.[ib]) || 0) : 0;
        const w = Math.pow(Math.abs(coef), RISK_CFG_FFL.p);
        if (!Number.isFinite(w) || w <= 0) continue;
        const aZ = Number.isFinite(zr[CLUSTERS.risk[a]]) ? zr[CLUSTERS.risk[a]] : 0;
        const bZ = Number.isFinite(zr[CLUSTERS.risk[b]]) ? zr[CLUSTERS.risk[b]] : 0;
        apdfWeighted += w * (0.5 * (aZ + bZ));
        pconWeighted += w * ((aZ > 0 && bZ > 0) ? 1 : 0);
        weightsAll += w;
      }
    }
    for (let a = 0; a < CLUSTERS.safe.length; a += 1) {
      for (let b = a + 1; b < CLUSTERS.safe.length; b += 1) {
        const ia = indexLookup[CLUSTERS.safe[a]];
        const ib = indexLookup[CLUSTERS.safe[b]];
        if (!Number.isInteger(ia) || !Number.isInteger(ib)) continue;
        const coef = Array.isArray(matrix) ? (Number(matrix?.[ia]?.[ib]) || 0) : 0;
        const w = Math.pow(Math.abs(coef), RISK_CFG_FFL.p);
        if (!Number.isFinite(w) || w <= 0) continue;
        const aZ = Number.isFinite(zr[CLUSTERS.safe[a]]) ? zr[CLUSTERS.safe[a]] : 0;
        const bZ = Number.isFinite(zr[CLUSTERS.safe[b]]) ? zr[CLUSTERS.safe[b]] : 0;
        apdfWeighted += w * (-0.5 * (aZ + bZ));
        pconWeighted += w * ((aZ <= 0 && bZ <= 0) ? 1 : 0);
        weightsAll += w;
      }
    }

    const Jbar = weightSum > 0 ? fluxSum / weightSum : 0;
    const Jnorm = Math.tanh(Jbar / RISK_CFG_FFL.lambda);
    jFlux[i] = Number.isFinite(Jnorm) ? Jnorm * bearDamp(priceIndex) : null;
    fluxRaw[i] = weightSum > 0 ? Jbar : null;
    fluxIntensity[i] = weightSum > 0 ? absSum / weightSum : null;
    // flux 1일 기울기(양수면 위험→안전 확산이 강화되는 중)
    if (i > 0 && Number.isFinite(jFlux[i]) && Number.isFinite(jFlux[i - 1])) {
      fluxSlope[i] = jFlux[i] - jFlux[i - 1];
    } else {
      fluxSlope[i] = null;
    }

    // Risk-beta 방향성 플럭스: 위험군 내부 Δcorr의 방향성(모멘텀 합) 반영
    let rbW = 0; let rbSum = 0;
    if (Array.isArray(prevMatrix)) {
      CLUSTERS.risk.forEach((a, ai) => {
        const ia = SIGNAL.symbols.indexOf(a);
        CLUSTERS.risk.forEach((b, bi) => {
          if (bi <= ai) return;
          const ib = SIGNAL.symbols.indexOf(b);
          const curr = Array.isArray(matrix?.[ia]) ? matrix[ia][ib] : null;
          const prev = Array.isArray(prevMatrix?.[ia]) ? prevMatrix[ia][ib] : null;
          if (!Number.isFinite(curr) || !Number.isFinite(prev)) return;
          const delta = curr - prev;
          const w = Math.pow(Math.abs(curr), RISK_CFG_FFL.p);
          const dir = Math.sign((zr[a] ?? 0) + (zr[b] ?? 0));
          rbW += w;
          rbSum += w * delta * dir;
        });
      });
    }
    const rbBar = rbW > 0 ? rbSum / rbW : 0;
    jRiskBeta[i] = Math.tanh(rbBar / (RISK_CFG_FFL.lambda || 0.25));

    // Full-matrix ΔCorr 기반 Flux-Guard(Z)
    const fraw = frobeniusDiff(matrix, prevMatrix);
    prevMatrix = matrix;
    fullFlux[i] = Number.isFinite(fraw) ? fraw : null;
    const z = rollingZScore(fullFlux, i, Math.min(63, Math.max(15, Math.floor(length / 4))));
    fullFluxZ[i] = Number.isFinite(z) ? z : null;
    // finalize APDF/PCON
    const apdfRaw = weightsAll > 0 ? apdfWeighted / weightsAll : 0;
    apdf[i] = Number.isFinite(apdfRaw) ? Math.tanh(apdfRaw / (RISK_CFG_FFL.lambda || 0.25)) : null;
    pcon[i] = weightsAll > 0 ? clamp01(pconWeighted / weightsAll) : null;

    const riskZValues = CLUSTERS.risk
      .map((symbol) => zr[symbol])
      .filter((value) => Number.isFinite(value));
    const allZValues = labels.map((symbol) => zr[symbol]).filter((v) => Number.isFinite(v));
    const downAll = allZValues.length > 0 ? (allZValues.filter((v) => v < 0).length / allZValues.length) : null;
    coDownAll[i] = Number.isFinite(downAll) ? downAll : null;
    comboMomentum[i] = riskZValues.length > 0
      ? riskZValues.reduce((acc, value) => acc + value, 0) / riskZValues.length
      : null;
    breadth[i] = riskZValues.length > 0
      ? riskZValues.filter((value) => value > 0).length / riskZValues.length
      : null;

    const safePen = normalizeRangeSafe(safeNeg[i], 0.35, 0.60);
    const mmPen = normalizeRangeSafe(mm[i], 0.85, 0.97);
    const deltaPen = normalizeRangeSafe(Math.max(0, -(record?.delta ?? 0)), 0.015, 0.05);
    const fluxGuard = Number.isFinite(fullFluxZ[i]) ? sigmoid(fullFluxZ[i], 0.85) : 1;
    const guardVal = 0.4 * mmPen + 0.2 * safePen + 0.2 * deltaPen + 0.2 * fluxGuard;
    guard[i] = Number.isFinite(guardVal) ? Number(guardVal.toFixed(6)) : null;

    const fluxScore = jFlux[i] == null ? null : 0.5 * (1 + jFlux[i]);
    const comboNorm = comboMomentum[i] == null ? null : clamp01(((comboMomentum[i] ?? 0) + 1) / 2);
    const breadthNorm = breadth[i] == null ? null : clamp01(breadth[i] ?? 0);
    const guardRelief = guardVal == null ? null : clamp01(1 - guardVal);
    const components = [];
    if (Number.isFinite(fluxScore)) components.push({ weight: 0.5, value: fluxScore });
    if (Number.isFinite(comboNorm)) components.push({ weight: 0.2, value: comboNorm });
    if (Number.isFinite(breadthNorm)) components.push({ weight: 0.2, value: breadthNorm });
    if (Number.isFinite(guardRelief)) components.push({ weight: 0.1, value: guardRelief });
    const totalWeight = components.reduce((acc, item) => acc + item.weight, 0);
    const aggregated = totalWeight > 0
      ? components.reduce((acc, item) => acc + item.weight * item.value, 0) / totalWeight
      : (Number.isFinite(fluxScore) ? fluxScore : 0.5);
    score[i] = Number(aggregated.toFixed(6));

    // Fick's First Law-inspired 확산 점수: J_norm에서 (흡수 증가, 플럭스 둔화)을 패널티
    const k1 = 0.50; // mmTrend 패널티 계수(보수적)
    const k2 = 0.15; // 음의 fluxSlope 패널티 계수
    const mt = Number.isFinite(mmTrend[i]) ? Math.max(0, mmTrend[i]) : 0; // 상승분만 페널티
    const fsNeg = Number.isFinite(fluxSlope[i]) ? Math.max(0, -fluxSlope[i]) : 0; // 둔화(음수)만 페널티
    const diff = (Number.isFinite(jFlux[i]) ? jFlux[i] : 0) - (k1 * mt) - (k2 * fsNeg);
    diffusionScore[i] = Number.isFinite(diff) ? Number(diff.toFixed(6)) : null;

    // PC1 velocity (market mode) using top eigenvector · z-momentum
    try {
      if (Array.isArray(matrix) && matrix.length > 0) {
        const e1 = topEigenvectorLocal(matrix);
        if (Array.isArray(e1)) {
          let num = 0; let den = 0;
          for (let ei = 0; ei < e1.length; ei += 1) {
            const sym = SIGNAL.symbols[ei];
            const zi = Number.isFinite(zr[sym]) ? zr[sym] : 0;
            num += e1[ei] * zi;
            den += Math.abs(e1[ei]);
          }
          const vproj = den > 0 ? (num / den) : 0;
          vPC1[i] = Math.tanh(vproj / 0.5);
        } else {
          vPC1[i] = null;
        }
      } else {
        vPC1[i] = null;
      }
    } catch (e) {
      vPC1[i] = null;
    }
  }

  const th = RISK_CFG_FFL.thresholds;
  const variant = (state.riskMode === 'ffl_exp') ? 'exp' : (state.riskMode === 'ffl_stab') ? 'stab' : RISK_CFG_FFL.variant;
  const validFlux = jFlux.filter((value) => Number.isFinite(value));
  const validScore = score.filter((value) => Number.isFinite(value));
  let dynOnFlux = th.jOn;
  let dynOffFlux = th.jOff;
  let dynScoreOn = th.scoreOn;
  let dynScoreOff = th.scoreOff;
  if (variant !== 'classic' && validFlux.length >= 50) {
    const qOn = Number.isFinite(RISK_CFG_FFL?.expTune?.qOn) ? RISK_CFG_FFL.expTune.qOn : 0.75;
    const qOff = Number.isFinite(RISK_CFG_FFL?.expTune?.qOff) ? RISK_CFG_FFL.expTune.qOff : 0.25;
    dynOnFlux = Math.max(th.jOn, quantile(validFlux, qOn));
    dynOffFlux = Math.min(th.jOff, quantile(validFlux, qOff));
  }
  if (variant !== 'classic' && validScore.length >= 50) {
    dynScoreOn = Math.max(th.scoreOn, quantile(validScore, 0.75));
    dynScoreOff = Math.min(th.scoreOff, quantile(validScore, 0.25));
  }

  const stateArr = new Array(length).fill(0);
  let prevState = 0;
  let holdDays = 0;
  let onCand = 0;
  let offCand = 0;
  // Track guard-only exit sequences to require confirmation in high-correlation bursts
  let offGuardSeq = 0;
  // 기본 확인일은 2일이지만, 확산/흡수 상황에 따라 동적으로 조정
  // Drift epoch tracking
  const DRIFT_MIN_DAYS = Number.isFinite(RISK_CFG_FFL.thresholds.driftMinDays) ? RISK_CFG_FFL.thresholds.driftMinDays : 5;
  const DRIFT_COOLDOWN_DAYS = Number.isFinite(RISK_CFG_FFL.thresholds.driftCool) ? RISK_CFG_FFL.thresholds.driftCool : 2;
  let driftSeq = 0;
  let driftCooldown = 0;
  let inDriftEpoch = false;
  // EXP timing buffers
  const expSt = new Array(length).fill(null);
  const expStPrev = new Array(length).fill(null);
  const expSigma = new Array(length).fill(null);

  function median(values) {
    const xs = values.slice().sort((a,b)=>a-b);
    const n = xs.length; if (n===0) return null; const m = Math.floor(n/2);
    return (n%2===1) ? xs[m] : 0.5*(xs[m-1]+xs[m]);
  }
  function rollingSigmaMAD(arr, end, win) {
    const xs = [];
    for (let k=Math.max(0,end-win+1); k<=end; k+=1) { const v=arr[k]; if (Number.isFinite(v)) xs.push(v); }
    if (xs.length < 5) return null;
    const med = median(xs);
    const dev = xs.map(v=>Math.abs(v-med));
    const mad = median(dev);
    if (!Number.isFinite(mad)) return null;
    return 1.4826 * mad;
  }

  // Precompute EXP timing series if needed
  if (variant === 'exp') {
    const win = Number.isFinite(RISK_CFG_FFL?.exp?.ti?.win) ? RISK_CFG_FFL.exp.ti.win : 63;
    const ratios = slice.__expRatio || [];
    // Build EMAs for ratio
    const emaFast = ema(ratios.map((v)=>Number.isFinite(v)?v:0), 3);
    const emaSlow = ema(ratios.map((v)=>Number.isFinite(v)?v:0), 10);
    for (let i=0;i<length;i+=1){
      const rf = Number.isFinite(emaFast[i])?emaFast[i]:null;
      const rs = Number.isFinite(emaSlow[i])?emaSlow[i]:null;
      const st = (rf!=null && rs!=null) ? (rf - rs) : null;
      expSt[i] = st;
      expStPrev[i] = i>0 && Number.isFinite(expSt[i-1]) && Number.isFinite(st) ? (st - expSt[i-1]) : null;
      expSigma[i] = rollingSigmaMAD(expSt, i, win);
    }
  }

  // Precompute STAB slope/z if needed
  const stab = (RISK_CFG_FFL?.stabTune) || {};
  const sFastN = Number.isFinite(stab.fast) ? stab.fast : 3;
  const sSlowN = Number.isFinite(stab.slow) ? stab.slow : 10;
  const sZWin = Number.isFinite(stab.zWin) ? stab.zWin : 63;
  const sNeutralLo = Number.isFinite(stab.neutralLo) ? stab.neutralLo : 0.30;
  const sNeutralHi = Number.isFinite(stab.neutralHi) ? stab.neutralHi : 0.40;
  const sMin = Number.isFinite(stab.slopeMin) ? stab.slopeMin : 0.010;
  const sZUp = Number.isFinite(stab.zUp) ? stab.zUp : 1.1;
  const sZDown = Number.isFinite(stab.zDown) ? stab.zDown : 1.1;

  const Sseries = slice.map((rec) => safeNumber(rec.stability));
  const sEmaF = ema(Sseries, sFastN);
  const sEmaS = ema(Sseries, sSlowN);
  const sSlope = new Array(length).fill(null);
  const sSigma = new Array(length).fill(null);
  const sZ = new Array(length).fill(null);
  let sUpSeq = 0;
  let sDownSeq = 0;
  for (let i = 0; i < length; i += 1) {
    const vF = Number.isFinite(sEmaF[i]) ? sEmaF[i] : null;
    const vS = Number.isFinite(sEmaS[i]) ? sEmaS[i] : null;
    const d = (vF != null && vS != null) ? (vF - vS) : null;
    sSlope[i] = d;
    sSigma[i] = rollingSigmaMAD(sSlope, i, sZWin);
    const sig = sSigma[i];
    const z = (Number.isFinite(d) && Number.isFinite(sig) && sig > 0) ? (d / sig) : null;
    sZ[i] = z;
    // sustained detection sequences (monthly-level): require consecutive days
    const upCond = Number.isFinite(z) && Number.isFinite(d) && z >= sZUp && d >= sMin;
    const downCond = Number.isFinite(z) && Number.isFinite(d) && z <= -sZDown && d <= -sMin;
    sUpSeq = upCond ? (sUpSeq + 1) : 0;
    sDownSeq = downCond ? (sDownSeq + 1) : 0;
  }

  // STAB phase counters (not used in simplified variant, kept for compatibility)
  let stabGraceLeft = 0;
  let stabHazardLeft = 0;
  let stabLeadOnLeft = 0;
  // neutral clamp run for stab (consecutive days)
  let stabNeutRun = 0;
  // Running downtrend counter for simple 'accelerate Off' rule
  let sDownRun = 0;

  for (let i = 0; i < length; i += 1) {
    const fluxVal = jFlux[i] ?? 0;
    const diffVal = diffusionScore[i] ?? fluxVal;
    const scoreVal = score[i] ?? 0.5;
    const mmValue = mm[i] ?? 0;
    const comboValue = comboMomentum[i] ?? null;
    const breadthValue = breadth[i] ?? null;
    const guardVal = guard[i] ?? 1;
    const vpc1Val = Number.isFinite(vPC1[i]) ? vPC1[i] : 0;
    const lam = Number.isFinite(RISK_CFG_FFL.kLambda) ? RISK_CFG_FFL.kLambda : 1.0;
    const kappaVal = (Math.abs(diffVal)) / (Math.abs(diffVal) + lam * Math.abs(vpc1Val) + 1e-6);
    kappa[i] = Number.isFinite(kappaVal) ? kappaVal : null;

    // EXP: compute signed Diffusion/Drift ratio (smoothed)
    if (variant === 'exp') {
      const lamExp = Number.isFinite(RISK_CFG_FFL?.exp?.lam) ? RISK_CFG_FFL.exp.lam : 1.0;
      const denom = lamExp * Math.abs(vpc1Val) + 1e-6;
      const raw = Number.isFinite(diffVal) ? (diffVal / denom) : null;
      const prev = i > 0 ? slice?.__expRatio?.[i - 1] : null;
      const smooth = (Number.isFinite(raw) && Number.isFinite(prev)) ? (0.5 * prev + 0.5 * raw)
        : (Number.isFinite(raw) ? raw : Number.isFinite(prev) ? prev : null);
      slice.__expRatio = slice.__expRatio || new Array(length).fill(null);
      slice.__expRatio[i] = smooth;
    }

    if (variant === 'classic' || variant === 'exp' || variant === 'stab') {
      const guardValue = Number.isFinite(guardVal) ? guardVal : 1;
      const guardSoft = 0.95;
      const guardHard = 0.98;
      const breadthGate = (breadthValue ?? 0) >= ((th.breadthOn ?? 0.5) * 0.6);
      // 메인 On: 위험↔안전 플럭스 양(+) + 가드 양호 + 폭 확보
      // mm 높을수록 On 문턱 상향(고상관 구간에서 보수화)
      let dynOnAdj = dynOnFlux + (mmValue >= 0.94 ? 0.05 : mmValue >= 0.90 ? 0.03 : 0);
      // STAB simple flags
      let dynOffLocal = dynOffFlux;
      const sVal_now = Number.isFinite(sSlope[i]) ? sSlope[i] : null;
      const sZ_now = Number.isFinite(sZ[i]) ? sZ[i] : null;
      const stabUpTrend = (variant === 'stab') && Number.isFinite(sVal_now) && sVal_now > 0;
      const stabPlunge = (variant === 'stab') && Number.isFinite(sVal_now) && Number.isFinite(sZ_now) && (sZ_now <= -sZDown) && (sVal_now < 0);
      if (variant === 'stab') {
        if (Number.isFinite(sVal_now) && sVal_now < -Math.max(1e-6, sMin)) sDownRun += 1; else sDownRun = 0;
      }
      // 확산 점수(diffVal) + 전쌍 일관성(PCON) + APDF 약한 필터
      const pconOkBase = !Number.isFinite(pcon[i]) || pcon[i] >= (th.pconOn ?? 0.55);
      const apdfOk = !Number.isFinite(apdf[i]) || apdf[i] >= -0.05;
      let pconOk = pconOkBase || (diffVal >= (dynOnAdj + 0.07));
      // 고상관·약세(벤치 10일 ≤ 0) 잠금: 더 강한 확인 요구
      const idxPrice = baseIdx + i + windowOffset;
      const bench10 = rollingReturnFromSeries(prices[SIGNAL.trade.baseSymbol], idxPrice, 10);
      const bench20 = rollingReturnFromSeries(prices[SIGNAL.trade.baseSymbol], idxPrice, 20);
      const hiCorrBear = (mmValue >= 0.90) && (Number.isFinite(bench10) ? bench10 <= 0 : true);
      // Drift-Lock: 고상관 하락 드리프트(짧은 랠리 무시)
      // Drift-Lock(간단): 5자산 중 60% 이상이 하락(z<0)이고 J_norm<=0 이면 드리프트
      const hiCorrDrift = ((Number.isFinite(coDownAll[i]) ? coDownAll[i] >= 0.60 : false) && ((fluxVal ?? 0) <= 0)) ||
        ((mmValue >= 0.90) && (Number.isFinite(bench20) ? bench20 <= 0 : true));
      // Drift epoch update
      if (hiCorrDrift) { driftSeq += 1; driftCooldown = 0; }
      else { driftSeq = 0; driftCooldown += 1; }
      if (driftSeq >= DRIFT_MIN_DAYS) inDriftEpoch = true;
      if (inDriftEpoch && driftCooldown >= DRIFT_COOLDOWN_DAYS) inDriftEpoch = false;
      const stricter = !hiCorrBear || (
        (diffVal >= (dynOnAdj + 0.05)) && (Number.isFinite(pcon[i]) ? pcon[i] >= 0.65 : true) && (Number.isFinite(apdf[i]) ? apdf[i] >= 0 : true) && (Number.isFinite(comboValue) ? comboValue >= 0.10 : true)
      );
      // EXP constraints + timing
      let expOkOn = true;
      let expForceOff = false;
      if (variant === 'exp') {
        const ratioS = slice.__expRatio?.[i];
        const rOn = Number.isFinite(RISK_CFG_FFL?.exp?.rOn) ? RISK_CFG_FFL.exp.rOn : 1.20;
        const rOff = Number.isFinite(RISK_CFG_FFL?.exp?.rOff) ? RISK_CFG_FFL.exp.rOff : -1.10;
        const breadthMin = Number.isFinite(RISK_CFG_FFL?.exp?.breadth1dMin) ? RISK_CFG_FFL.exp.breadth1dMin : 0.55;
        // Day-0 price veto
        const idxPrice0 = baseIdx + i + windowOffset;
        const rQQQ = rollingReturnFromSeries(prices[SIGNAL.trade.baseSymbol], idxPrice0, 1);
        const rIWM = rollingReturnFromSeries(prices['IWM'], idxPrice0, 1);
        const anyPos = (Number.isFinite(rQQQ) ? rQQQ > 0 : false) || (Number.isFinite(rIWM) ? rIWM > 0 : false);
        const bothPos = (Number.isFinite(rQQQ) ? rQQQ > 0 : false) && (Number.isFinite(rIWM) ? rIWM > 0 : false);
        const needBoth = (RISK_CFG_FFL?.exp?.d0BothPosHiCorr ? (mmValue >= 0.90) : false);
        const day0Ok = needBoth ? bothPos : (RISK_CFG_FFL?.exp?.d0AnyPos ? anyPos : true);
        const breadth1dOk = (mmValue >= 0.90) ? ((breadthValue ?? 0) >= breadthMin) : true;
        // Timing layer
        const onK = Number.isFinite(RISK_CFG_FFL?.exp?.ti?.onK) ? RISK_CFG_FFL.exp.ti.onK : 0.75;
        const offK = Number.isFinite(RISK_CFG_FFL?.exp?.ti?.offK) ? RISK_CFG_FFL.exp.ti.offK : 0.50;
        const hiScale = Number.isFinite(RISK_CFG_FFL?.exp?.ti?.hiCorrScale) ? RISK_CFG_FFL.exp.ti.hiCorrScale : 1.25;
        const sig = expSigma[i];
        const tauOn = Number.isFinite(sig) ? onK * sig : 0;
        const tauOff = Number.isFinite(sig) ? offK * sig : 0;
        const scale = mmValue >= 0.90 ? hiScale : 1.0;
        const St = expSt[i];
        const prevSt = i>0 ? expSt[i-1] : null;
        const dSt = expStPrev[i];
        const stOn = Number.isFinite(St) && Number.isFinite(dSt) && (St > scale * tauOn) && (dSt > 0) && (Number(ratioS) > 0);
        const stOff = Number.isFinite(St) && Number.isFinite(dSt) && (St < -scale * tauOff) && (dSt < 0) && (Number(ratioS) < 0);
        const crossUp = Number.isFinite(prevSt) && Number.isFinite(St) && prevSt <= 0 && St > 0 && (dSt ?? 0) > 0;
        // On Booster: high corr-with-BTC & risk breadth or momentum combo, when not hiCorrBear
        const sc = Number(scCorr?.[i]) || 0;
        const booster = (!hiCorrBear && ((sc >= 0.50 && ((breadthValue ?? 0) >= 0.55 || (comboValue ?? 0) >= 0.05))));
        expOkOn = (Number.isFinite(ratioS) ? (ratioS >= rOn) : true) && day0Ok && breadth1dOk && (stOn || (!Number.isFinite(sig))); // fall back if sigma unavailable
        // Allow On when crossing to positive timing or booster pattern
        if (!expOkOn && day0Ok && (crossUp || booster)) {
          expOkOn = true;
        }
        expForceOff = (Number.isFinite(ratioS) ? (ratioS <= rOff) : false) || stOff;
        // relax pcon when booster triggers
        if (booster) pconOk = true;
      }

      const onClassicMain = (diffVal >= dynOnAdj) && pconOk && apdfOk && stricter && guardValue < guardSoft && mmValue < th.mmOff && breadthGate && expOkOn;
      // 보조 On: 위험군 내부 베타 플럭스가 양(+)이고 Combo 모멘텀이 양(+)일 때(2023 저상관 상승 대응)
      // 약세 고상관 구간에서는 RB_Flux 대안 경로 비활성화(베어랠리 진입 억제)
      const onClassicAlt = !hiCorrBear && (Number.isFinite(jRiskBeta[i]) && jRiskBeta[i] >= 0.06) &&
        (Number.isFinite(comboValue) && comboValue >= 0.10) &&
        guardValue < 0.90;
      // Strong-On: diffusion 우세(κ) + PC1 drift 양(+) + 일관성 높음
      const strongOn = (diffVal >= (dynOnAdj + 0.03))
        && (Number.isFinite(kappaVal) ? kappaVal >= (RISK_CFG_FFL.thresholds.kOn ?? 0.60) : true)
        && (Number.isFinite(pcon[i]) ? pcon[i] >= Math.max(0.65, (th.pconOn ?? 0.55)) : true)
        && (Number.isFinite(vpc1Val) ? vpc1Val >= (th.vOn ?? 0.05) : true)
        && guardValue < guardSoft
        && mmValue < th.mmOff;
      const onClassic = (onClassicMain || onClassicAlt || strongOn) && !hiCorrDrift;
      const offFlux = fluxVal <= dynOffLocal;
      const offGuard = (guardValue >= guardHard) || (mmValue >= th.mmOff);
      // Guard-only exit detection (no flux break)
      const guardOnly = offGuard && !offFlux;
      // Trend support in high-correlation: mild positive flux/breadth & non-negative combo
      const breadthGateLoosen = (breadthValue ?? 0) >= ((th.breadthOn ?? 0.5) * 0.5);
      const trendSupport = (fluxVal >= (dynOnFlux - 0.03)) && breadthGateLoosen && (Number.isFinite(comboValue) ? comboValue >= 0 : true);
      const hiCorr = mmValue >= 0.92;
      // Require confirmation for guard-only exits; extend to 3 days when hiCorr + trend support
      const guardConfirmDays = (hiCorr && trendSupport) ? 3 : 2;
      if (guardOnly) {
        offGuardSeq += 1;
      } else {
        offGuardSeq = 0;
      }
      const offGuardConfirmed = guardOnly && offGuardSeq >= guardConfirmDays;
      // Off by relative strength of negative drift or weak diffusion dominance
      const offByRel = ((Number.isFinite(vpc1Val) ? vpc1Val <= (th.vOff ?? -0.05) : false) && (Math.abs(vpc1Val) >= 0.05))
        || ((diffVal <= dynOffFlux) && (Number.isFinite(kappaVal) ? kappaVal < 0.55 : false))
        || (variant === 'exp' && expForceOff);
      let offClassic = offByRel || offFlux || offGuardConfirmed || (Number.isFinite(pcon[i]) && pcon[i] <= (th.pconOff ?? 0.40) && mmValue >= 0.92);
      // Drift-Lock 중에는 즉시 Off(확인 불요)
      if (hiCorrDrift) {
        offClassic = true;
      }
      // PC1 방향(위험군 평균 3일 수익) 음수 & mm 높음이면 On 차단(고상관 하락 방지)
      const risk3 = (() => {
        const idxPrice = baseIdx + i + windowOffset;
        const rList = CLUSTERS.risk.map((sym) => rollingReturnFromSeries(prices[sym], idxPrice, 3)).filter(Number.isFinite);
        return rList.length ? rList.reduce((a, b) => a + b, 0) / rList.length : null;
      })();
      const blockOnHighCorrDown = Number.isFinite(risk3) && risk3 <= 0 && mmValue >= 0.90;

      // 원시 판정으로 후보 카운트
      let rawOn = onClassic && !blockOnHighCorrDown;
      let rawOff = offClassic;
      if (variant === 'stab') {
        if (stabUpTrend) {
          rawOff = false; // do not switch to Off while stability uptrend
        }
        if (stabPlunge) {
          rawOff = true; // switch to Off early on plunge
        }
        // Accelerate Off on sustained downtrend (time-lagged effect)
        const needDown = Math.max(1, Number.isFinite(stab.lagDown) ? Math.floor(stab.lagDown) : 4);
        if (!stabUpTrend && !stabPlunge && sDownRun >= needDown) {
          rawOff = true;
        }
      }
      // 동적 On 확인일: 확산 가속(+), 흡수 하락(≤0)이면 1일, 기본 2일, 고상관/흡수 상승·ΔCorr-Z↑면 3일
      const hiCorrRisk = (mmValue >= 0.90) || ((mmTrend[i] ?? 0) > 0.005) || ((fullFluxZ[i] ?? 0) >= 1.5);
      const accel = (fluxSlope[i] ?? 0) > 0 && (mmTrend[i] ?? 0) <= 0;
      const strongPcon = Number.isFinite(pcon[i]) && pcon[i] >= 0.68;
      // Strong-On path shortens confirmation to 1 day
      let confirmOnDays = strongOn ? 1 : (hiCorrRisk ? 3 : strongPcon ? 1 : (accel ? 1 : 2));
      if (variant === 'exp') {
        const strongK = Number.isFinite(RISK_CFG_FFL?.exp?.ti?.strongK) ? RISK_CFG_FFL.exp.ti.strongK : 1.5;
        const sig = expSigma[i];
        const St = expSt[i];
        const tauOn = Number.isFinite(sig) ? (Number.isFinite(RISK_CFG_FFL?.exp?.ti?.onK) ? RISK_CFG_FFL.exp.ti.onK : 0.75) * sig : null;
        if (Number.isFinite(St) && Number.isFinite(tauOn) && St >= strongK * tauOn) confirmOnDays = 1;
        const prevSt = i>0 ? expSt[i-1] : null;
        const dSt = expStPrev[i];
        if (Number.isFinite(prevSt) && Number.isFinite(St) && prevSt <= 0 && St > 0 && (dSt ?? 0) > 0) {
          // bullish timing cross-up: shorten confirmation by 1 day
          confirmOnDays = Math.max(1, confirmOnDays - 1);
        }
      }
      // no confirm-day modification for STAB (keep simple)
      onCand = rawOn ? onCand + 1 : 0;
      offCand = rawOff ? offCand + 1 : 0;

      // Neutral clamp (FFL+STAB only): mid-level S, tiny slope, tiny flux → force Neutral after 2 days
      if (variant === 'stab') {
        const S_now = Number.isFinite(Sseries[i]) ? Sseries[i] : null;
        const slopeAbs = Math.abs(Number.isFinite(sSlope[i]) ? sSlope[i] : NaN);
        const fluxAbs = Math.abs(Number.isFinite(jFlux[i]) ? jFlux[i] : NaN);
        const inMidBand = Number.isFinite(S_now) && S_now >= sNeutralLo && S_now <= sNeutralHi;
        const tinySlope = Number.isFinite(slopeAbs) && slopeAbs < 0.005;
        const tinyFlux = Number.isFinite(fluxAbs) && fluxAbs < 0.03;
        const clampCandidate = inMidBand && tinySlope && tinyFlux;
        stabNeutRun = clampCandidate ? (stabNeutRun + 1) : 0;
      }

      // Sticky 로직: 중립보다 On/Off 지속을 우선, 전환은 확인일수 요구
      let decided = prevState;
      if (prevState === 1) {
        // On 유지. Up-shock lead 중에는 Off에도 최소 확인일 요구
        if (rawOff) {
          const upLeadActive = (stabLeadOnLeft > 0) || (sUpSeq >= Math.max(1, Number.isFinite(stab.lagUp) ? Math.floor(stab.lagUp) : 3));
          if (variant === 'stab' && upLeadActive) {
            const need = Math.max(1, Number.isFinite(stab.upConfirmOffMin) ? Math.floor(stab.upConfirmOffMin) : 2);
            decided = (offCand >= need) ? -1 : 1;
          } else {
            decided = -1;
          }
        } else {
          decided = 1;
        }
      } else if (prevState === -1) {
        // Off 유지, On은 2일 연속 확인
        if (onCand >= confirmOnDays) decided = 1;
        else decided = -1;
      } else {
        // Neutral에서는 Off가 우선, 아니면 On 2일 확인
        if (offCand >= 1) decided = -1;
        else if (onCand >= confirmOnDays) decided = 1;
        else {
          // EXP tie-breaker: use Flux sign when undecided
          if (variant === 'exp') {
            decided = (Number.isFinite(jFlux[i]) && Math.abs(jFlux[i]) > 0) ? (jFlux[i] > 0 ? 1 : -1) : 0;
          } else {
            decided = 0;
          }
        }
      }

      // Long drift makes entire regime Off
      if (inDriftEpoch) decided = -1;
      // Apply neutral clamp after decision (takes priority only in stab)
      if (variant === 'stab' && stabNeutRun >= 2) decided = 0;
      stateArr[i] = decided;
      fragile[i] = decided >= 0 && (guardValue >= th.mmFragile || (guardValue >= guardSoft && guardValue < guardHard));
      holdDays = decided === prevState ? holdDays + 1 : 0;
      prevState = decided;
    } else {
      const on = scoreVal >= dynScoreOn &&
        fluxVal >= dynOnFlux &&
        (comboValue ?? 0) > -0.005 &&
        (breadthValue ?? 0) >= th.breadthOn &&
        guardVal < 0.90;
      const off = scoreVal <= dynScoreOff ||
        fluxVal <= dynOffFlux ||
        (comboValue ?? 0) <= -0.03 ||
        mmValue >= th.mmOff ||
        guardVal >= 1.0;
      if (on) {
        stateArr[i] = 1;
        fragile[i] = guardVal >= 0.55;
      } else if (off) {
        stateArr[i] = -1;
      } else {
        stateArr[i] = 0;
        fragile[i] = guardVal >= 0.80 && fluxVal >= 0 && (comboValue ?? 0) >= -0.02;
      }
    }

    far[i] = Number.isFinite(jFlux[i]) && Number.isFinite(mmValue) && mmValue > 0
      ? Math.abs(jFlux[i]) / (mmValue + 1e-9)
      : null;
  }

  const executedState = stateArr.map((value, idx) => (idx === 0 ? 0 : stateArr[idx - 1] || 0));
  const diagnostics = {
    fluxThresholds: { on: dynOnFlux, off: dynOffFlux },
    scoreThresholds: { on: dynScoreOn, off: dynScoreOff },
    scoreLatest: score.length > 0 ? score[score.length - 1] ?? null : null,
  };

  return {
    dates,
    score,
    riskScore: score,
    state: stateArr,
    executedState,
    fragile,
    guard,
    mm,
    far,
    fflFlux: jFlux,
    riskBetaFlux: jRiskBeta,
    apdf,
    pcon,
    diffusionScore,
    fluxSlope,
    mmTrend,
    fullFlux,
    fullFluxZ,
    fluxRaw,
    fluxIntensity,
    comboMomentum,
    breadth,
    vPC1,
    kappa,
    scCorr,
    safeNeg,
    diagnostics,
  };
}

function rollingReturnFromSeries(series, index, lookback) {
  if (!Array.isArray(series) || !Number.isInteger(index) || !Number.isInteger(lookback) || lookback <= 0) {
    return null;
  }
  if (index >= series.length || index - lookback < 0) {
    return null;
  }
  const end = series[index];
  const start = series[index - lookback];
  if (!Number.isFinite(end) || !Number.isFinite(start) || start === 0) {
    return null;
  }
  return end / start - 1;
}

function rollingStdFromSeries(series, index, lookback) {
  if (!Array.isArray(series) || !Number.isInteger(index) || !Number.isInteger(lookback) || lookback < 2) {
    return null;
  }
  if (index >= series.length || index - lookback < 0) {
    return null;
  }
  const start = Math.max(0, index - lookback);
  const windowReturns = [];
  for (let i = start + 1; i <= index; i += 1) {
    const prev = series[i - 1];
    const cur = series[i];
    if (!Number.isFinite(prev) || !Number.isFinite(cur) || prev === 0) {
      continue; // eslint-disable-line no-continue
    }
    windowReturns.push(cur / prev - 1);
  }
  if (windowReturns.length < 2) {
    return null;
  }
  const meanReturn = windowReturns.reduce((acc, value) => acc + value, 0) / windowReturns.length;
  const variance = windowReturns.reduce((acc, value) => acc + ((value - meanReturn) ** 2), 0) / (windowReturns.length - 1);
  return Math.sqrt(Math.max(variance, 0));
}

function zMomentumFromSeries(series, index, k, v, zSat = 2.0) {
  if (!Array.isArray(series) || !Number.isInteger(index) || !Number.isInteger(k) || !Number.isInteger(v)) {
    return null;
  }
  if (k <= 0 || v <= 1 || index >= series.length || index - k < 0) {
    return null;
  }
  const base = series[index - k];
  const latest = series[index];
  if (!Number.isFinite(base) || !Number.isFinite(latest) || base === 0) {
    return null;
  }
  const momentum = latest / base - 1;
  const std = rollingStdFromSeries(series, index, v);
  if (!Number.isFinite(momentum) || !Number.isFinite(std) || std === 0) {
    return null;
  }
  return Math.tanh((momentum / std) / zSat);
}

function computeRiskBreadth(priceIndex, lookback) {
  if (!Number.isInteger(priceIndex)) return null;
  const prices = state.priceSeries || {};
  let considered = 0;
  let positive = 0;
  RISK_BREADTH_SYMBOLS.forEach((symbol) => {
    const ret = rollingReturnFromSeries(prices[symbol], priceIndex, lookback);
    if (!Number.isFinite(ret)) return;
    considered += 1;
    if (ret > 0) {
      positive += 1;
    }
  });
  if (considered === 0) return null;
  return positive / considered;
}

// FLL-Fusion: mix Classic and FFL+STAB by rolling Hit-rate and IC with softmax weights, guard by Absorption.
function computeRiskSeriesFLLFusion(metrics, recordsOverride, opts = {}) {
  if (!metrics || !Array.isArray(metrics.records) || metrics.records.length === 0) return null;
  const records = Array.isArray(recordsOverride) && recordsOverride.length > 0 ? recordsOverride : metrics.records;
  const cfg = Object.assign({ win: 40, lam: 0.50, tau: 4.0, floor: 0.10, onThr: +0.20, offThr: -0.20 }, opts || {});

  // 1) Individual series aligned to the same filtered dates
  const classic = computeRiskSeriesClassic(metrics, records);
  if (!classic) return null;
  // Force FFL variant = STAB without changing UI state
  const prevMode = state.riskMode;
  state.riskMode = 'ffl_stab';
  const fflStab = computeRiskSeriesFFL(metrics, records);
  state.riskMode = prevMode;
  if (!fflStab) return null;

  const dates = classic.dates || fflStab.dates || [];
  const n = dates.length;
  if (n === 0) return null;

  // 2) Align QQQ returns to dates
  const pxQQQ = state.priceSeries?.[SIGNAL.trade.baseSymbol] || [];
  if (!Array.isArray(pxQQQ) || pxQQQ.length < 2) return null;
  const rQQQfull = toReturns(pxQQQ); // length = analysisDates.length - 1
  const firstDate = dates[0];
  const gIdx = (state.analysisDates || []).indexOf(firstDate);
  const startRetIdx = Math.max(0, gIdx - 1);
  let rQQQ = rQQQfull.slice(startRetIdx, startRetIdx + n);
  if (rQQQ.length < n) {
    // pad front with zeros if needed
    rQQQ = new Array(n - rQQQ.length).fill(0).concat(rQQQ);
  }

  // 3) Predictions and one-day execution lag
  const predC_now = (classic.state || []).map((s) => (Number.isFinite(s) ? (s > 0 ? 1 : (s < 0 ? -1 : 0)) : 0));
  const predF_now = (fflStab.state || []).map((s) => (Number.isFinite(s) ? (s > 0 ? 1 : (s < 0 ? -1 : 0)) : 0));
  const predC = [0, ...predC_now.slice(0, n - 1)];
  const predF = [0, ...predF_now.slice(0, n - 1)];

  // 4) Rolling hit-rate and IC
  function rollingHit(pred, r, W) {
    let hits = 0; let tot = 0; const buf = []; const out = new Array(r.length).fill(0.5);
    for (let t = 0; t < r.length; t += 1) {
      const p = pred[t]; const y = Math.sign(r[t]);
      if (p !== 0) { const ok = (Math.sign(p) === y) ? 1 : 0; buf.push([ok, 1]); hits += ok; tot += 1; } else { buf.push([0, 0]); }
      if (buf.length > W) { const [okRem, totRem] = buf.shift(); hits -= okRem; tot -= totRem; }
      out[t] = tot > 0 ? (hits / tot) : 0.5;
    }
    return out;
  }
  function rollingIC(pred, r, W) {
    const out = new Array(r.length).fill(0); const bx = []; const by = [];
    let sumXY = 0, sumR = 0, sumR2 = 0;
    for (let t = 0; t < r.length; t += 1) {
      const x = pred[t]; const y = r[t];
      bx.push(x); by.push(y);
      sumXY += (x * y); sumR += y; sumR2 += y * y;
      if (bx.length > W) { const x0 = bx.shift(); const y0 = by.shift(); sumXY -= (x0 * y0); sumR -= y0; sumR2 -= y0 * y0; }
      const nWin = by.length; const mu = nWin ? (sumR / nWin) : 0;
      const varR = Math.max(1e-12, (sumR2 / nWin) - mu * mu); const sd = Math.sqrt(varR);
      out[t] = sd > 0 ? (sumXY / nWin) / sd : 0;
    }
    return out;
  }
  function softmax2(a, b, tau, floor) {
    const ezA = Math.exp(tau * a), ezB = Math.exp(tau * b);
    let wA = ezA / (ezA + ezB), wB = 1 - wA;
    wA = Math.max(floor, wA); wB = Math.max(floor, wB);
    const s = wA + wB; return [wA / s, wB / s];
  }

  const hitC = rollingHit(predC, rQQQ, cfg.win);
  const hitF = rollingHit(predF, rQQQ, cfg.win);
  const icC = rollingIC(predC, rQQQ, cfg.win);
  const icF = rollingIC(predF, rQQQ, cfg.win);
  const scoreC = hitC.map((h, i) => cfg.lam * (h - 0.5) + (1 - cfg.lam) * icC[i]);
  const scoreF = hitF.map((h, i) => cfg.lam * (h - 0.5) + (1 - cfg.lam) * icF[i]);
  const wClassic = new Array(n).fill(0.5); const wFFL = new Array(n).fill(0.5);
  for (let i = 0; i < n; i += 1) { const [a, b] = softmax2(scoreC[i], scoreF[i], cfg.tau, cfg.floor); wClassic[i] = a; wFFL[i] = b; }

  // 5) Guard by absorption ratio (mm)
  const mmFragile = (RISK_CFG_FFL?.thresholds?.mmFragile ?? 0.88);
  const mmOff = (RISK_CFG_FFL?.thresholds?.mmOff ?? 0.96);
  const mm = fflStab.mm || classic.mm || new Array(n).fill(0);
  const guardFragile = mm.map((x) => Number.isFinite(x) && x >= mmFragile);
  const guardHardOff = mm.map((x) => Number.isFinite(x) && x >= mmOff);

  // 6) Fused raw and state
  const fusedRaw = new Array(n).fill(0);
  for (let i = 0; i < n; i += 1) fusedRaw[i] = wClassic[i] * predC_now[i] + wFFL[i] * predF_now[i];
  const stateArr = fusedRaw.map((v, i) => {
    if (guardHardOff[i]) return -1;
    if (guardFragile[i] && v > 0) return 0;
    if (v >= cfg.onThr) return 1;
    if (v <= cfg.offThr) return -1;
    return 0;
  });
  const executedState = stateArr.map((val, idx) => (idx === 0 ? 0 : stateArr[idx - 1] || 0));
  const score = fusedRaw.map((v) => Math.max(0, Math.min(1, (v + 1) / 2)));

  return {
    dates,
    state: stateArr,
    executedState,
    score,
    wClassic,
    wFFL,
    mm,
    fragile: guardFragile,
    scCorr: classic.scCorr || fflStab.scCorr,
    safeNeg: classic.safeNeg || fflStab.safeNeg,
  };
}

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

function normalizeRangeSafe(value, min, max) {
  if (!Number.isFinite(value) || max <= min) return 0;
  if (value <= min) return 0;
  if (value >= max) return 1;
  return (value - min) / (max - min);
}

function averageFinite(...values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return null;
  return finite.reduce((acc, value) => acc + value, 0) / finite.length;
}

function quantile(values, q) {
  if (!Array.isArray(values) || values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const pos = Math.min(Math.max(q, 0), 1) * (sorted.length - 1);
  const lower = Math.floor(pos);
  const upper = Math.ceil(pos);
  if (lower === upper) return sorted[lower];
  const weight = pos - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

// --- New math helpers for full-matrix flux guard ---
function frobeniusDiff(matrix, prevMatrix) {
  if (!Array.isArray(matrix) || !Array.isArray(prevMatrix)) return null;
  let sumSq = 0;
  let count = 0;
  const n = Math.min(matrix.length, prevMatrix.length);
  for (let i = 0; i < n; i += 1) {
    const row = matrix[i];
    const prow = prevMatrix[i];
    if (!Array.isArray(row) || !Array.isArray(prow)) continue; // eslint-disable-line no-continue
    const m = Math.min(row.length, prow.length);
    for (let j = 0; j < m; j += 1) {
      const a = Number(row[j]);
      const b = Number(prow[j]);
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue; // eslint-disable-line no-continue
      const d = a - b;
      sumSq += d * d;
      count += 1;
    }
  }
  if (count === 0) return null;
  return Math.sqrt(sumSq / count);
}

function rollingMeanVariance(arr, index, lookback) {
  if (!Array.isArray(arr) || !Number.isInteger(index) || index < 0) return null;
  if (!Number.isInteger(lookback) || lookback <= 1) return null;
  const start = Math.max(0, index - lookback + 1);
  const window = [];
  for (let i = start; i <= index; i += 1) {
    const v = Number(arr[i]);
    if (Number.isFinite(v)) window.push(v);
  }
  if (window.length < 2) return null;
  const mean = window.reduce((a, b) => a + b, 0) / window.length;
  const variance = window.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (window.length - 1);
  return { mean, variance: Math.max(0, variance) };
}

function rollingZScore(arr, index, lookback) {
  const stats = rollingMeanVariance(arr, index, lookback);
  if (!stats) return null;
  const v = Number(arr[index]);
  const std = Math.sqrt(stats.variance);
  if (!Number.isFinite(v) || !Number.isFinite(std) || std === 0) return null;
  return (v - stats.mean) / std;
}

// Power-iteration top eigenvector for symmetric matrices (PC1 direction)
function topEigenvectorLocal(matrix, maxIter = 50, tol = 1e-6) {
  if (!Array.isArray(matrix) || matrix.length === 0) return null;
  const n = matrix.length;
  let v = new Array(n).fill(1 / Math.sqrt(n));
  for (let it = 0; it < maxIter; it += 1) {
    const w = new Array(n).fill(0);
    for (let i = 0; i < n; i += 1) {
      const row = matrix[i];
      let s = 0;
      for (let j = 0; j < n; j += 1) s += row[j] * v[j];
      w[i] = s;
    }
    const norm = Math.sqrt(w.reduce((a, x) => a + x * x, 0)) || 1;
    const nv = w.map((x) => x / norm);
    let diff = 0;
    for (let i = 0; i < n; i += 1) { const d = nv[i] - v[i]; diff += d * d; }
    v = nv;
    if (diff < tol * tol) break;
  }
  return v;
}

function sigmoid(x, slope = 1) {
  if (!Number.isFinite(x)) return null;
  const t = Math.max(Math.min(x * slope, 60), -60);
  return 1 / (1 + Math.exp(-t));
}

function leveragedReturn(baseReturn, leverage = 3) {
  if (!Number.isFinite(baseReturn)) return 0;
  const levered = leverage * baseReturn;
  // Prevent daily drop worse than -100% to avoid negative equity
  return Math.max(-0.99, levered);
}

async function maybeAutoRefreshAfterLoad() {
  if (state.autoRefreshAttempted || state.refreshing) {
    return;
  }

  state.autoRefreshAttempted = true;
  const refreshButton = document.getElementById('refresh-button');
  const originalLabel = refreshButton ? refreshButton.textContent : '';
  const { shouldFetch: willFetch } = evaluateRefreshNeeds();

  if (refreshButton && willFetch) {
    refreshButton.disabled = true;
    refreshButton.textContent = '갱신 중...';
  }

  try {
    if (willFetch) {
      state.refreshing = true;
    }
    const fetched = await maybeRefreshData();
    if (fetched) {
      renderAll();
    }
  } catch (error) {
    console.error('자동 리프레시 실패', error);
  } finally {
    if (refreshButton && willFetch) {
      refreshButton.disabled = false;
      refreshButton.textContent = originalLabel || '리프레시';
    }
    state.refreshing = false;
  }
}

async function handleRefreshClick() {
  const refreshButton = document.getElementById('refresh-button');
  if (state.refreshing) {
    return;
  }

  state.refreshing = true;
  const originalLabel = refreshButton ? refreshButton.textContent : '';
  if (refreshButton) {
    refreshButton.disabled = true;
    refreshButton.textContent = '갱신 중...';
  }

  try {
    const fetched = await maybeRefreshData();
    if (fetched) {
      hideError();
    }
    renderAll();
  } catch (error) {
    console.error(error);
    showError('최신 데이터를 불러오지 못했습니다. API 키와 네트워크 상태를 확인해 주세요.');
  } finally {
    if (refreshButton) {
      refreshButton.disabled = false;
      refreshButton.textContent = originalLabel || '리프레시';
    }
    state.refreshing = false;
  }
}

function handleDownloadReport(event) {
  if (event) {
    event.preventDefault();
  }
  try {
    const payload = buildTextReportPayload();
    triggerTextDownload(payload.text, payload.filename);
  } catch (error) {
    console.error('리포트 다운로드 실패', error);
    showError('리포트를 생성하지 못했습니다. 화면 데이터가 충분한지 확인한 뒤 다시 시도해 주세요.');
  }
}

function buildTextReportPayload() {
  const metrics = state.metrics[state.window];
  if (!metrics) {
    throw new Error('metrics-unavailable');
  }
  const { records, empty, rangeLabel } = getFilteredRecords(metrics);
  if (empty || !Array.isArray(records) || records.length === 0) {
    throw new Error('records-unavailable');
  }
  const latest = records[records.length - 1];
  if (!latest) {
    throw new Error('latest-record-missing');
  }
  const first = records[0];
  const generatedAt = state.generatedAt instanceof Date ? state.generatedAt : new Date();
  const rangeText = rangeLabel || `${state.range}일`;
  const riskSeries = state.riskMode === 'enhanced'
    ? computeRiskSeriesEnhanced(metrics, records)
    : (state.riskMode === 'ffl' || state.riskMode === 'ffl_exp' || state.riskMode === 'ffl_stab')
      ? computeRiskSeriesFFL(metrics, records)
      : computeRiskSeriesClassic(metrics, records);
  const headerLines = [
    '자산 결합 강도 TXT 리포트',
    '================================',
    `생성 시각: ${formatDateTimeLocal(generatedAt)}`,
    `데이터 기준일: ${latest.date || 'N/A'}`,
    `윈도우: ${state.window}일 | 표시 범위: ${rangeText} | 레짐 모드: ${state.riskMode === 'enhanced' ? 'Enhanced' : state.riskMode === 'ffl' ? 'FFL' : state.riskMode === 'ffl_exp' ? 'FFL+EXP' : state.riskMode === 'ffl_stab' ? 'FFL+STAB' : state.riskMode === 'fll_fusion' ? 'FLL-Fusion' : 'Classic'}`,
    '※ 결합 강도는 시장이 얼마나 동조화되어 있는지를 알려주는 맥락 지표이며, 실제 Risk-On/Off 판단은 모멘텀·Guard·Safe-NEG가 결합된 레짐 점수가 담당합니다. 값이 높을수록 자금이 한 방향으로 쏠린 것이므로 Guard·히트맵을 함께 확인하세요.',
    '',
    '[범위 요약]',
    `- 표본 구간: ${first?.date || 'N/A'} ~ ${latest?.date || 'N/A'} (${records.length}일)`,
    `- 사용자 선택: ${state.customRange?.start || '시작 미지정'} ~ ${state.customRange?.end || '종료 미지정'}`,
  ];

  const bodySections = [
    buildMethodologySection(),
    buildStabilitySection(records, metrics, rangeText),
    buildSubIndexSection(records),
    buildRiskScoreSection(riskSeries),
    buildRegimeTimelineSection(riskSeries),
    buildPriceSection(records),
    buildHeatmapSection(metrics),
    buildPairSection(),
  ].filter((section) => Array.isArray(section) && section.length > 0);

  const text = [
    ...headerLines,
    '',
    ...bodySections.flatMap((section, index) => (index === bodySections.length - 1 ? section : [...section, ''])),
  ].join('\n');

  const filename = buildReportFilename(latest?.date, generatedAt);
  return {
    text,
    filename,
  };
}

function triggerTextDownload(text, filename) {
  if (!text) {
    throw new Error('empty-text');
  }
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename || 'stability-report.txt';
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function buildReportFilename(latestDate, generatedAt) {
  const stampFromLatest = typeof latestDate === 'string'
    ? latestDate.replace(/[^0-9]/g, '')
    : '';
  const fallbackStamp = formatDateToken(generatedAt || new Date());
  const safeStamp = stampFromLatest && stampFromLatest.length >= 8 ? stampFromLatest.slice(0, 8) : fallbackStamp;
  return `stability-report-${safeStamp}.txt`;
}

function buildMethodologySection() {
  const assetList = SIGNAL.symbols.join(', ');
  return [
    '[0. 산출 로직 및 근거]',
    `- Stability Index = Σ w(i,j)·|corr(i,j)| ÷ Σ w(i,j), 대상 자산: ${assetList}, 주식-채권 쌍에 더 큰 가중을 적용합니다.`,
    '- 하위 지수: (a) 주식-암호화폐(+) = |corr(IWM/SPY, BTC)| 평균, (b) 전통자산(+) = |corr(IWM, SPY, TLT, GLD)| 평균, (c) Safe-NEG(-) = max(0, -corr(주식, TLT/GLD)) 평균.',
    '- Classic Risk Score = 0.70·max(0, corr(IWM, BTC)) + 0.30·Safe-NEG. corr ≥ 0.50 또는 Score ≥ 0.65 → Risk-On, corr ≤ -0.05 또는 Score ≤ 0.30 → Risk-Off.',
    '- Enhanced Mode = Classic Score + 10일 공동 모멘텀(IWM & BTC) + 5일 리스크 폭(IWM·SPY·BTC 상승비중) + Absorption/안정성 Guard. Guard ≥ 0.9 또는 Combo ≤ 0%면 On이 차단됩니다.',
    '- FFL Mode = Classic 점수에 Safe↔Risk 플럭스(J_norm), Flux Intensity, FAR, Guard를 조합한 Classic+Flux 레짐입니다.',
    '- FFL+EXP Mode = Classic(현재 레짐) + Flux(J_norm, 전환 감지) + Stability(시장 결합도, EMA/레벨) 3요소를 단순 결합합니다. 다수결로 On/Off를 결정(동률이면 Flux 우선), 점수는 세 값을 단순 평균합니다. 별도 수식은 추가하지 않습니다.',
    '- 기본 동작은 Classic이며 사용자가 명시적으로 변경한 경우에만 Enhanced/FFL이 적용됩니다.',
    `- 히트맵과 Absorption Ratio는 ${state.window}일 롤링 상관행렬·1차 고유값 비중으로 계산하며, 동일 데이터가 레짐 Guard에도 쓰입니다.`,
    `- 지수 추이 표는 신호 주지수(${SIGNAL.primaryStock})와 벤치마크(${SIGNAL.trade.baseSymbol})의 종가/일간 수익률을 그대로 나열해 백테스트 결과를 재현할 수 있도록 합니다.`,
  ];
}

function buildStabilitySection(records, metrics, rangeText) {
  const lines = ['[1. 자산 결합 강도 (Stability Index)]'];
  if (!Array.isArray(records) || records.length === 0) {
    lines.push('- 데이터가 없습니다.');
    return lines;
  }
  const latest = records[records.length - 1];
  lines.push(`- 최신 Stability: ${formatNumberOrNA(latest.stability)} | Smoothed(EMA10): ${formatNumberOrNA(latest.smoothed)} | Delta(3-10): ${formatSignedNumber(latest.delta)}`);
  lines.push(`- 180일 평균: ${formatNumberOrNA(metrics?.average180)} | 관측 범위: ${records[0].date} ~ ${latest.date} (${rangeText})`);
  const tableRows = records.map((rec) => [
    rec.date,
    formatNumberOrNA(rec.stability),
    formatNumberOrNA(rec.smoothed),
    formatSignedNumber(rec.delta),
  ]);
  lines.push(...formatTable(['날짜', 'Stability', 'Smoothed', 'Delta(3-10)'], tableRows));
  return lines;
}

function buildSubIndexSection(records) {
  const lines = ['[2. 하위 지수]'];
  if (!Array.isArray(records) || records.length === 0) {
    lines.push('- 데이터가 없습니다.');
    return lines;
  }
  const tableRows = records.map((rec) => [
    rec.date,
    formatNumberOrNA(rec.sub?.stockCrypto),
    formatNumberOrNA(rec.sub?.traditional),
    formatNumberOrNA(rec.sub?.safeNegative),
  ]);
  lines.push(...formatTable(['날짜', '주식-암호화폐(+)', '전통자산(+)', 'Safe-NEG(-)'], tableRows));
  return lines;
}

function buildRiskScoreSection(riskSeries) {
  const lines = ['[3. 리스크온 점수]', '- 결합 강도, Safe-NEG(안전자산 역동), 모멘텀, 리스크 폭, Guard(흡수비·안정성) 등을 결합한 실제 행동 신호입니다. 결합 강도만으로 On/Off가 결정되지 않습니다.'];
  if (!riskSeries || !Array.isArray(riskSeries.dates) || riskSeries.dates.length === 0 || !Array.isArray(riskSeries.state)) {
    lines.push('- 레짐 데이터를 계산하지 못했습니다.');
    return lines;
  }

  // Always build a regime-agnostic, unified table so columns never change.
  // Fill missing metrics from overlay series (FFL/Enhanced/Classic) when available.
  const metrics = state.metrics[state.window];
  const { records } = getFilteredRecords(metrics);
  const classicAll = computeRiskSeriesClassic(metrics, records) || {};
  const enhancedAll = computeRiskSeriesEnhanced(metrics, records) || {};
  const fflAll = computeRiskSeriesFFL(metrics, records) || {};

  const headers = [
    '날짜',
    '상태',
    'State',     // numeric state code (-1/0/1)
    'Exec',      // executed state (T+1, numeric)
    '점수',
    'Corr',
    'Safe-NEG',
    'Guard',
    'Absorption',
    'J_norm',
    'FINT',
    'FAR',
    'RB_Flux',
    'ΔCorr-Z',
    'Diff',
    'mmΔ',
    'APDF',
    'PCON',
    'v_PC1',
    'κ(diff↔drift)',
    'Combo(Z)',
    'Breadth',
  ];

  function pick(arrName, idx) {
    const a = riskSeries?.[arrName]?.[idx];
    if (a != null && Number.isFinite(a)) return a;
    const b = fflAll?.[arrName]?.[idx];
    if (b != null && Number.isFinite(b)) return b;
    const c = enhancedAll?.[arrName]?.[idx];
    if (c != null && Number.isFinite(c)) return c;
    const d = classicAll?.[arrName]?.[idx];
    if (d != null && Number.isFinite(d)) return d;
    return null;
  }

  const rows = riskSeries.dates.map((date, idx) => {
    const st = Number.isFinite(riskSeries.state?.[idx]) ? riskSeries.state[idx] : (Number.isFinite(fflAll.state?.[idx]) ? fflAll.state[idx] : (Number.isFinite(enhancedAll.state?.[idx]) ? enhancedAll.state[idx] : (Number.isFinite(classicAll.state?.[idx]) ? classicAll.state[idx] : 0)));
    const label = formatRegimeLabel(st, riskSeries.fragile?.[idx] || fflAll.fragile?.[idx] || enhancedAll.fragile?.[idx]);
    const exec = Number.isFinite(riskSeries.executedState?.[idx])
      ? riskSeries.executedState[idx]
      : Number.isFinite(fflAll.executedState?.[idx])
        ? fflAll.executedState[idx]
        : (idx === 0 ? 0 : (Number.isFinite(st) ? (riskSeries.state?.[idx - 1] ?? st) : 0));

    // Score: prefer regime's native score key
    const scoreVal = Number.isFinite(riskSeries.riskScore?.[idx]) ? riskSeries.riskScore[idx]
      : Number.isFinite(riskSeries.score?.[idx]) ? riskSeries.score[idx]
      : Number.isFinite(fflAll.score?.[idx]) ? fflAll.score[idx]
      : Number.isFinite(enhancedAll.riskScore?.[idx]) ? enhancedAll.riskScore[idx]
      : Number.isFinite(classicAll.score?.[idx]) ? classicAll.score[idx] : null;

    const scCorr = pick('scCorr', idx);
    const safeNeg = pick('safeNeg', idx);
    const guard = pick('guard', idx);
    const mm = pick('mm', idx);
    const jn = pick('fflFlux', idx);
    const fint = pick('fluxIntensity', idx);
    const far = pick('far', idx);
    const rb = pick('riskBetaFlux', idx);
    const zcorr = pick('fullFluxZ', idx);
    const diff = pick('diffusionScore', idx);
    const mmd = pick('mmTrend', idx);
    const apdf = pick('apdf', idx);
    const pcon = pick('pcon', idx);
    const vpc1 = pick('vPC1', idx);
    const kappa = pick('kappa', idx);
    const combo = pick('comboMomentum', idx);
    const br = pick('breadth', idx);

    return [
      date,
      label,
      Number.isFinite(st) ? st : '',
      Number.isFinite(exec) ? exec : '',
      formatNumberOrNA(scoreVal),
      formatNumberOrNA(scCorr),
      formatNumberOrNA(safeNeg),
      formatNumberOrNA(guard),
      formatNumberOrNA(mm),
      formatNumberOrNA(jn),
      formatNumberOrNA(fint),
      formatNumberOrNA(far),
      formatNumberOrNA(rb),
      formatNumberOrNA(zcorr),
      formatNumberOrNA(diff),
      formatNumberOrNA(mmd),
      formatNumberOrNA(apdf),
      (Number.isFinite(pcon) ? `${(pcon * 100).toFixed(0)}%` : 'N/A'),
      formatNumberOrNA(vpc1),
      (Number.isFinite(kappa) ? `${(kappa * 100).toFixed(0)}%` : 'N/A'),
      formatSignedPercent(combo),
      formatPercentOrNA(br),
    ];
  });

  lines.push(...formatTable(headers, rows));
  return lines;
}

function buildRegimeTimelineSection(riskSeries) {
  const lines = ['[4. 레짐 타임라인]'];
  if (
    !riskSeries ||
    !Array.isArray(riskSeries.dates) ||
    !Array.isArray(riskSeries.state) ||
    riskSeries.dates.length === 0
  ) {
    lines.push('- 레짐 데이터를 계산하지 못했습니다.');
    return lines;
  }
  const transitions = [];
  let prev = null;
  for (let i = 0; i < riskSeries.state.length; i += 1) {
    const current = riskSeries.state[i];
    if (prev === null || current !== prev) {
      transitions.push(formatRegimeTransitionLine(riskSeries, i));
      prev = current;
    }
  }
  if (transitions.length === 0) {
    lines.push('- 표시 범위 내 상태 변화가 없습니다.');
  } else {
    lines.push(...transitions);
  }
  return lines;
}

function buildPriceSection(records) {
  const lines = ['[5. 지수 추이 (자산 가격)]'];
  if (!Array.isArray(records) || records.length === 0) {
    lines.push('- 데이터가 없습니다.');
    return lines;
  }
  const signalSymbol = SIGNAL.primaryStock;
  const benchmarkSymbol = SIGNAL.trade.baseSymbol;
  lines.push(`- 대상: 신호 주지수 ${signalSymbol}, 벤치마크 ${benchmarkSymbol}`);
  const rows = records.map((rec) => {
    const date = rec.date;
    const signalPrice = lookupPriceByDate(signalSymbol, date);
    const benchmarkPrice = lookupPriceByDate(benchmarkSymbol, date);
    const signalRet = computeDailyReturnForSymbol(signalSymbol, date);
    const benchmarkRet = computeDailyReturnForSymbol(benchmarkSymbol, date);
    return [
      date,
      formatNumberOrNA(signalPrice, 2),
      formatSignedPercent(signalRet),
      formatNumberOrNA(benchmarkPrice, 2),
      formatSignedPercent(benchmarkRet),
    ];
  });
  lines.push(...formatTable(
    ['날짜', `${signalSymbol} 종가`, `${signalSymbol} 일간`, `${benchmarkSymbol} 종가`, `${benchmarkSymbol} 일간`],
    rows,
  ));
  return lines;
}

function buildHeatmapSection(metrics) {
  const lines = ['[6. 히트맵 매트릭스]'];
  if (!metrics || !Array.isArray(metrics.records) || metrics.records.length === 0) {
    lines.push('- 상관행렬 데이터를 찾지 못했습니다.');
    return lines;
  }
  const targetDate = state.heatmapDate
    || metrics.records[metrics.records.length - 1]?.date
    || '';
  const record = metrics.records.find((item) => item.date === targetDate)
    || metrics.records[metrics.records.length - 1];
  if (!record || !Array.isArray(record.matrix)) {
    lines.push('- 상관행렬 데이터를 찾지 못했습니다.');
    return lines;
  }
  const labels = SIGNAL.symbols.slice();
  lines.push(`- 기준일: ${record.date}`);
  const rows = labels.map((rowSymbol, rowIdx) => {
    const rowValues = labels.map((_, colIdx) => {
      const value = record.matrix?.[rowIdx]?.[colIdx];
      return Number.isFinite(value) ? value.toFixed(3) : 'N/A';
    });
    return [rowSymbol, ...rowValues];
  });
  lines.push(...formatTable(['자산 \\ 자산', ...labels], rows));
  return lines;
}

function buildPairSection() {
  const lines = ['[7. 기준자산 상관도]'];
  const pair = state.pair || DEFAULT_PAIR;
  const [assetA, assetB] = pair.split('|');
  lines.push(`- 선택한 페어: ${assetA} / ${assetB}`);
  const pairSeries = computePairSeries(assetA, assetB);
  if (!pairSeries) {
    lines.push('- 페어 데이터를 계산하지 못했습니다.');
    return lines;
  }
  const indices = computeVisibleIndices(pairSeries.dates);
  if (!Array.isArray(indices) || indices.length === 0) {
    lines.push('- 표시 구간에 해당하는 데이터가 없습니다.');
    return lines;
  }
  const rows = indices.map((idx) => [
    pairSeries.dates[idx],
    formatNumberOrNA(pairSeries.priceA?.[idx], 2),
    formatNumberOrNA(pairSeries.priceB?.[idx], 2),
    formatNumberOrNA(pairSeries.correlation?.[idx]),
  ]);
  lines.push(...formatTable(
    ['날짜', `${assetA} 가격`, `${assetB} 가격`, `${state.window}일 롤링 상관`],
    rows,
  ));
  return lines;
}

function formatRegimeTransitionLine(riskSeries, idx) {
  const label = formatRegimeLabel(riskSeries.state?.[idx], riskSeries.fragile?.[idx]);
  const scoreValue = state.riskMode === 'enhanced'
    ? formatNumberOrNA(riskSeries.riskScore?.[idx])
    : formatNumberOrNA(riskSeries.score?.[idx]);
  const corrValue = formatNumberOrNA(riskSeries.scCorr?.[idx]);
  const safeValue = formatNumberOrNA(riskSeries.safeNeg?.[idx]);
  let extras = '';
  if (state.riskMode === 'enhanced') {
    const guard = formatNumberOrNA(riskSeries.guard?.[idx]);
    const mm = formatNumberOrNA(riskSeries.mm?.[idx]);
    const combo = formatSignedPercent(riskSeries.comboMomentum?.[idx]);
    extras = `, Guard=${guard}, Absorption=${mm}, Combo=${combo}`;
  } else if (state.riskMode === 'ffl') {
    const flux = formatNumberOrNA(riskSeries.fflFlux?.[idx]);
    const fint = formatNumberOrNA(riskSeries.fluxIntensity?.[idx]);
    const far = formatNumberOrNA(riskSeries.far?.[idx]);
    const guard = formatNumberOrNA(riskSeries.guard?.[idx]);
    const mm = formatNumberOrNA(riskSeries.mm?.[idx]);
    const combo = formatSignedPercent(riskSeries.comboMomentum?.[idx]);
    const breadth = formatPercentOrNA(riskSeries.breadth?.[idx]);
    extras = `, J_norm=${flux}, FINT=${fint}, FAR=${far}, Guard=${guard}, Absorption=${mm}, Combo=${combo}, Breadth=${breadth}`;
  } else if (state.riskMode === 'fll_fusion') {
    const wC = Number.isFinite(riskSeries.wClassic?.[idx]) ? `${(riskSeries.wClassic[idx] * 100).toFixed(0)}%` : 'N/A';
    const wF = Number.isFinite(riskSeries.wFFL?.[idx]) ? `${(riskSeries.wFFL[idx] * 100).toFixed(0)}%` : 'N/A';
    const mm = formatNumberOrNA(riskSeries.mm?.[idx]);
    extras = `, wC=${wC}, wFFL=${wF}, Absorption=${mm}`;
  }
  return `- ${riskSeries.dates[idx]} · ${label} (Score=${scoreValue}, Corr=${corrValue}, Safe-NEG=${safeValue}${extras})`;
}

function formatRegimeLabel(value, fragile) {
  if (value > 0) {
    return fragile ? 'Risk-On (Fragile)' : 'Risk-On';
  }
  if (value < 0) {
    return 'Risk-Off';
  }
  return fragile ? 'Neutral (Fragile)' : 'Neutral';
}

function formatTable(headers, rows) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return ['(데이터 없음)'];
  }
  const safeHeaders = headers.map((header) => header || '');
  const output = [];
  output.push(safeHeaders.join('\t'));
  rows.forEach((row) => {
    const safeRow = (row || []).map((value) => (value == null ? '' : String(value)));
    output.push(safeRow.join('\t'));
  });
  return output;
}

function lookupPriceByDate(symbol, date) {
  if (!symbol || !date) return null;
  const dates = state.analysisDates || [];
  const idx = dates.indexOf(date);
  if (idx < 0) return null;
  const series = state.priceSeries?.[symbol];
  if (!Array.isArray(series)) return null;
  const price = series[idx];
  return Number.isFinite(price) ? price : null;
}

function computeDailyReturnForSymbol(symbol, date) {
  if (!symbol || !date) return null;
  const dates = state.analysisDates || [];
  const idx = dates.indexOf(date);
  if (idx <= 0) return null;
  const series = state.priceSeries?.[symbol];
  if (!Array.isArray(series)) return null;
  const current = series[idx];
  const prev = series[idx - 1];
  if (!Number.isFinite(current) || !Number.isFinite(prev) || prev === 0) return null;
  return current / prev - 1;
}

function formatNumberOrNA(value, digits = 3) {
  if (!Number.isFinite(value)) return 'N/A';
  return Number(value).toFixed(digits);
}

function formatSignedNumber(value, digits = 3) {
  if (!Number.isFinite(value)) return 'N/A';
  const fixed = Number(value).toFixed(digits);
  return value > 0 ? `+${fixed}` : fixed;
}

function formatPercentOrNA(value, digits = 0) {
  if (!Number.isFinite(value)) return 'N/A';
  return `${(value * 100).toFixed(digits)}%`;
}

function formatSignedPercent(value, digits = 1) {
  if (!Number.isFinite(value)) return 'N/A';
  const fixed = (value * 100).toFixed(digits);
  return value > 0 ? `+${fixed}%` : `${fixed}%`;
}

function formatDateTimeLocal(value) {
  const date = value instanceof Date ? value : new Date(value);
  if (!date || Number.isNaN(date.getTime())) return '알 수 없음';
  return date.toLocaleString(undefined, { hour12: false });
}

function formatDateToken(value) {
  const date = value instanceof Date ? value : new Date(value);
  if (!date || Number.isNaN(date.getTime())) return 'latest';
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, '0');
  const day = String(date.getUTCDate()).padStart(2, '0');
  return `${year}${month}${day}`;
}

function canBuildTextReport() {
  try {
    const metrics = state.metrics[state.window];
    if (!metrics) return false;
    const { records, empty } = getFilteredRecords(metrics);
    if (empty || !Array.isArray(records) || records.length === 0) return false;
    return true;
  } catch (error) {
    return false;
  }
}

function updateDownloadButtonState() {
  const button = document.getElementById('download-report');
  if (!button) return;
  button.disabled = !canBuildTextReport();
}

async function maybeRefreshData() {
  if (!isLocalhost()) {
    return false;
  }

  if (!state.fmpKey) {
    await hydrateFmpKeyFromEnvironment();
  }

  const { shouldFetch } = evaluateRefreshNeeds();

  if (!shouldFetch) {
    return false;
  }

  if (!state.fmpKey) {
    showNotice('FMP API 키를 찾을 수 없어 자동 갱신을 건너뜁니다. GitHub Actions/Pages Secrets 또는 환경 변수 FMP_API_KEY를 설정해 주세요.');
    return false;
  }

  showNotice('Financial Modeling Prep에서 최신 데이터를 불러오는 중입니다...');
  await loadFromFmp(state.fmpKey);
  hideError();
  return true;
}

function evaluateRefreshNeeds() {
  const metrics = state.metrics[state.window];
  const records = metrics?.records || [];
  const earliest = state.analysisDates?.[0]
    || records[0]?.date
    || null;
  const latest = state.analysisDates?.[state.analysisDates.length - 1]
    || records[records.length - 1]?.date
    || null;

  const requiredCutoff = computeFmpCutoffDate();
  const needsRangeCoverage = !earliest || earliest > requiredCutoff;
  const needsFreshLatest = isLatestDateStale(latest);

  return {
    metrics,
    earliest,
    latest,
    needsRangeCoverage,
    needsFreshLatest,
    shouldFetch: needsRangeCoverage || needsFreshLatest || !metrics,
  };
}

function updateMeta() {
  const metrics = state.metrics[state.window];
  if (!metrics) return;
  const latestRecord = metrics.records[metrics.records.length - 1];
  const lastUpdated = document.getElementById('last-updated');
  const generatedAt = document.getElementById('generated-at');
  lastUpdated.textContent = `데이터 기준일: ${latestRecord.date}`;
  const localeTime = state.generatedAt.toLocaleString(undefined, { hour12: false });
  generatedAt.textContent = `계산 시각: ${localeTime}`;
}

function getFilteredRecords(metrics) {
  const fallback = { records: [], empty: true, feedbackMessage: '', rangeLabel: '' };
  if (!metrics) return fallback;

  const customRange = state.customRange || { start: null, end: null, valid: true };
  const startTime = parseDateSafe(customRange.start);
  const endTime = parseDateSafe(customRange.end);
  const hasCustomRange = customRange.valid && (startTime !== null || endTime !== null);

  if (hasCustomRange) {
    const filtered = metrics.records.filter((item) => {
      const time = parseDateSafe(item.date);
      if (time === null) return false;
      if (startTime !== null && time < startTime) return false;
      if (endTime !== null && time > endTime) return false;
      return true;
    });

    if (filtered.length === 0) {
      return {
        records: [],
        empty: true,
        feedbackMessage: '선택한 기간에 해당하는 데이터가 없습니다.',
        rangeLabel: '맞춤 기간',
      };
    }

    const dayCount = computeCustomRangeDays(startTime, endTime, filtered.length);
    const label = typeof dayCount === 'number' && dayCount > 0
      ? `맞춤 ${dayCount}일`
      : '맞춤 기간';

    return {
      records: filtered,
      empty: false,
      feedbackMessage: '맞춤 기간이 설정되어 있습니다.',
      rangeLabel: label,
    };
  }

  const rangeDays = state.range;
  const sliced = metrics.records.slice(-rangeDays);
  return {
    records: sliced,
    empty: sliced.length === 0,
    feedbackMessage: '',
    rangeLabel: `${rangeDays}일`,
  };
}

function renderGauge() {
  const metrics = state.metrics[state.window];
  if (!metrics) return;
  const element = document.getElementById('stability-gauge');
  if (!element) return;
  const chart = charts.stability || echarts.init(element);
  charts.stability = chart;

  const { records: filteredRecords, empty, feedbackMessage, rangeLabel } = getFilteredRecords(metrics);

  if (empty) {
    chart.clear();
    setGaugePanelFeedback(feedbackMessage || '표시할 데이터가 없습니다.');
    return;
  }

  setGaugePanelFeedback(feedbackMessage || '');

  const latest = filteredRecords[filteredRecords.length - 1];
  const stabilityValues = filteredRecords.map((item) => safeNumber(item.stability));
  const averageWindowSize = Math.min(stabilityValues.length, 180);
  const averageWindow = stabilityValues.slice(-averageWindowSize);
  const average180 = mean(averageWindow);
  const shortEmaSeries = ema(stabilityValues, 3);
  const longEmaSeries = ema(stabilityValues, 10);
  const delta = shortEmaSeries.length > 0 && longEmaSeries.length > 0
    ? safeNumber(shortEmaSeries[shortEmaSeries.length - 1] - longEmaSeries[longEmaSeries.length - 1])
    : 0;

  const averageLabel = averageWindowSize > 0 ? `최근 ${averageWindowSize}일 평균` : '평균';
  const rangeDescriptor = rangeLabel || `${averageWindowSize}일`;

  // Regime summary (Classic/Enhanced/FFL/FLL-Fusion)
  const riskSeries = state.riskMode === 'enhanced'
    ? computeRiskSeriesEnhanced(state.metrics[state.window], filteredRecords)
    : (state.riskMode === 'fll_fusion')
      ? computeRiskSeriesFLLFusion(state.metrics[state.window], filteredRecords)
      : ((state.riskMode === 'ffl' || state.riskMode === 'ffl_exp' || state.riskMode === 'ffl_stab')
        ? computeRiskSeriesFFL(state.metrics[state.window], filteredRecords)
        : computeRiskSeriesClassic(state.metrics[state.window], filteredRecords));
  if (riskSeries && riskSeries.score.length > 0) {
    const idx = riskSeries.score.length - 1;
    const rState = riskSeries.state[idx];
    const rLabel = rState > 0 ? 'Risk-On' : rState < 0 ? 'Risk-Off' : 'Neutral';
    const rScore = safeNumber(riskSeries.score[idx]).toFixed(3);
    const modeLabel = state.riskMode === 'enhanced'
      ? 'Enhanced'
      : (state.riskMode === 'ffl' ? 'FFL'
      : (state.riskMode === 'ffl_exp' ? 'FFL+EXP'
      : (state.riskMode === 'ffl_stab' ? 'FFL+STAB'
      : (state.riskMode === 'fll_fusion' ? 'FLL-Fusion' : 'Classic'))));
    const summary = `${modeLabel} 레짐: ${rLabel} • 점수 ${rScore} • 창 ${state.window}일 • ${rangeDescriptor}`;
    setGaugePanelFeedback(summary);
  }

  chart.setOption({
      series: [{
        type: 'gauge',
        startAngle: 210,
        endAngle: -30,
        min: 0,
        max: 1,
        splitNumber: 10,
        axisLine: {
          lineStyle: {
            width: 20,
            color: [
              [BANDS.red[1], '#f87171'],
              [BANDS.yellow[1], '#facc15'],
              [BANDS.green[1], '#4ade80'],
            ],
          },
        },
        pointer: {
          length: '70%',
          width: 6,
        },
        detail: {
          formatter: () => `결합 강도: ${safeNumber(latest.stability).toFixed(3)}\n${averageLabel}: ${safeNumber(average180).toFixed(3)}\n추세(${rangeDescriptor}): ${delta >= 0 ? '▲' : '▼'} ${(Math.abs(delta)).toFixed(3)}`,
          fontSize: 16,
          color: TEXT_PRIMARY,
          backgroundColor: 'rgba(15,23,42,0.45)',
          padding: 8,
          borderRadius: 8,
          offsetCenter: [0, '65%'],
        },
        data: [{ value: safeNumber(latest.stability) }],
      }],
    });
  }

function renderSubGauges() {
  const metrics = state.metrics[state.window];
  if (!metrics) return;

  const { records: filteredRecords, empty } = getFilteredRecords(metrics);
  const mapping = [
    { key: 'stockCrypto', element: 'stock-crypto-gauge' },
    { key: 'traditional', element: 'traditional-gauge' },
    { key: 'safeNegative', element: 'safe-neg-gauge' },
  ];
  // Always expose FFL auxiliary gauges regardless of regime mode.
  const extraFFL = [
    { keyFrom: 'fflFlux', element: 'ffl-flux-gauge', min: -1, max: 1, formatter: (v) => safeNumber(v).toFixed(3) },
    { keyFrom: 'fluxIntensity', element: 'ffl-fint-gauge', min: 0, max: 2, formatter: (v) => safeNumber(v).toFixed(3) },
    { keyFrom: 'far', element: 'ffl-far-gauge', min: 0, max: 5, formatter: (v) => safeNumber(v).toFixed(3) },
  ];

  mapping.forEach(({ key, element }) => {
    const container = document.getElementById(element);
    if (!container) return;
    const chart = charts[element] || echarts.init(container);
    charts[element] = chart;

    if (empty) {
      chart.clear();
      return;
    }

    const latest = filteredRecords[filteredRecords.length - 1];
    const gaugeValue = safeNumber(latest.sub[key]);
    chart.setOption({
      series: [{
        type: 'gauge',
        startAngle: 210,
        endAngle: -30,
        min: 0,
        max: 1,
        splitNumber: 10,
        axisLine: {
          lineStyle: {
            width: 16,
            color: [
              [BANDS.red[1], '#f87171'],
              [BANDS.yellow[1], '#facc15'],
              [BANDS.green[1], '#4ade80'],
            ],
          },
        },
        pointer: {
          length: '65%',
          width: 4,
        },
        detail: {
          formatter: (value) => safeNumber(value).toFixed(3),
          fontSize: 14,
          color: TEXT_PRIMARY,
          offsetCenter: [0, '60%'],
        },
        data: [{ value: gaugeValue }],
      }],
    });
  });

  if (extraFFL.length === 0) {
    return;
  }

  const fflSeries = empty ? null : computeRiskSeriesFFL(metrics, filteredRecords);

  extraFFL.forEach(({ keyFrom, element, min, max, formatter }) => {
    const container = document.getElementById(element);
    if (!container) return;
    const chart = charts[element] || echarts.init(container);
    charts[element] = chart;

    if (empty || !fflSeries || !Array.isArray(fflSeries[keyFrom]) || fflSeries[keyFrom].length === 0) {
      chart.clear();
      return;
    }

    const latestIndex = fflSeries[keyFrom].length - 1;
    const rawValue = fflSeries[keyFrom][latestIndex];
    if (!Number.isFinite(rawValue)) {
      chart.clear();
      return;
    }

    chart.setOption({
      series: [{
        type: 'gauge',
        startAngle: 210,
        endAngle: -30,
        min,
        max,
        splitNumber: 10,
        axisLine: {
          lineStyle: {
            width: 16,
            color: [
              [0.5, '#f87171'],
              [0.75, '#facc15'],
              [1, '#4ade80'],
            ],
          },
        },
        pointer: {
          length: '65%',
          width: 4,
        },
        detail: {
          formatter: (value) => (typeof formatter === 'function' ? formatter(value) : safeNumber(value).toFixed(3)),
          fontSize: 14,
          color: TEXT_PRIMARY,
          offsetCenter: [0, '60%'],
        },
        data: [{ value: rawValue }],
      }],
    });
  });
}

function renderHistory() {
  const element = document.getElementById('history-chart');
  const chart = charts.history || echarts.init(element);
  charts.history = chart;

  const metrics = state.metrics[state.window];
  if (!metrics) return;

  const { records: series, feedbackMessage, empty } = getFilteredRecords(metrics);
  const riskSeries = state.riskMode === 'enhanced'
    ? computeRiskSeriesEnhanced(metrics, series)
    : (state.riskMode === 'ffl' || state.riskMode === 'ffl_exp' || state.riskMode === 'ffl_stab')
      ? computeRiskSeriesFFL(metrics, series)
      : computeRiskSeriesClassic(metrics, series);
  let markAreas = [];
  if (riskSeries && Array.isArray(riskSeries.state) && riskSeries.state.length > 0) {
    const segments = computeRegimeSegments(riskSeries.dates, riskSeries.state) || [];
    markAreas = segments.map((seg) => [
      { xAxis: seg.xAxis },
      { xAxis: seg.xAxis2, itemStyle: seg.itemStyle, name: seg.name },
    ]);
  }

  if (typeof feedbackMessage === 'string') {
    setCustomRangeFeedback(feedbackMessage);
  }

  if (empty) {
    chart.clear();
    return;
  }

  const dates = series.map((item) => item.date);
  const stabilityValues = series.map((item) => (Number.isFinite(item.stability) ? item.stability : null));
  const smoothedValues = series.map((item) => (Number.isFinite(item.smoothed) ? item.smoothed : null));
  const subConfigs = [
    { key: 'stockCrypto', label: '주식-암호화폐 (+)', color: '#f97316' },
    { key: 'traditional', label: '전통자산 (+)', color: '#38bdf8' },
    { key: 'safeNegative', label: '안전자산 결합력 (-)', color: '#a855f7' },
  ];
  const legendNames = ['결합 강도', '결합 강도 (EMA)', ...subConfigs.map((item) => item.label)];
  const legendSelected = { '결합 강도': true, '결합 강도 (EMA)': true };
  subConfigs.forEach((cfg) => {
    legendSelected[cfg.label] = false;
  });

  const subSeries = subConfigs.map((cfg) => ({
    name: cfg.label,
    type: 'line',
    data: series.map((item) => {
      const value = item?.sub?.[cfg.key];
      return Number.isFinite(value) ? Number(value.toFixed(6)) : null;
    }),
    smooth: true,
    showSymbol: false,
    lineStyle: {
      width: 1.5,
      type: 'dashed',
      color: cfg.color,
    },
    emphasis: { focus: 'series' },
  }));

  chart.setOption({
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        if (!Array.isArray(params) || params.length === 0) return '';
        const header = params[0]?.axisValueLabel || params[0]?.axisValue || '';
        const lines = params
          .map((item) => {
            const value = Number.isFinite(item.value) ? item.value : null;
            if (value === null) return null;
            return `${item.marker}${item.seriesName}: ${value.toFixed(3)}`;
          })
          .filter(Boolean)
          .join('<br/>');
        return lines ? `${header}<br/>${lines}` : header;
      },
    },
    legend: {
      data: legendNames,
      selected: legendSelected,
    },
    xAxis: {
      type: 'category',
      data: dates,
      axisLabel: { color: TEXT_AXIS },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      axisLabel: { color: TEXT_AXIS },
    },
    series: [
      {
        name: '결합 강도',
        type: 'line',
        data: stabilityValues,
        smooth: true,
        markArea: markAreas.length > 0 ? {
          silent: true,
          itemStyle: { opacity: 1 },
          data: markAreas,
        } : undefined,
      },
      {
        name: '결합 강도 (EMA)',
        type: 'line',
        data: smoothedValues,
        smooth: true,
        lineStyle: { width: 2, color: '#0ea5e9' },
      },
      ...subSeries,
    ],
  });
}

function renderHeatmap() {
  const element = document.getElementById('heatmap-chart');
  const slider = document.getElementById('heatmap-slider');
  const sliderLabel = document.getElementById('heatmap-slider-label');
  if (!element) {
    return;
  }
  const chart = charts.heatmap || echarts.init(element);
  charts.heatmap = chart;

  const metrics = state.metrics[state.window];
  if (!metrics) {
    chart.clear();
    if (slider) slider.disabled = true;
    if (sliderLabel) sliderLabel.textContent = '';
    return;
  }

  const { records, empty } = getFilteredRecords(metrics);
  if (empty) {
    chart.clear();
    if (slider) slider.disabled = true;
    if (sliderLabel) sliderLabel.textContent = '데이터 없음';
    return;
  }

  let targetIndex = records.length - 1;
  if (state.heatmapDate) {
    const matched = records.findIndex((rec) => rec.date === state.heatmapDate);
    if (matched >= 0) {
      targetIndex = matched;
    }
  }
  const targetRecord = records[targetIndex] || records[records.length - 1];
  state.heatmapDate = targetRecord.date;

  if (slider) {
    slider.min = 0;
    slider.max = Math.max(records.length - 1, 0);
    slider.value = targetIndex;
    slider.disabled = records.length <= 1;
    slider.oninput = (event) => {
      const idx = Number(event.target.value);
      const nextRecord = records[idx];
      if (!nextRecord) return;
      state.heatmapDate = nextRecord.date;
      renderHeatmap();
    };
  }
  if (sliderLabel) {
    sliderLabel.textContent = `선택 일자: ${targetRecord.date}`;
  }

  const matrix = Array.isArray(targetRecord.matrix) ? targetRecord.matrix : null;
  if (!matrix) {
    chart.clear();
    return;
  }

  const labels = SIGNAL.symbols.slice();
  const data = [];
  for (let i = 0; i < labels.length; i += 1) {
    for (let j = 0; j < labels.length; j += 1) {
      const value = i === j ? 1 : matrix[i][j];
      const display = Number.isFinite(value) ? Number(value.toFixed(3)) : 0;
      data.push([i, j, display]);
    }
  }

  chart.setOption({
    tooltip: {
      position: 'top',
      formatter: (params) => `${state.heatmapDate} · ${labels[params.data[1]]} / ${labels[params.data[0]]}: ${params.data[2]}`,
    },
    grid: {
      height: '80%',
      top: '10%',
    },
    xAxis: { type: 'category', data: labels, axisLabel: { color: TEXT_AXIS } },
    yAxis: { type: 'category', data: labels, axisLabel: { color: TEXT_AXIS } },
    visualMap: {
      min: -1,
      max: 1,
      show: false,
    },
    series: [{
      name: 'Correlation',
      type: 'heatmap',
      data,
      label: { show: true },
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowColor: 'rgba(0, 0, 0, 0.5)',
        },
      },
    }],
  });
}

function renderPair() {
  const priceElement = document.getElementById('pair-price-chart');
  const correlationElement = document.getElementById('pair-correlation-chart');
  if (!priceElement || !correlationElement) {
    return;
  }

  const priceChart = charts.pairPrice || echarts.init(priceElement);
  const correlationChart = charts.pairCorrelation || echarts.init(correlationElement);
  charts.pairPrice = priceChart;
  charts.pairCorrelation = correlationChart;

  const pairSelect = document.getElementById('pair-select');
  if (pairSelect) {
    if (!Array.from(pairSelect.options).some((option) => option.value === state.pair)) {
      state.pair = pairSelect.options[0]?.value;
    }
    if (pairSelect.value !== state.pair && state.pair) {
      pairSelect.value = state.pair;
    }
  }

  const pair = state.pair || pairSelect?.options[0]?.value;
  state.pair = pair;
  if (!pair) {
    priceChart.clear();
    correlationChart.clear();
    return;
  }

  const [assetA, assetB] = pair.split('|');
  const pairSeries = computePairSeries(assetA, assetB);
  if (!pairSeries) {
    priceChart.clear();
    correlationChart.clear();
    return;
  }

  const indices = computeVisibleIndices(pairSeries.dates);
  if (indices.length === 0) {
    priceChart.clear();
    correlationChart.clear();
    return;
  }

  const dates = indices.map((index) => pairSeries.dates[index]);
  const priceA = indices.map((index) => pairSeries.priceA[index]);
  const priceB = indices.map((index) => pairSeries.priceB[index]);
  const correlationValues = indices.map((index) => pairSeries.correlation[index]);

  const assetALabel = `${assetA} 가격`;
  const assetBLabel = `${assetB} 가격`;
  const correlationLabel = '롤링 상관계수';

  renderPriceChart(priceChart, dates, priceA, priceB, assetALabel, assetBLabel);
  renderCorrelationChart(correlationChart, dates, correlationValues, priceA, correlationLabel, assetALabel);
}

function computePairSeries(assetA, assetB) {
  if (!assetA || !assetB) {
    return null;
  }

  const dates = Array.isArray(state.analysisDates) ? state.analysisDates : [];
  const priceSeries = state.priceSeries || {};
  const pricesA = Array.isArray(priceSeries?.[assetA]) ? priceSeries[assetA] : null;
  const pricesB = Array.isArray(priceSeries?.[assetB]) ? priceSeries[assetB] : null;

  if (
    !pricesA ||
    !pricesB ||
    pricesA.length !== dates.length ||
    pricesB.length !== dates.length ||
    dates.length === 0
  ) {
    return null;
  }

  const windowSize = Math.max(Number(state.window) || DEFAULT_WINDOW, 2);
  const returnsA = toReturns(pricesA);
  const returnsB = toReturns(pricesB);
  const correlation = dates.map((_, index) => {
    if (index < windowSize - 1) {
      return null;
    }
    const start = Math.max(index - windowSize + 1, 0);
    const sliceA = returnsA.slice(start, index + 1);
    const sliceB = returnsB.slice(start, index + 1);
    const value = corr(sliceA, sliceB);
    return Number.isFinite(value) ? value : null;
  });

  return {
    dates: dates.slice(),
    priceA: pricesA.slice(),
    priceB: pricesB.slice(),
    correlation,
  };
}

function computeVisibleIndices(dates) {
  if (!Array.isArray(dates)) {
    return [];
  }

  const customRange = state.customRange || { start: null, end: null, valid: true };
  const startTime = parseDateSafe(customRange.start);
  const endTime = parseDateSafe(customRange.end);
  const hasCustomRange = customRange.valid && (startTime !== null || endTime !== null);

  if (hasCustomRange) {
    const indices = [];
    dates.forEach((date, index) => {
      const time = parseDateSafe(date);
      if (time === null) return;
      if (startTime !== null && time < startTime) return;
      if (endTime !== null && time > endTime) return;
      indices.push(index);
    });
    return indices;
  }

  const rangeDays = state.range;
  const indices = [];
  const startIndex = Math.max(dates.length - rangeDays, 0);
  for (let idx = startIndex; idx < dates.length; idx += 1) {
    indices.push(idx);
  }
  return indices;
}

function renderPriceChart(chart, dates, priceA, priceB, assetALabel, assetBLabel) {
  if (!chart) {
    return;
  }

  chart.setOption({
    tooltip: {
      trigger: 'axis',
    },
    legend: {
      data: [assetALabel, assetBLabel],
    },
    xAxis: [
      {
        type: 'category',
        data: dates,
        axisPointer: { type: 'shadow' },
        axisLabel: { color: TEXT_AXIS },
      },
    ],
    yAxis: [
      {
        type: 'value',
        name: assetALabel,
        position: 'left',
        scale: true,
        axisLabel: { color: TEXT_AXIS },
      },
      {
        type: 'value',
        name: assetBLabel,
        position: 'right',
        scale: true,
        axisLabel: { color: TEXT_AXIS },
      },
    ],
    series: [
      {
        name: assetALabel,
        type: 'line',
        data: priceA,
        smooth: true,
        yAxisIndex: 0,
      },
      {
        name: assetBLabel,
        type: 'line',
        data: priceB,
        smooth: true,
        yAxisIndex: 1,
      },
    ],
  });
}

function renderCorrelationChart(chart, dates, correlationValues, priceA, correlationLabel, assetALabel) {
  if (!chart) {
    return;
  }

  chart.setOption({
    tooltip: {
      trigger: 'axis',
    },
    legend: {
      data: [correlationLabel, assetALabel],
    },
    xAxis: [
      {
        type: 'category',
        data: dates,
        axisPointer: { type: 'shadow' },
        axisLabel: { color: TEXT_AXIS },
      },
    ],
    yAxis: [
      {
        type: 'value',
        name: '상관계수',
        min: -1,
        max: 1,
        position: 'left',
        axisLabel: { color: TEXT_AXIS },
      },
      {
        type: 'value',
        name: assetALabel,
        position: 'right',
        scale: true,
        axisLabel: { color: TEXT_AXIS },
      },
    ],
    series: [
      {
        name: correlationLabel,
        type: 'line',
        data: correlationValues,
        smooth: true,
        yAxisIndex: 0,
      },
      {
        name: assetALabel,
        type: 'line',
        data: priceA,
        smooth: true,
        yAxisIndex: 1,
      },
    ],
  });
}

function computeFmpCutoffDate() {
  return FMP_DATA_START;
}

async function fetchFmpSeriesBrowser(asset, apiKey, startDate, endDate) {
  const symbol = asset.fmpSymbol || asset.symbol.replace(/[^A-Za-z0-9]/g, '');
  const url = new URL(`https://financialmodelingprep.com/api/v3/historical-price-full/${encodeURIComponent(symbol)}`);
  if (apiKey) {
    url.searchParams.set('apikey', apiKey);
  }
  if (startDate) {
    url.searchParams.set('from', startDate);
  }
  if (endDate) {
    url.searchParams.set('to', endDate);
  }

  const json = await fetchFmpJson(url, asset.symbol);
  const rows = Array.isArray(json?.historical) ? json.historical : [];
  if (!rows || rows.length === 0) {
    throw buildFmpError(json, asset.symbol);
  }

  const filtered = rows
    .filter((row) => typeof row?.date === 'string')
    .filter((row) => (!startDate || row.date >= startDate) && (!endDate || row.date <= endDate))
    .map((row) => {
      const rawClose = Number(row.close ?? row.price);
      const adjCloseCandidate = Number(row.adjClose ?? row.adj_close);
      const close = Number.isFinite(adjCloseCandidate) ? adjCloseCandidate : rawClose;
      if (!Number.isFinite(close)) {
        return null;
      }
      const rawOpenValue = Number(row.open ?? row.adj_open);
      let open = Number.isFinite(rawOpenValue) ? rawOpenValue : null;
      if (Number.isFinite(rawOpenValue) && Number.isFinite(adjCloseCandidate) && Number.isFinite(rawClose) && rawClose !== 0) {
        open = rawOpenValue * (adjCloseCandidate / rawClose);
      }
      if (!Number.isFinite(open)) {
        open = close;
      }
      return {
        date: row.date,
        close,
        open,
      };
    })
    .filter((row) => row && Number.isFinite(row.close))
    .sort((a, b) => a.date.localeCompare(b.date));

  if (filtered.length < 2) {
    throw new Error(`${asset.symbol}: FMP에서 충분한 표본을 받지 못했습니다.`);
  }

  return {
    symbol: asset.symbol,
    category: asset.category,
    dates: filtered.map((row) => row.date),
    prices: filtered.map((row) => row.close),
    opens: filtered.map((row) => (row.open != null ? row.open : row.close)),
  };
}

async function fetchFmpJson(url, symbol) {
  const response = await fetch(url.toString(), { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`${symbol || 'FMP'}: 요청 실패 (${response.status})`);
  }
  const json = await response.json();
  if (json?.Note || json?.Information || json?.['Error Message']) {
    throw buildFmpError(json, symbol || url.pathname.split('/').pop() || '요청');
  }
  return json;
}

function buildFmpError(payload, symbol) {
  if (payload && typeof payload === 'object') {
    const message = payload['Error Message'] || payload.Note || payload.Information;
    if (message) {
      return new Error(`${symbol}: FMP 오류 - ${message}`);
    }
  }
  return new Error(`${symbol}: FMP 응답을 해석할 수 없습니다.`);
}

function isLatestDateStale(dateStr) {
  if (!dateStr) return true;
  const latestTime = parseDateSafe(dateStr);
  if (!Number.isFinite(latestTime)) return true;
  const today = new Date();
  const todayMidnight = Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate());
  return todayMidnight - latestTime > MS_PER_DAY;
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function fetchYahooSeries(asset) {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${asset.symbol}?interval=1d&range=5y&includeAdjustedClose=true`;
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${asset.symbol}`);
  }
  const json = await response.json();
  const result = json?.chart?.result?.[0];
  if (!result) {
    throw new Error(`Invalid response for ${asset.symbol}`);
  }

  const timestamps = result.timestamp || [];
  const closes = (result.indicators?.adjclose?.[0]?.adjclose || result.indicators?.quote?.[0]?.close || []).map(Number);
  const prices = [];
  const dates = [];
  timestamps.forEach((ts, index) => {
    const price = closes[index];
    if (Number.isFinite(price)) {
      const date = new Date(ts * 1000).toISOString().split('T')[0];
      dates.push(date);
      prices.push(price);
    }
  });

  return {
    symbol: asset.symbol,
    category: asset.category,
    dates,
    prices,
  };
}

function computeAllMetrics(returns, aligned) {
  WINDOWS.forEach((window) => {
    const metrics = computeWindowMetrics(window, returns, aligned);
    state.metrics[window] = metrics;
  });
}

function computeWindowMetrics(window, returns, aligned) {
  const allSymbols = ASSETS.map((asset) => asset.symbol);
  const signalSymbols = SIGNAL.symbols;
  const categories = aligned.categories;
  const dates = returns.dates;
  const records = [];
  const stabilityValues = [];

  for (let endIndex = window - 1; endIndex < returns.dates.length; endIndex += 1) {
    const startIndex = endIndex - window + 1;
    const fullMatrix = buildCorrelationMatrix(allSymbols, returns.returns, startIndex, endIndex);
    const signalMatrix = buildCorrelationMatrix(signalSymbols, returns.returns, startIndex, endIndex);
    const stability = computeStability(signalMatrix, signalSymbols, categories);
    const sub = computeSubIndices(signalMatrix, signalSymbols, categories);
    const smoothed = stability; // placeholder for EMA, updated later

    records.push({
      date: dates[endIndex],
      stability,
      sub,
      matrix: signalMatrix,
      fullMatrix,
      smoothed,
      delta: 0,
    });
    stabilityValues.push(stability);
  }

  if (records.length === 0) {
    throw new Error(`윈도우 ${window}일에 필요한 표본이 부족합니다.`);
  }

  const smoothedSeries = ema(stabilityValues, 10);
  const shortEma = ema(stabilityValues, 3);
  const longEma = ema(stabilityValues, 10);

  records.forEach((record, index) => {
    record.smoothed = smoothedSeries[index];
    record.delta = shortEma[index] - longEma[index];
  });

  const latest = records[records.length - 1];
  const average180 = mean(stabilityValues.slice(-180));
  const pairs = buildPairSeries(records, window, allSymbols);

  return {
    records,
    average180,
    pairs,
    latest,
  };
}

function refreshPairSeriesFromPrices() {
  if (!state.priceSeries || Object.keys(state.priceSeries).length === 0) {
    return;
  }

  Object.entries(state.metrics || {}).forEach(([windowSize, metrics]) => {
    const numericWindow = Number(windowSize);
    if (!metrics || !Array.isArray(metrics.records)) {
      return;
    }
    metrics.pairs = buildPairSeries(metrics.records, numericWindow, ASSETS.map((asset) => asset.symbol));
  });
}

function selectMatrixForSymbols(record, symbols) {
  if (!record || !Array.isArray(symbols)) {
    return null;
  }
  if (Array.isArray(record.fullMatrix) && record.fullMatrix.length === symbols.length) {
    return record.fullMatrix;
  }
  if (Array.isArray(record.matrix) && record.matrix.length === symbols.length) {
    return record.matrix;
  }
  return null;
}

function buildPairSeries(records, window, symbols = ASSETS.map((asset) => asset.symbol)) {
  const pairs = {};
  const priceOffset = window - 1;

  for (let i = 0; i < symbols.length; i += 1) {
    for (let j = i + 1; j < symbols.length; j += 1) {
      const key = `${symbols[i]}|${symbols[j]}`;
      pairs[key] = {
        dates: [],
        correlation: [],
        priceA: [],
        priceB: [],
      };
    }
  }

  records.forEach((record, idx) => {
    const matrix = selectMatrixForSymbols(record, symbols);
    const priceIndex = priceOffset + idx;
    for (let i = 0; i < symbols.length; i += 1) {
      for (let j = i + 1; j < symbols.length; j += 1) {
        const key = `${symbols[i]}|${symbols[j]}`;
        const pair = pairs[key];
        pair.dates.push(record.date);
        const corrValue = matrix ? matrix[i][j] : null;
        pair.correlation.push(Number.isFinite(corrValue) ? corrValue : null);
        const seriesA = state.priceSeries?.[symbols[i]] || [];
        const seriesB = state.priceSeries?.[symbols[j]] || [];
        const valueA = seriesA[priceIndex];
        const valueB = seriesB[priceIndex];
        pair.priceA.push(Number.isFinite(valueA) ? valueA : null);
        pair.priceB.push(Number.isFinite(valueB) ? valueB : null);
      }
    }
  });

  return pairs;
}

function safeNumber(value, fallback = 0) {
  if (Number.isFinite(value)) {
    return value;
  }
  if (Number.isFinite(fallback)) {
    return fallback;
  }
  return 0;
}

function computeCustomRangeDays(startTime, endTime, recordLength = 0) {
  if (Number.isFinite(startTime) && Number.isFinite(endTime) && endTime >= startTime) {
    const diff = endTime - startTime;
    const days = Math.floor(diff / MS_PER_DAY) + 1;
    if (days > 0) {
      return days;
    }
  }

  if (Number.isFinite(startTime) || Number.isFinite(endTime)) {
    return recordLength > 0 ? recordLength : null;
  }

  return recordLength > 0 ? recordLength : null;
}

function parseDateSafe(value) {
  if (!value) return null;
  const time = Date.parse(`${value}T00:00:00Z`);
  return Number.isFinite(time) ? time : null;
}

function setCustomRangeFeedback(message) {
  const feedback = document.getElementById('custom-range-feedback');
  if (!feedback) return;
  const text = message || '';
  if (feedback.textContent === text) return;
  feedback.textContent = text;
}

function setGaugePanelFeedback(message) {
  const panel = document.getElementById('gauge-panel');
  if (!panel) return;
  let helper = panel.querySelector('.panel-feedback');
  if (!message) {
    if (helper) {
      helper.remove();
    }
    return;
  }

  if (!helper) {
    helper = document.createElement('p');
    helper.className = 'panel-feedback';
    const legend = panel.querySelector('.legend');
    if (legend) {
      panel.insertBefore(helper, legend);
    } else {
      panel.appendChild(helper);
    }
  }

  helper.textContent = message;
}

function showEmptyState(
  msg = '실제 데이터가 없습니다.',
  sub = 'API 생성 실패 또는 제한. 다음 스케줄 실행 후 자동 갱신됩니다.',
) {
  const existing = document.querySelector('.empty-state-banner');
  if (existing) {
    existing.querySelector('.empty-state-primary').textContent = msg;
    existing.querySelector('.empty-state-secondary').textContent = sub;
    return;
  }

  const div = document.createElement('div');
  div.className = 'empty-state-banner';
  div.style.cssText = 'padding:12px;border:1px solid #ddd;border-radius:8px;margin:12px;background:#fffbe6;';
  div.innerHTML = `<strong class="empty-state-primary">${msg}</strong><div class="empty-state-secondary" style="margin-top:6px">${sub}</div>`;
  document.body.prepend(div);
}

function showError(message) {
  const box = document.getElementById('error-box');
  if (!box) return;
  box.textContent = message;
  box.classList.remove('hidden');
  box.classList.add('error');
  box.classList.remove('notice');
}

function showNotice(message) {
  const box = document.getElementById('error-box');
  if (!box) return;
  box.textContent = message;
  box.classList.remove('hidden');
  box.classList.add('notice');
  box.classList.remove('error');
}

function hideError() {
  const box = document.getElementById('error-box');
  if (!box) return;
  box.textContent = '';
  box.classList.add('hidden');
  box.classList.remove('error');
  box.classList.remove('notice');
}

function maybeAlignDatesToCurrent() {
  const metricsValues = Object.values(state.metrics || {});
  if (metricsValues.length === 0) return;

  const latestTimestamps = metricsValues
    .map((metrics) => metrics?.records?.[metrics.records.length - 1]?.date)
    .filter(Boolean)
    .map((dateStr) => Date.parse(`${dateStr}T00:00:00Z`))
    .filter(Number.isFinite);

  if (latestTimestamps.length === 0) {
    return;
  }

  const latestTime = Math.max(...latestTimestamps);
  const targetDate = state.generatedAt instanceof Date ? state.generatedAt : new Date();
  const targetMidnight = new Date(Date.UTC(targetDate.getUTCFullYear(), targetDate.getUTCMonth(), targetDate.getUTCDate()));
  const diffDays = Math.round((targetMidnight.getTime() - latestTime) / MS_PER_DAY);

  if (diffDays <= MAX_STALE_DAYS) {
    state.generatedAt = targetDate;
    return;
  }

  const shiftDate = (dateStr) => {
    const date = new Date(`${dateStr}T00:00:00Z`);
    if (!Number.isFinite(date.getTime())) return dateStr;
    date.setUTCDate(date.getUTCDate() + diffDays);
    return date.toISOString().split('T')[0];
  };

  state.analysisDates = (state.analysisDates || []).map(shiftDate);

  metricsValues.forEach((metrics) => {
    metrics.records = metrics.records.map((record) => ({
      ...record,
      date: shiftDate(record.date),
    }));
    Object.values(metrics.pairs || {}).forEach((pair) => {
      pair.dates = pair.dates.map(shiftDate);
    });
    metrics.latest = metrics.records[metrics.records.length - 1];
  });

  state.metrics = Object.fromEntries(
    Object.entries(state.metrics).map(([windowSize, metrics]) => [windowSize, metrics])
  );

  state.generatedAt = targetDate;
}
