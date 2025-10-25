const DEFAULT_WINDOW = 30;
const WINDOWS = [20, 30, 60];
const RANGE_OPTIONS = [30, 60, 90, 180];
const BANDS = { red: [0, 0.3], yellow: [0.3, 0.4], green: [0.4, 1.0] };
const DEFAULT_PAIR = 'QQQ|BTC-USD';
const MAX_STALE_DAYS = 7;
const MS_PER_DAY = 24 * 60 * 60 * 1000;
const ALPHA_RANGE_YEARS = 5;
const MINIMUM_ALPHA_CUTOFF = '2020-01-01';
const ALPHA_RATE_DELAY = Math.round((60 * 1000) / 5) + 1500;
const ACTIOND_WAIT_MS = 5000;
const ACTIOND_POLL_INTERVAL_MS = 75;
const IS_LOCAL = ['localhost', '127.0.0.1', '::1'].includes(window.location.hostname);
const DEFAULT_DATA_PATH = './data/precomputed.json';

let datasetPath = DEFAULT_DATA_PATH;
let cacheTagRaw = null;
let versionManifest = null;

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

async function hydrateAlphaKeyFromEnvironment() {
  const direct = resolveActiondEnvironment('ALPHAVANTAGE_API_KEY');
  if (direct) {
    state.alphaKey = direct;
    return;
  }

  const fetched = await fetchActiondSecret('ALPHAVANTAGE_API_KEY');
  if (fetched) {
    state.alphaKey = fetched;
  }
}

const ASSETS = [
  { symbol: 'QQQ', label: 'QQQ (NASDAQ 100 ETF)', category: 'stock', source: 'TIME_SERIES_DAILY_ADJUSTED' },
  { symbol: 'SPY', label: 'SPY (S&P 500 ETF)', category: 'stock', source: 'TIME_SERIES_DAILY_ADJUSTED' },
  { symbol: 'TLT', label: 'TLT (미국 장기채)', category: 'bond', source: 'TIME_SERIES_DAILY_ADJUSTED' },
  { symbol: 'GLD', label: 'GLD (금 ETF)', category: 'gold', source: 'TIME_SERIES_DAILY_ADJUSTED' },
  { symbol: 'BTC-USD', label: 'BTC-USD (비트코인)', category: 'crypto', source: 'DIGITAL_CURRENCY_DAILY' },
];

const state = {
  window: DEFAULT_WINDOW,
  range: 180,
  metrics: {},
  normalizedPrices: {},
  priceSeries: {},
  priceSeriesSource: 'actual',
  analysisDates: [],
  generatedAt: null,
  pair: DEFAULT_PAIR,
  alphaKey: '',
  refreshing: false,
  autoRefreshAttempted: false,
  customRange: {
    start: null,
    end: null,
    valid: true,
  },
};

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

const {
  alignSeries,
  computeReturns: computeReturnsForState,
  buildCorrelationMatrix,
  computeStability,
  computeSubIndices,
  ema,
  mean,
  toReturns,
  corr,
} = window.MarketMetrics;

document.addEventListener('DOMContentLoaded', init);

async function init() {
  showError('데이터를 불러오는 중입니다...');
  state.metrics = {};
  await hydrateAlphaKeyFromEnvironment();
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
  state.priceSeriesSource = 'actual';
  computeAllMetrics(returns, aligned);
  maybeAlignDatesToCurrent();
}

async function loadFromAlphaVantage(apiKey) {
  const effectiveCutoff = computeAlphaCutoffDate(ALPHA_RANGE_YEARS);
  const assetSeries = [];

  for (let index = 0; index < ASSETS.length; index += 1) {
    const asset = ASSETS[index];
    const series = await fetchAlphaSeriesBrowser(asset, apiKey, effectiveCutoff);
    assetSeries.push(series);
    if (index < ASSETS.length - 1) {
      await delay(ALPHA_RATE_DELAY);
    }
  }

  const aligned = alignSeries(assetSeries);
  const returns = computeReturnsForState(aligned);
  state.analysisDates = returns.dates;
  state.generatedAt = new Date();
  state.normalizedPrices = returns.normalizedPrices;
  state.priceSeries = returns.priceSeries;
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

  const rawPriceSeries = data.priceSeries && typeof data.priceSeries === 'object' ? data.priceSeries : null;
  if (!rawPriceSeries || Object.keys(rawPriceSeries).length === 0) {
    showEmptyState('priceSeries 누락');
    return;
  }

  const priceSeries = cloneSeriesMap(rawPriceSeries);
  const normalizedSource =
    data.normalizedPrices && typeof data.normalizedPrices === 'object'
      ? data.normalizedPrices
      : buildNormalizedPricesFromSeries(priceSeries);
  const normalizedPrices = cloneSeriesMap(normalizedSource);

  state.priceSeries = priceSeries;
  state.normalizedPrices = normalizedPrices;
  state.priceSeriesSource = 'actual';
  state.metrics = {};
  if (data.windows && typeof data.windows === 'object') {
    Object.entries(data.windows).forEach(([windowSize, metrics]) => {
      const numericWindow = Number(windowSize);
      state.metrics[numericWindow] = metrics;
    });
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

  const dataFile =
    manifest && typeof manifest === 'object' && typeof manifest.data === 'string' && manifest.data.trim()
      ? manifest.data.trim()
      : 'precomputed.json';
  const cacheSeed =
    manifest?.build || manifest?.buildId || manifest?.generatedAt || manifest?.timestamp || `${now}`;
  const targetUrl = `./data/${dataFile}?ts=${encodeURIComponent(cacheSeed)}`;
  const response = await fetch(targetUrl, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
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
    });
    windowSelect.dataset.bound = 'true';
  }

  if (!rangeSelect.dataset.bound) {
    rangeSelect.addEventListener('change', (event) => {
      state.range = Number(event.target.value);
    });
    rangeSelect.dataset.bound = 'true';
  }

  if (!pairSelect.dataset.bound) {
    pairSelect.addEventListener('change', (event) => {
      state.pair = event.target.value;
    });
    pairSelect.dataset.bound = 'true';
  }

  const refreshButton = document.getElementById('refresh-button');
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
        target.classList.toggle('visible');
      });
      button.dataset.bound = 'true';
    }
  });
}

function renderAll() {
  updateMeta();
  renderGauge();
  renderSubGauges();
  renderHistory();
  renderHeatmap();
  renderPair();
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

async function maybeRefreshData() {
  if (!isLocalhost()) {
    return false;
  }

  if (!state.alphaKey) {
    await hydrateAlphaKeyFromEnvironment();
  }

  const { shouldFetch } = evaluateRefreshNeeds();

  if (!shouldFetch) {
    return false;
  }

  if (!state.alphaKey) {
    showNotice('Alpha Vantage API 키를 찾을 수 없어 자동 갱신을 건너뜁니다. GitHub Actions/Pages Secrets 또는 환경 변수 ALPHAVANTAGE_API_KEY를 설정해 주세요.');
    return false;
  }

  showNotice('Alpha Vantage에서 최신 데이터를 불러오는 중입니다...');
  await loadFromAlphaVantage(state.alphaKey);
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

  const requiredCutoff = computeAlphaCutoffDate(ALPHA_RANGE_YEARS);
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
        formatter: () => `지수: ${safeNumber(latest.stability).toFixed(3)}\n${averageLabel}: ${safeNumber(average180).toFixed(3)}\n추세(${rangeDescriptor}): ${delta >= 0 ? '▲' : '▼'} ${(Math.abs(delta)).toFixed(3)}`,
        fontSize: 16,
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
          offsetCenter: [0, '60%'],
        },
        data: [{ value: gaugeValue }],
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

  if (typeof feedbackMessage === 'string') {
    setCustomRangeFeedback(feedbackMessage);
  }

  if (empty) {
    chart.clear();
    return;
  }

  chart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['Stability', 'Smoothed'] },
    xAxis: {
      type: 'category',
      data: series.map((item) => item.date),
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
    },
    series: [
      {
        name: 'Stability',
        type: 'line',
        data: series.map((item) => item.stability),
        smooth: true,
      },
      {
        name: 'Smoothed',
        type: 'line',
        data: series.map((item) => item.smoothed),
        smooth: true,
      },
    ],
  });
}

function renderHeatmap() {
  const element = document.getElementById('heatmap-chart');
  if (!element) {
    return;
  }
  const chart = charts.heatmap || echarts.init(element);
  charts.heatmap = chart;

  const labels = ASSETS.map((asset) => asset.symbol);
  const matrix = computeHeatmapMatrix();
  if (!matrix) {
    chart.clear();
    return;
  }

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
      formatter: (params) => `${labels[params.data[1]]} / ${labels[params.data[0]]}: ${params.data[2]}`,
    },
    grid: {
      height: '80%',
      top: '10%',
    },
    xAxis: { type: 'category', data: labels },
    yAxis: { type: 'category', data: labels },
    visualMap: {
      min: -1,
      max: 1,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
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

function computeHeatmapMatrix() {
  const symbols = ASSETS.map((asset) => asset.symbol);
  const dates = Array.isArray(state.analysisDates) ? state.analysisDates : [];
  const priceSeries = state.priceSeries || {};
  if (dates.length === 0) {
    return null;
  }

  const windowSize = Math.max(Number(state.window) || DEFAULT_WINDOW, 2);
  if (dates.length < windowSize) {
    return null;
  }

  const returnsMap = {};
  for (let index = 0; index < symbols.length; index += 1) {
    const symbol = symbols[index];
    const prices = priceSeries[symbol];
    if (!Array.isArray(prices) || prices.length !== dates.length) {
      return null;
    }
    returnsMap[symbol] = toReturns(prices);
  }

  const endIndex = dates.length - 1;
  const startIndex = Math.max(endIndex - windowSize + 1, 0);
  const matrix = [];

  for (let i = 0; i < symbols.length; i += 1) {
    const row = [];
    for (let j = 0; j < symbols.length; j += 1) {
      if (i === j) {
        row.push(1);
      } else {
        const sliceA = returnsMap[symbols[i]].slice(startIndex, endIndex + 1);
        const sliceB = returnsMap[symbols[j]].slice(startIndex, endIndex + 1);
        const value = corr(sliceA, sliceB);
        row.push(Number.isFinite(value) ? value : null);
      }
    }
    matrix.push(row);
  }

  return matrix;
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
      },
    ],
    yAxis: [
      {
        type: 'value',
        name: assetALabel,
        position: 'left',
        scale: true,
      },
      {
        type: 'value',
        name: assetBLabel,
        position: 'right',
        scale: true,
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
      },
    ],
    yAxis: [
      {
        type: 'value',
        name: '상관계수',
        min: -1,
        max: 1,
        position: 'left',
      },
      {
        type: 'value',
        name: assetALabel,
        position: 'right',
        scale: true,
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

function computeAlphaCutoffDate() {
  return MINIMUM_ALPHA_CUTOFF;
}

async function fetchAlphaSeriesBrowser(asset, apiKey, cutoffDate) {
  if (asset.source === 'DIGITAL_CURRENCY_DAILY') {
    return fetchAlphaDigital(asset, apiKey, cutoffDate);
  }
  return fetchAlphaEquity(asset, apiKey, cutoffDate);
}

async function fetchAlphaEquity(asset, apiKey, cutoffDate) {
  const url = new URL('https://www.alphavantage.co/query');
  url.searchParams.set('function', 'TIME_SERIES_DAILY_ADJUSTED');
  url.searchParams.set('symbol', asset.symbol);
  url.searchParams.set('outputsize', 'full');
  url.searchParams.set('apikey', apiKey);

  const json = await fetchAlphaJson(url, asset.symbol);
  const series = json['Time Series (Daily)'];
  if (!series) {
    throw buildAlphaError(json, asset.symbol);
  }
  return normalizeAlphaSeries(asset, series, cutoffDate, (value) => Number(value['5. adjusted close'] || value['4. close']));
}

async function fetchAlphaDigital(asset, apiKey, cutoffDate) {
  const url = new URL('https://www.alphavantage.co/query');
  url.searchParams.set('function', 'DIGITAL_CURRENCY_DAILY');
  url.searchParams.set('symbol', asset.symbol.split('-')[0]);
  url.searchParams.set('market', 'USD');
  url.searchParams.set('apikey', apiKey);

  const json = await fetchAlphaJson(url, asset.symbol);
  const series = json['Time Series (Digital Currency Daily)'];
  if (!series) {
    throw buildAlphaError(json, asset.symbol);
  }
  return normalizeAlphaSeries(asset, series, cutoffDate, (value) => Number(value['4a. close (USD)'] || value['4b. close (USD)']));
}

async function fetchAlphaJson(url, symbol) {
  const response = await fetch(url.toString(), { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`${symbol || 'Alpha Vantage'}: 요청 실패 (${response.status})`);
  }
  const json = await response.json();
  if (json?.Note || json?.Information || json?.['Error Message']) {
    throw buildAlphaError(json, symbol || url.searchParams.get('symbol') || '요청');
  }
  return json;
}

function normalizeAlphaSeries(asset, rawSeries, cutoffDate, extractClose) {
  const dates = Object.keys(rawSeries)
    .filter((date) => date >= cutoffDate)
    .sort();

  if (dates.length === 0) {
    throw new Error(`${asset.symbol}: ${cutoffDate} 이후 데이터가 없습니다.`);
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
    throw new Error(`${asset.symbol}: Alpha Vantage에서 충분한 표본을 받지 못했습니다.`);
  }

  return {
    symbol: asset.symbol,
    category: asset.category,
    dates: filteredDates,
    prices,
  };
}

function buildAlphaError(payload, symbol) {
  if (payload && typeof payload === 'object') {
    const message = payload['Error Message'] || payload.Note || payload.Information;
    if (message) {
      return new Error(`${symbol}: Alpha Vantage 오류 - ${message}`);
    }
  }
  return new Error(`${symbol}: Alpha Vantage 응답을 해석할 수 없습니다.`);
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
  const symbols = ASSETS.map((asset) => asset.symbol);
  const categories = aligned.categories;
  const dates = returns.dates;
  const records = [];
  const stabilityValues = [];

  for (let endIndex = window - 1; endIndex < returns.dates.length; endIndex += 1) {
    const startIndex = endIndex - window + 1;
    const matrix = buildCorrelationMatrix(symbols, returns.returns, startIndex, endIndex);
    const stability = computeStability(matrix, symbols, categories);
    const sub = computeSubIndices(matrix, symbols, categories);
    const smoothed = stability; // placeholder for EMA, updated later

    records.push({
      date: dates[endIndex],
      stability,
      sub,
      matrix,
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
  const pairs = buildPairSeries(records, window);

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
    metrics.pairs = buildPairSeries(metrics.records, numericWindow);
  });
}

function buildPairSeries(records, window) {
  const pairs = {};
  const symbols = ASSETS.map((asset) => asset.symbol);
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
    const priceIndex = priceOffset + idx;
    for (let i = 0; i < symbols.length; i += 1) {
      for (let j = i + 1; j < symbols.length; j += 1) {
        const key = `${symbols[i]}|${symbols[j]}`;
        const pair = pairs[key];
        pair.dates.push(record.date);
        pair.correlation.push(record.matrix[i][j]);
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
