const DEFAULT_WINDOW = 30;
const WINDOWS = [20, 30, 60];
const RANGE_OPTIONS = [30, 60, 90, 180];
const BANDS = { red: [0, 0.3], yellow: [0.3, 0.4], green: [0.4, 1.0] };

const ASSETS = [
  { symbol: 'QQQ', label: 'QQQ (NASDAQ 100 ETF)', category: 'stock' },
  { symbol: 'SPY', label: 'SPY (S&P 500 ETF)', category: 'stock' },
  { symbol: 'TLT', label: 'TLT (미국 장기채)', category: 'bond' },
  { symbol: 'GLD', label: 'GLD (금 ETF)', category: 'gold' },
  { symbol: 'BTC-USD', label: 'BTC-USD (비트코인)', category: 'crypto' },
];

const state = {
  window: DEFAULT_WINDOW,
  range: 180,
  metrics: {},
  normalizedPrices: {},
  analysisDates: [],
  generatedAt: null,
};

const charts = {};

const {
  alignSeries,
  computeReturns: computeReturnsForState,
  buildCorrelationMatrix,
  computeStability,
  computeSubIndices,
  ema,
  mean,
} = window.MarketMetrics;

document.addEventListener('DOMContentLoaded', init);

async function init() {
  showError('데이터를 불러오는 중입니다...');
  state.metrics = {};
  try {
    const precomputed = await fetchPrecomputed();
    if (precomputed.status === 'ok') {
      hydrateFromPrecomputed(precomputed.payload);
      hideError();
    } else if (precomputed.status === 'missing') {
      const fallback = await fetchFallbackPrecomputed();
      if (fallback.status === 'ok') {
        hydrateFromPrecomputed(fallback.payload);
        showNotice('사전 계산된 데이터를 찾지 못해 샘플 데이터를 표시합니다. 최신 지표를 보려면 GitHub Actions 배치를 확인해 주세요.');
      } else if (isLocalhost()) {
        await loadFromYahoo();
        hideError();
      } else {
        throw new Error('PRECOMPUTED_DATA_UNAVAILABLE');
      }
    } else if (precomputed.status === 'error' && isLocalhost()) {
      await loadFromYahoo();
      hideError();
    } else {
      throw precomputed.error || new Error('PRECOMPUTED_DATA_UNAVAILABLE');
    }

    populateControls();
    renderAll();
  } catch (error) {
    console.error(error);
    if (error && error.message === 'PRECOMPUTED_DATA_UNAVAILABLE') {
      showError('사전 계산된 데이터가 없습니다. README에 안내된 GitHub Actions 배치를 실행하거나 `npm run generate:data` 후 커밋해 주세요.');
    } else {
      showError('데이터를 불러오지 못했습니다. 네트워크 상태를 확인하거나 데이터를 다시 생성해 주세요.');
    }
  }
}

async function loadFromYahoo() {
  const assetSeries = await Promise.all(ASSETS.map(fetchYahooSeries));
  const aligned = alignSeries(assetSeries);
  const returns = computeReturnsForState(aligned);
  state.analysisDates = returns.dates;
  state.generatedAt = new Date();
  state.normalizedPrices = returns.normalizedPrices;
  computeAllMetrics(returns, aligned);
}

function hydrateFromPrecomputed(data) {
  state.generatedAt = data.generatedAt ? new Date(data.generatedAt) : new Date();
  state.analysisDates = data.analysisDates || [];
  state.normalizedPrices = data.normalizedPrices || {};
  state.metrics = {};
  Object.entries(data.windows).forEach(([windowSize, metrics]) => {
    const numericWindow = Number(windowSize);
    state.metrics[numericWindow] = metrics;
  });
}

async function fetchPrecomputed() {
  try {
    const response = await fetch('./data/precomputed.json', { cache: 'no-store' });
    if (response.status === 404) {
      return { status: 'missing' };
    }
    if (!response.ok) {
      return { status: 'error', error: new Error(`precomputed fetch failed: ${response.status}`) };
    }
    const payload = await response.json();
    return { status: 'ok', payload };
  } catch (error) {
    return { status: 'error', error };
  }
}

async function fetchFallbackPrecomputed() {
  try {
    const response = await fetch('./data/precomputed-sample.json', { cache: 'no-store' });
    if (!response.ok) {
      return { status: 'error', error: new Error(`fallback fetch failed: ${response.status}`) };
    }
    const payload = await response.json();
    return { status: 'ok', payload };
  } catch (error) {
    return { status: 'error', error };
  }
}

function isLocalhost() {
  return ['localhost', '127.0.0.1', '::1'].includes(window.location.hostname);
}

function populateControls() {
  const windowSelect = document.getElementById('window-select');
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
  pairSelect.innerHTML = '';
  for (let i = 0; i < ASSETS.length; i += 1) {
    for (let j = i + 1; j < ASSETS.length; j += 1) {
      const pair = `${ASSETS[i].symbol}|${ASSETS[j].symbol}`;
      const option = document.createElement('option');
      option.value = pair;
      option.textContent = `${ASSETS[i].symbol} / ${ASSETS[j].symbol}`;
      pairSelect.appendChild(option);
    }
  }

  document.getElementById('window-select').addEventListener('change', (event) => {
    state.window = Number(event.target.value);
    renderAll();
  });

  document.getElementById('range-select').addEventListener('change', (event) => {
    state.range = Number(event.target.value);
    renderHistory();
    renderPair();
  });

  document.getElementById('pair-select').addEventListener('change', () => {
    renderPair();
  });

  document.querySelectorAll('button.info').forEach((button) => {
    button.addEventListener('click', () => {
      const target = document.getElementById(button.dataset.target);
      target.classList.toggle('visible');
    });
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

function renderGauge() {
  const metrics = state.metrics[state.window];
  const latest = metrics.records[metrics.records.length - 1];
  const element = document.getElementById('stability-gauge');
  const chart = charts.stability || echarts.init(element);
  charts.stability = chart;

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
        formatter: () => `지수: ${latest.stability.toFixed(3)}\n180일 평균: ${metrics.average180.toFixed(3)}\n추세: ${latest.delta >= 0 ? '▲' : '▼'} ${(Math.abs(latest.delta)).toFixed(3)}`,
        fontSize: 16,
        offsetCenter: [0, '65%'],
      },
      data: [{ value: latest.stability }],
    }],
  });
}

function renderSubGauges() {
  const metrics = state.metrics[state.window];
  const latest = metrics.records[metrics.records.length - 1];
  const mapping = [
    { key: 'stockCrypto', element: 'stock-crypto-gauge' },
    { key: 'traditional', element: 'traditional-gauge' },
    { key: 'safeNegative', element: 'safe-neg-gauge' },
  ];

  mapping.forEach(({ key, element }) => {
    const chart = charts[element] || echarts.init(document.getElementById(element));
    charts[element] = chart;
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
  const rangeDays = state.range;
  const series = metrics.records.slice(-rangeDays);

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
  const chart = charts.heatmap || echarts.init(element);
  charts.heatmap = chart;

  const metrics = state.metrics[state.window];
  const latest = metrics.records[metrics.records.length - 1];
  const labels = ASSETS.map((asset) => asset.symbol);
  const data = [];
  for (let i = 0; i < labels.length; i += 1) {
    for (let j = 0; j < labels.length; j += 1) {
      const value = i === j ? 1 : latest.matrix[i][j];
      data.push([i, j, Number(value.toFixed(3))]);
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

function renderPair() {
  const element = document.getElementById('pair-chart');
  const chart = charts.pair || echarts.init(element);
  charts.pair = chart;

  const pairSelect = document.getElementById('pair-select');
  const pair = pairSelect.value || pairSelect.options[0]?.value;
  if (!pair) return;

  const [assetA, assetB] = pair.split('|');
  const metrics = state.metrics[state.window];
  const pairSeries = metrics.pairs[pair];
  const rangeDays = state.range;
  const sliced = {
    dates: pairSeries.dates.slice(-rangeDays),
    correlation: pairSeries.correlation.slice(-rangeDays),
    priceA: pairSeries.priceA.slice(-rangeDays),
    priceB: pairSeries.priceB.slice(-rangeDays),
  };

  chart.setOption({
    tooltip: {
      trigger: 'axis',
    },
    legend: {
      data: [`${assetA} 가격지수`, `${assetB} 가격지수`, '롤링 상관계수'],
    },
    xAxis: [{
      type: 'category',
      data: sliced.dates,
      axisPointer: { type: 'shadow' },
    }],
    yAxis: [
      {
        type: 'value',
        name: '가격지수',
        position: 'left',
      },
      {
        type: 'value',
        name: '상관계수',
        min: -1,
        max: 1,
        position: 'right',
      },
    ],
    series: [
      {
        name: `${assetA} 가격지수`,
        type: 'line',
        data: sliced.priceA,
        smooth: true,
      },
      {
        name: `${assetB} 가격지수`,
        type: 'line',
        data: sliced.priceB,
        smooth: true,
      },
      {
        name: '롤링 상관계수',
        type: 'line',
        yAxisIndex: 1,
        data: sliced.correlation,
        smooth: true,
      },
    ],
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
        pair.priceA.push(state.normalizedPrices[symbols[i]][priceIndex]);
        pair.priceB.push(state.normalizedPrices[symbols[j]][priceIndex]);
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
