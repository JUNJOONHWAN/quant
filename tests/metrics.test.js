const test = require('node:test');
const assert = require('node:assert/strict');

const metrics = require('../static_site/assets/metrics.js');

test('alignSeries returns the intersection of dates', () => {
  const seriesList = [
    {
      symbol: 'AAA',
      category: 'stock',
      dates: ['2024-01-01', '2024-01-02', '2024-01-03'],
      prices: [100, 102, 101],
    },
    {
      symbol: 'BBB',
      category: 'bond',
      dates: ['2024-01-02', '2024-01-03', '2024-01-04'],
      prices: [99, 98, 100],
    },
  ];

  const aligned = metrics.alignSeries(seriesList);
  assert.deepEqual(aligned.dates, ['2024-01-02', '2024-01-03']);
  assert.deepEqual(aligned.prices.AAA, [102, 101]);
  assert.deepEqual(aligned.prices.BBB, [99, 98]);
  assert.equal(aligned.categories.AAA, 'stock');
  assert.equal(aligned.categories.BBB, 'bond');
});

test('computeReturns emits log returns and normalized prices', () => {
  const aligned = {
    dates: ['2024-01-01', '2024-01-02', '2024-01-03'],
    prices: {
      AAA: [100, 110, 121],
      BBB: [200, 198, 210],
    },
  };

  const result = metrics.computeReturns(aligned);
  assert.deepEqual(result.dates, ['2024-01-02', '2024-01-03']);
  assert.equal(result.returns.AAA.length, 2);
  assert.equal(result.returns.BBB.length, 2);

  const epsilon = 1e-12;
  assert.ok(Math.abs(result.returns.AAA[0] - Math.log(110 / 100)) < epsilon);
  assert.ok(Math.abs(result.returns.AAA[1] - Math.log(121 / 110)) < epsilon);
  assert.ok(Math.abs(result.normalizedPrices.AAA[0] - 1) < epsilon);
  assert.ok(Math.abs(result.normalizedPrices.AAA[1] - (121 / 110)) < epsilon);
});

test('computeReturns throws if asset has fewer than two prices', () => {
  const aligned = {
    dates: ['2024-01-01'],
    prices: {
      AAA: [100],
    },
  };

  assert.throws(() => {
    metrics.computeReturns(aligned);
  }, /AAA/);
});

test('computeStability applies pair weights', () => {
  const symbols = ['AAA', 'BBB', 'CCC'];
  const categories = { AAA: 'stock', BBB: 'stock', CCC: 'bond' };
  const matrix = [
    [1, 0.8, -0.4],
    [0.8, 1, -0.5],
    [-0.4, -0.5, 1],
  ];

  const stability = metrics.computeStability(matrix, symbols, categories);
  assert.ok(Math.abs(stability - 0.52) < 1e-9);
});

test('computeSubIndices returns expected averages', () => {
  const symbols = ['AAA', 'BBB', 'CCC', 'DDD'];
  const categories = { AAA: 'stock', BBB: 'stock', CCC: 'bond', DDD: 'crypto' };
  const matrix = [
    [1, 0.6, -0.3, 0.2],
    [0.6, 1, -0.4, 0.25],
    [-0.3, -0.4, 1, 0.1],
    [0.2, 0.25, 0.1, 1],
  ];

  const sub = metrics.computeSubIndices(matrix, symbols, categories);
  assert.ok(Math.abs(sub.stockCrypto - 0.225) < 1e-9);
  assert.ok(Math.abs(sub.traditional - (0.6 + 0.3 + 0.4) / 3) < 1e-9);
  assert.ok(Math.abs(sub.safeNegative - 0.35) < 1e-9);
});

test('ema replicates exponential smoothing', () => {
  const values = [1, 2, 3, 4];
  const result = metrics.ema(values, 2);
  const expected = [1, 1.6666666666666665, 2.5555555555555554, 3.518518518518518];
  assert.equal(result.length, expected.length);
  result.forEach((value, index) => {
    assert.ok(Math.abs(value - expected[index]) < 1e-12);
  });
});

test('correlation detects perfect positive relationship', () => {
  const a = [1, 2, 3, 4];
  const b = [2, 4, 6, 8];
  assert.equal(metrics.correlation(a, b), 1);
});
