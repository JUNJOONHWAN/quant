# Market Stability Dashboard

A static web application that visualizes cross-asset correlations and a derived market stability index directly in the browser.
The site no longer talks to Yahoo Finance from the client; instead, a GitHub Actions workflow runs `scripts/generate-data.js`, which
pulls **adjusted open/close prices** for QQQ, IWM, SPY, TLT, GLD, and BTC-USD from Financial Modeling Prep (FMP), precomputes the correlation metrics (signals are built
on IWM·SPY·TLT·GLD·BTC-USD, while the backtest benchmark stays on QQQ/TQQQ), and uploads the
resulting JSON bundle (`static_site/data/precomputed.json`) alongside the static assets. Browsers simply download that JSON and
render gauges, history charts, Fusion-Newgate (aggressive_plus) regime bands, and pair analytics—no backend services required.

## Repository layout

- `docs/market_stability_app_plan.md` – system design and background notes for the dashboard.
- `scripts/generate-data.js` – FMP fetcher + metric precomputation used by CI and local previews.
- `static_site/` – GitHub Pages–ready frontend (HTML, CSS, JavaScript, and the generated data file).

## Deploy to GitHub Pages

### 1. Provision an FMP API key

Create a key at <https://site.financialmodelingprep.com/developer>. You will need it both for local previews and for the automated
GitHub Actions deployment. The free tier is sufficient for the daily refresh cadence used by this project.

### 2. Store the key in the repository secrets

1. Open your GitHub repository.
2. Navigate to **Settings → Secrets and variables → Actions**.
3. Create a new repository secret named **`FMP_API_KEY`** and paste the key value.

### 3. One-click deploy with GitHub Actions (recommended)

The `.github/workflows/deploy.yml` workflow installs Node.js, generates the precomputed dataset, and publishes `static_site/` on every
push to `main`.

1. Push the repository to GitHub with your work living on the `main` branch.
2. In **Settings → Pages → Build and deployment**, pick **GitHub Actions** as the source (only required once).
3. Every push to `main` now triggers the workflow. The job fails fast if the API key is missing or FMP returns an error, so
you immediately know when the data refresh needs attention. You can also run the workflow manually from the **Actions** tab.

### Manual branch deployment (fallback option)

If you disable or delete the workflow—or want to publish without GitHub Actions—you can still configure Pages to serve directly from a
branch, but you must generate the JSON yourself first:

1. Export the API key and run `npm run generate:data` locally to create `static_site/data/precomputed.json`.
2. Commit the generated file (or copy it into the branch you're deploying).
3. In **Settings → Pages → Build and deployment**, choose **Deploy from a branch**, then select the branch and the `/static_site` folder.
4. Wait for the deployment to finish and open the URL that GitHub Pages provides.

### Push changes from your local clone

GitHub에 반영되지 않는다면 로컬 변경분이 커밋되고 원격 저장소로 푸시되었는지 확인하세요.

```bash
# 1. 현재 작업 상태 확인
git status

# 2. 변경된 파일을 스테이징
git add .

# 3. 커밋 생성 (메시지는 상황에 맞게 수정)
git commit -m "Update static site"

# 4. 원격(main 브랜치)으로 푸시
git push origin main
```

푸시 후 Pages 배포 상태는 **Settings → Pages** 페이지에서 확인할 수 있으며, 필요하면 “Rebuild”를 눌러 수동으로 재배포할 수 있습니다.

## Local preview

1. Export your FMP key (or rely on the repository secret if you are running inside GitHub Codespaces with inherited secrets).
2. Generate the dataset locally.
3. Serve the static files.

> **Note:** The generator requires Node.js 18 or newer to use the built-in `fetch` API.

```bash
export FMP_API_KEY=your-key-here
npm run generate:data
cd static_site
python -m http.server 8000
```

Then open <http://localhost:8000/>. The frontend will detect the freshly generated JSON and render the dashboard without making any
external network calls. If the JSON is missing, the page shows a helpful error explaining how to create it.

## Automated tests

The shared numerical utilities that power the dashboard can be exercised with Node's built-in test runner:

```bash
npm test
```

Running the suite is a quick way to verify that correlation, EMA, and weighting logic behave as expected when you make code changes.

## Historical cache (2017+)

Both the Node generator and the Python helper pull from the same FMP endpoints. The cache now stores **adjusted closes and matching adjusted opens** (`prices` / `opens` arrays) so that all downstream backtests can model T+1 open execution. To refresh the long-horizon cache that lives in
`static_site/data/historical_prices.json`, run:

1. Export `FMP_API_KEY`.
2. Download the full history (2017-01-01 → today):

   ```bash
   python scripts/fetch_long_history.py --force
   ```

3. Regenerate the precomputed metrics:

   ```bash
   npm run generate:data
   ```

The Python script keeps every asset in sync with the JSON format that the frontend expects (including the `opens` arrays). Keeping the cache under version control
helps guarantee reproducible full-history datasets even if rate limits block a fresh pull.

## Execution model

Signals are calculated using end-of-day data, but **all strategy backtests (CLI + UI) assume trades execute at the next session’s adjusted open**. The generated datasets therefore expose both closes (`priceSeries`) and opens (`priceSeriesOpen`), and the JavaScript/Python backtests convert On positions to open→close P&L while Off stays in cash. Neutral mode continues to hold the underlying (close→close) so that benchmark comparisons remain intuitive.

## Data notice

The published numbers are derived from Financial Modeling Prep daily endpoints and are intended for informational purposes only. Accuracy and
availability are not guaranteed, and the project does not provide investment advice.
