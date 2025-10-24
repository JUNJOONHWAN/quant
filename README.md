# Market Stability Dashboard

A static web application that visualizes cross-asset correlations and a derived market stability index directly in the browser.
The site no longer talks to Yahoo Finance from the client; instead, a GitHub Actions workflow runs `scripts/generate-data.js`, which
pulls daily prices for QQQ, SPY, TLT, GLD, and BTC-USD from Alpha Vantage, precomputes the correlation metrics, and uploads the
resulting JSON bundle (`static_site/data/precomputed.json`) alongside the static assets. Browsers simply download that JSON and
render gauges, history charts, and pair analytics—no backend services required.

## Repository layout

- `docs/market_stability_app_plan.md` – system design and background notes for the dashboard.
- `scripts/generate-data.js` – Alpha Vantage fetcher + metric precomputation used by CI and local previews.
- `static_site/` – GitHub Pages–ready frontend (HTML, CSS, JavaScript, and the generated data file).

## Deploy to GitHub Pages

### 1. Provision an Alpha Vantage API key

Create a free key at <https://www.alphavantage.co/support/#api-key>. You will need it both for local previews and for the automated
GitHub Actions deployment.

### 2. Store the key in the repository secrets

1. Open your GitHub repository.
2. Navigate to **Settings → Secrets and variables → Actions**.
3. Create a new repository secret named **`ALPHAVANTAGE_API_KEY`** and paste the key value.

### 3. One-click deploy with GitHub Actions (recommended)

The `.github/workflows/deploy.yml` workflow installs Node.js, generates the precomputed dataset, and publishes `static_site/` on every
push to `main`.

1. Push the repository to GitHub with your work living on the `main` branch.
2. In **Settings → Pages → Build and deployment**, pick **GitHub Actions** as the source (only required once).
3. Every push to `main` now triggers the workflow. The job fails fast if the API key is missing or Alpha Vantage returns an error, so
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

1. Export your Alpha Vantage key (or rely on the repository secret if you are running inside GitHub Codespaces with inherited secrets).
2. Generate the dataset locally.
3. Serve the static files.

> **Note:** The generator requires Node.js 18 or newer to use the built-in `fetch` API.

```bash
export ALPHAVANTAGE_API_KEY=your-key-here
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

## Data notice

The published numbers are derived from Alpha Vantage daily endpoints and are intended for informational purposes only. Accuracy and
availability are not guaranteed, and the project does not provide investment advice.
