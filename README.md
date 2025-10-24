# Market Stability Dashboard

A static web application that visualizes cross-asset correlations and a derived market stability index directly in the browser. It fetches daily prices for QQQ, SPY, TLT, GLD, and BTC-USD from the unofficial Yahoo Finance API at runtime and renders gauges, historical charts, and pair-level analytics without any backend services.

## Repository layout

- `docs/market_stability_app_plan.md` – system design and background notes for the dashboard.
- `static_site/` – GitHub Pages–ready frontend (HTML, CSS, and JavaScript).

## Deploy to GitHub Pages

1. Push the contents of this repository to the branch that GitHub Pages serves (commonly `main`).
2. In **Settings → Pages → Build and deployment**, choose **Deploy from a branch**, then select the branch and the `/static_site` folder.
3. Wait for the deployment to finish and open the URL that GitHub Pages provides.

### Push changes from your local clone

If GitHub에서 변경 사항이 보이지 않는다면 로컬 변경분이 커밋되고 원격 저장소로 푸시되었는지 확인하세요.

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

푸시가 완료되면 GitHub 저장소에서 커밋이 보여야 하며, 이후 Pages 배포가 자동으로(또는 몇 분 지연 후) 반영됩니다. 만약 Pages가 자동으로 갱신되지 않으면 **Settings → Pages** 화면에서 배포 상태를 확인하거나 “Rebuild” 버튼을 눌러 다시 배포를 트리거하세요.

## Local preview

Use a lightweight HTTP server so that the browser can perform HTTPS requests to Yahoo Finance:

```bash
cd static_site
python -m http.server 8000
```

Open <http://localhost:8000/> to check the dashboard before publishing.

## Data notice

The application relies on unofficial Yahoo Finance endpoints. Availability and accuracy are not guaranteed; treat the results as informational only and not as investment advice.
