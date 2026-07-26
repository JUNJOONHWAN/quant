# Market Structure Oracle

Hermes Worker가 Spark의 동결된 전수 일봉과 ETF Flow D+2 관측 뷰만으로
현재 시장 구조와 과거 유사 상태의 후속 경로 분포를 계산하는 관리형 앱이다.

## Contract

- 원천: `daily_observations.sqlite3`, read-only/query-only
- 가격 특징: 해당 거래일까지의 정보만 사용
- ETF Flow: `massive_etf_flow_us_sessions_v1`
- 미래 수익률: 유사 상태의 결과 라벨과 워크포워드 검증에만 사용
- 외부 최신 데이터, 브라우저, FMP/Massive 재수집: 사용하지 않음
- 실행 주체: Hermes Operations Role Shell의 `hermes-worker-general`

## Output

실행별 결과:

`STOCKDATA/QUANT_AI_RADAR/oracle/<as_of>/runs/<operations_run_id>/`

최신 포인터:

- `STOCKDATA/QUANT_AI_RADAR/oracle/<as_of>/latest.json`
- `STOCKDATA/QUANT_AI_RADAR/oracle/<as_of>/latest.html`

`walk_forward_validation.forecast_confidence`가
`scenario_only_no_incremental_probability_skill`이면 유사국면 확률을 예측
우위로 표현하지 않고 시나리오 분포로만 취급한다.

## Hermes app

앱 ID는 `market-structure-oracle`이다. 앱 등록과 실행 receipt는
`/home/zooh/.hermes/operations/`에서 Operations App Manager가 관리한다.
자동 스케줄은 기본 비활성이고 Hermes Worker 온디맨드 실행을 기본으로 한다.
