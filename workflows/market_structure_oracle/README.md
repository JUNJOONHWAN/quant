# Market Structure Oracle

Hermes Worker가 Spark의 동결된 전수 일봉과 ETF Flow D+2 관측 뷰만으로
현재 시장 구조와 과거 유사 상태의 후속 경로 분포를 계산하는 관리형 앱이다.
한 번 만든 전시장 상태 큐브에서 섹터·테마·ETF 바스켓의 조건부 상태를
추출하므로 범위 분석에서도 전체 시장 맥락을 버리지 않는다.

## Contract

- 원천: `daily_observations.sqlite3`, read-only/query-only
- 가격 특징: 해당 거래일까지의 정보만 사용
- ETF Flow: `massive_etf_flow_us_sessions_v1`
- 미래 수익률: 유사 상태의 결과 라벨과 워크포워드 검증에만 사용
- 외부 최신 데이터, 브라우저, FMP/Massive 재수집: 사용하지 않음
- 실행 주체: Hermes Operations Role Shell의 `hermes-worker-general`
- 조건부 공식: `scope future | full market state + scope internal structure + relative position + ETF Flow D+2`

위 read-only 계약은 상태 큐브 분석 단계의 계약이다. 공통 증분 저장소는
별도의 Oracle single-writer 단계가 갱신하며 AI Radar와 Oracle 분석기가
동일한 seal을 읽는다.

### 공통 증분 저장소의 corporate-action 원장

- 테이블: `corporate_action_versions`(source-preserving raw version),
  `corporate_actions`(earliest-availability projection)
- 발견: Massive `/stocks/v1/splits`를 기간 전체로 조회
- 교차확인: 발견된 symbol을 FMP `/stable/splits`로 확인
- 공식 seed: 검증된 발행사·거래소 공지를 raw artifact로 함께 보존
- provider 가용일: 최초 capture 날짜이며 이후 재수집해도 앞당기지 않음
- 공식 가용일: 문서에 적힌 announcement date
- PIT 공개 조건: `effective_date<=as_of AND available_date<=as_of`
- seal: visible record count와 canonical projection SHA-256을 v3 Oracle
  receipt와 shared-source fingerprint에 결속
- 소비자 정책: 공식 이벤트 또는 Massive/FMP 동일 비율 교차확인만 사용;
  단일 provider 이벤트는 원장에는 남지만 분석 입력에서는 fail closed

분할·역분할 보정은 renderer가 아니라 분석 packet 생성 전에 수행한다.
효력일 이전 가격에는 `old_shares/new_shares`, 거래량에는
`new_shares/old_shares`를 적용한다. 원본 값, raw artifact SHA, provider,
endpoint, capture event는 보존한다.

## Output

실행별 결과:

`STOCKDATA/QUANT_AI_RADAR/oracle/<as_of>/runs/<operations_run_id>/`

최신 포인터:

- `STOCKDATA/QUANT_AI_RADAR/oracle/<as_of>/latest.json`
- `STOCKDATA/QUANT_AI_RADAR/oracle/<as_of>/latest.html`
- `STOCKDATA/QUANT_AI_RADAR/oracle/<as_of>/latest-scopes/<scope>.json`
- `STOCKDATA/QUANT_AI_RADAR/oracle/<as_of>/latest-scopes/<scope>.html`

재사용 상태 큐브:

`STOCKDATA/QUANT_AI_RADAR/oracle/state_cubes/<as_of>/`

상태 큐브는 전수 일봉 close/volume 행렬, 날짜/심볼 축, 전시장 원특징과
인과적 z-state를 저장한다. 원천 DB 크기·mtime과 유니버스 SHA-256이
같을 때만 재사용한다.

`walk_forward_validation.forecast_confidence`가
`scenario_only_no_incremental_probability_skill`이면 유사국면 확률을 예측
우위로 표현하지 않고 시나리오 분포로만 취급한다.

## Hermes app

앱 ID는 `market-structure-oracle`이다. 앱 등록과 실행 receipt는
`/home/zooh/.hermes/operations/`에서 Operations App Manager가 관리한다.
자동 스케줄은 기본 비활성이고 Hermes Worker 온디맨드 실행을 기본으로 한다.

전체 시장:

```bash
hermes apps run market-structure-oracle --json
```

반도체 조건부 분석:

```bash
hermes apps run market-structure-oracle \
  --input-json '{"query":"반도체 섹터를 분석해줘","scope":"semiconductors"}' \
  --json
```

App Manager는 요청 JSON을 실행별 불변 입력 파일로 봉인하고 앱에는 그
경로만 전달한다. 질의 문자열을 shell command에 삽입하지 않는다.
