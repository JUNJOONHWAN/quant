# Oracle 공통 DB 운영 계약

## 결론

Market Structure Oracle이 FMP·Massive 시장 데이터의 유일한 증분 writer다.
Quant AI Radar와 Oracle 리포트는 같은 봉인 SQLite 스냅샷을 읽는다.
ETF RADAR는 별도 제품이며 수집, 최신일 결정, 선택 점수, 리포트 생성,
성공 게이트 어디에도 의존하지 않는다.

```text
FMP 장기 PIT DB (불변)
             \
              +--> Oracle single-writer + lock --> 증분 DB + snapshot seal
             /                                  /                  \
Massive 일봉·ETF Flow + FMP ETF 구성종목       Oracle report       AI Radar
```

## 데이터 소유권

| 계층 | 데이터 | 역할 |
|---|---|---|
| 불변 base | FMP 장기 일봉, Massive Flow 역사, FMP ETF 구성 역사, FMP facts | 학습 및 장기 분석 기준 |
| Oracle incremental | base 종료 다음 거래일부터 Massive grouped daily | 신규 종목을 포함한 전시장 일봉 |
| Oracle incremental | Massive ETF Global fund flow | D+1/D+2 공개 지연을 반영한 ETF Flow |
| Oracle incremental | FMP ETF holdings | 신규 ETF 우선, 오래된 snapshot 다음 순서로 제한량 갱신 |
| Oracle snapshot seal | 목표일, 소스 계약, 행 게이트, Flow 신선도, receipt SHA-256 | 모든 소비자의 단일 완료 증거 |
| AI Radar feature snapshot | 가격·Flow·ETF-구성종목·섹터 회전 파생값 | 읽기 전용, 재수집 없음 |

## 매일 실행

1. `latest_closed_nyse_session()`은 미국 동부 오후 6시 이후 당일 완료
   세션을 목표로 선택한다. 주말과 NYSE 휴장일은 건너뛴다. 공급자 공개
   지연은 다음 수집 폴백에서 처리한다.
2. `fcntl` 잠금을 획득한다. Oracle과 AI Radar가 동시에 실행돼도 먼저
   들어온 한 프로세스만 writer가 된다.
3. 불변 base 종료일 이후 누락된 모든 NYSE 세션을 계산한다.
4. 각 세션은 Massive grouped daily를 먼저 요청한다. 현재 요금제의
   당일 접근이 403이면, 현재 실제 접근이 확인된 FMP 레거시
   `api/v4/batch-historical-eod` CSV를 원본 보존해 사용한다.
5. 레거시 bulk도 402/403/스키마 오류이면 최신 FMP 미국 주식·ETF
   universe를 갱신한 뒤 종목별 EOD를 240/min 제한으로 수집한다.
   2만 요청의 이론 실행시간은 약 83분이며 네트워크 재시도를 포함해
   75~90분 범위로 본다. 각 종목 checkpoint가 있으므로 중단 후에도
   완료분을 다시 받지 않고 이어서 실행한다.
6. 어느 경로든 소스별 과거 정상 커버리지의 90% 행 게이트를 통과한
   세션만 봉인한다. Massive와 FMP의 심볼 체계·커버리지가 다르므로
   Massive 기준 행 수를 FMP에 그대로 강요하지 않는다.
7. Massive ETF Flow 최근 processed-date 창을 갱신한다.
8. Flow의 `effective_date`, `processed_date`, `available_at_date`가 모두
   분석 기준일 이하인지 확인하며, 최신 effective가 D+2 허용 범위보다
   오래되면 봉인하지 않는다.
9. 신규 ETF 및 stale ETF 구성종목 후보를 계산한다. 신규 ETF를 먼저,
   오래된 snapshot을 다음으로 처리하며 일일 기본 예산은 50 ETF다.
10. `oracle_snapshot_seals`와
   `state/oracle_incremental_status.json`에 같은 receipt SHA-256을 기록한다.
11. AI Radar가 read-only overlay를 열 때 두 봉인의 일치, 목표일 전시장
   행 게이트, Flow 신선도, ETF 구성 가용일을 다시 검증한다.

## 신규 종목과 신규 ETF

- 신규 보통주: Massive grouped daily가 전시장 단위이므로 상장 후 첫
  거래일부터 Oracle 증분 DB와 정량 universe에 자동 등장한다.
- 신규 ETF: Massive ETF Flow ticker 또는 FMP ETF profile로 발견된다.
- 신규 ETF 구성종목: snapshot이 전혀 없는 ETF가 stale ETF보다 먼저
  FMP refresh 예산을 사용한다.
- 구성종목 자료가 아직 공개되지 않은 신규 ETF는 ETF 자체 가격·Flow
  분석은 가능하지만 종목 전달효과는 생성하지 않는다. 관계를 추측하지
  않고 다음 refresh backlog에 남긴다.
- 상폐 종목: 과거 base 행을 삭제하지 않는다. 현재 거래일에 거래 가능한
  행이 없으면 당일 universe에서만 빠진다.

## 룩어헤드 차단

- 가격: 확정된 거래일이 `as_of_date` 이하인 행만 사용한다.
- ETF Flow: `effective_date`, `processed_date`, `available_at_date`가 모두
  `as_of_date` 이하인 버전만 사용한다.
- ETF 구성종목: `effective_date`와 `available_date`가 기준일 이하인
  최신 visible snapshot만 사용한다.
- FMP profile의 섹터는 분류용이다. 과거 보유비중이나 Flow 공개시점을
  앞당기는 용도로 쓰지 않는다.
- AI 시장 synthesis는 파생 feature snapshot의 SHA-256에 결합되며,
  숫자를 새로 만들 수 없다.

## AI Radar 선택

모든 당일 거래 가능 종목을 먼저 정량 스캔한다. 생성형 추론은 고정 종목
목록이 아니라 다음 Oracle 증거를 점수화해 기본 최대 64 ETF와 192 종목만
선택한다.

- ETF: 5일·21일 가격 수익률, 5일·21일 Flow/자산, robust Flow z-score,
  가격-Flow 확인 또는 괴리
- 종목: 기준일에 visible한 ETF 구성 비중 × ETF Flow/자산의 합,
  영향 ETF 수, 상위 기여 ETF
- 시장: ETF 구성종목의 FMP 섹터를 이용한 회전 cluster, breadth,
  Flow 방향과 가격 방향

선택되지 않은 종목도 `coverage_ledger.jsonl`에 이유와 점수를 남긴다.
사용자가 `ai-radar analyze SYMBOL`을 요청하면 일일 선택 192개 밖의
종목도 같은 PIT·유동성·모델 계약을 통과하는 한 분석한다.

## 주요 명령

```bash
cd /home/zooh/Documents/GitHub/quant

# 최신 Oracle 공통 DB를 준비하거나 이미 완료된 봉인을 재사용
PYTHONPATH=$PWD /usr/bin/python3 \
  -m workflows.quant_ai_radar.prepare_shared_data

# 특정 완료 거래일을 명시해 복구
PYTHONPATH=$PWD /usr/bin/python3 \
  -m workflows.quant_ai_radar.prepare_shared_data \
  --target-as-of-date 2026-07-29 --force-repair

# 리포트 없이 준비 상태 확인
ai-radar preflight
ai-radar status

# 전체 일일 분석 / 비공개 shadow / 개별 종목
ai-radar daily
ai-radar daily --shadow
ai-radar analyze AAPL NVDA
```

## 실패·복구 계약

- 전시장 행 부족, 누락 NYSE 세션, stale Flow, snapshot seal hash 불일치,
  구성종목 global 가용일 초과, 모델 release 불일치 중 하나라도 있으면
  정상 리포트를 publish하지 않는다.
- 재실행은 완료 checkpoint와 기존 DB 행을 재사용한다.
- 수집 실패를 ETF RADAR 파일이나 고정 문구로 대체하지 않는다.
- 같은 미국 달력일의 Massive grouped daily가 `NOT_AUTHORIZED`이면 FMP
  레거시 bulk, FMP 종목별 EOD 순서로 복구한다.
- FMP 공식 가격표상 Starter는 Bulk and Batch Delivery를 포함하지 않고
  Ultimate부터 포함한다. 현재 레거시 v4 bulk 접근은 실측 성공했지만
  정식 entitlement가 아니므로 `legacy_best_effort`로 기록하며, 실패 시
  종목별 EOD가 최종 폴백이다.
- FMP 구성종목 일부 실패는 backlog와 오류 receipt에 남긴다. 기존
  visible snapshot이 허용 lag 안에 있으면 시장 전체 봉인은 유지할 수
  있지만 실패한 신규 ETF의 종목 전달효과는 만들지 않는다.

## 산출물

```text
STOCKDATA/QUANT_AI_RADAR/
  oracle/incremental/
    normalized/daily_observations.sqlite3
    state/oracle_incremental_status.json
    state/oracle_incremental.lock
  shared/etf_relation_index.sqlite3
  runs/YYYY-MM-DD/
    oracle_market_features.json
    universe_manifest.json
    selection_manifest.json
    coverage_ledger.jsonl
    selected_run_queue.sqlite3
    security_judgements.jsonl
    market_report.json
    market_report.html
```

## 2026-07-29 DGX 실측 인수 결과

- Massive grouped daily: 당일 접근 정책으로 403, 원본 오류 receipt 보존
- FMP stable EOD bulk: 현재 Starter 키에서 402
- FMP legacy v4 batch EOD: HTTP 200, 원본 60,871행
- 최신 FMP 미국 주식·ETF universe: 16,833개
- universe와 일치해 저장된 2026-07-29 FMP 일봉: 9,914개
- FMP 소스별 봉인 하한: 8,541개, 실제 9,914개로 통과
- Massive ETF Flow: 최신 effective/available date 2026-07-28, D+2 gate 통과
- 동일 날짜 재실행: `reused_existing_complete`, 신규 Flow/관계행 0
- AI Radar prepare-only: Oracle feature snapshot 생성, 64 ETF + 192 종목
- Market Structure Oracle: 2026-07-29 전체 시장 실행 `PASS`

이 결과는 레거시 endpoint가 영구 보장된다는 뜻이 아니다. 다음 실행에서
레거시가 막히면 최신 universe의 미수집 심볼만 종목별 EOD 경로로 자동
전환하고, 최종 행 게이트를 통과해야만 같은 방식으로 봉인한다.

## 재학습 여부

이번 변경은 원천 수집 소유권과 일일 feature 생성 경로를 정리한 것이다.
모델 입력의 `quant.analysis_packet.v3` 의미, PIT 정책, 학습 과제와 응답
계약은 바꾸지 않았으므로 즉시 재학습할 필요는 없다. 패킷 스키마나 Flow
가시성 정의, 구성종목 전달효과 공식을 바꿀 때만 새 dataset version,
동결 test, LoRA 재학습과 release 승인을 수행한다.
