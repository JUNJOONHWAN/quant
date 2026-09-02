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
FMP 일봉·활성종목·기업행위 + Massive ETF Flow  Oracle report       AI Radar
```

## 데이터 소유권

| 계층 | 데이터 | 역할 |
|---|---|---|
| 불변 base | FMP 장기 일봉, Massive Flow 역사, FMP ETF 구성 역사, FMP facts | 학습 및 장기 분석 기준 |
| Oracle incremental | base 종료 다음 거래일부터 FMP 일별 전시장 원장 | 활성 주식·ETF의 BAR/NO_BAR/QUARANTINED_INVALID_BAR/ERROR 전수 상태 |
| Oracle incremental | Massive ETF Global fund flow | D+1/D+2 공개 지연을 반영한 ETF Flow |
| Oracle incremental | FMP ETF holdings | 신규 ETF 우선, 오래된 snapshot 다음 순서로 제한량 갱신 |
| Oracle incremental | FMP split 및 symbol-change | 분할·병합 조정과 구티커→신티커 계보 |
| Oracle snapshot seal | 목표일, 소스 계약, 행 게이트, Flow 신선도, receipt SHA-256 | 모든 소비자의 단일 완료 증거 |
| AI Radar feature snapshot | 가격·Flow·ETF-구성종목·섹터 회전 파생값 | 읽기 전용, 재수집 없음 |

## 매일 실행

1. `latest_closed_nyse_session()`은 미국 동부 오후 6시 이후 당일 완료
   세션을 목표로 선택한다. 주말과 NYSE 휴장일은 건너뛴다. 공급자 공개
   지연은 다음 수집 폴백에서 처리한다.
2. `fcntl` 잠금을 획득한다. Oracle과 AI Radar가 동시에 실행돼도 먼저
   들어온 한 프로세스만 writer가 된다.
3. 불변 base 종료일 이후 누락된 모든 NYSE 세션을 계산한다.
4. FMP `company-screener`를 NASDAQ·NYSE·AMEX·CBOE별로 페이지 끝까지
   읽고, `actively-trading-list`, `available-traded/list`, stock/ETF
   reference 목록으로 교차 확인한 당일 활성 master를 불변 파일로 남긴다.
   `country=US` 필터는 미국 거래소 상장 해외 본사를 누락시키므로 쓰지 않는다.
   어느 shard든 페이지 반복, 상한 도달 의심, 빈 중간 페이지, 파싱 경고가
   있으면 `warnings`로 완료하지 않고 당일 봉인을 중단한다.
5. 각 세션은 FMP 레거시 `api/v4/batch-historical-eod` CSV를 원본 보존해
   일괄 수집한다. Massive grouped stock daily는 가격 원장에 쓰지 않는다.
6. 활성 master에서 bulk BAR가 없는 심볼만 FMP stable 종목별 EOD를
   240/min으로 조회한다. 정상 BAR는 `BAR`, 성공했지만 해당일 행이
   없으면 `NO_BAR`, bulk와 stable 모두 같은 비정상 OHLC를 반환하면
   원본·버전만 남기고 분석 projection에서 제외한
   `QUARANTINED_INVALID_BAR`, 요청·파싱 실패는 `ERROR`로 기록한다.
7. master 전 심볼이 `BAR`, `NO_BAR`, `QUARANTINED_INVALID_BAR` 중 하나이고
   `ERROR=0`이며 전시장 정상 BAR 하한을 통과한 세션만 봉인한다. 전수
   JSONL ledger의 SHA-256도 receipt에 결합한다.
8. 2026-07-15~최초 표준화일까지는 당시 활성 master가 보존돼 있지
   않으므로 같은 날 FMP legacy BAR가 존재한 심볼만 분석 master로
   복원한다. 현재 active 목록을 과거로 투영하지 않는다. 이후에는 매일
   저장한 active master가 진짜 일별 PIT 기준이다.
   현재 목표일은 저장된 active master와 같은 날 FMP legacy BAR의 합집합을
   사용한다. active 디렉터리가 장중·상태 변경으로 놓친 BAR도 버리지 않으며,
   추가 심볼은 ledger에
   `same_day_fmp_legacy_bar_not_in_active_directory`로 구분한다.
9. Massive ETF Flow 최근 processed-date 창을 갱신한다.
10. Flow의 `effective_date`, `processed_date`, `available_at_date`가 모두
   분석 기준일 이하인지 확인하며, 최신 effective가 D+2 허용 범위보다
   오래되면 봉인하지 않는다.
11. 신규 ETF 및 stale ETF 구성종목 후보를 계산한다. 신규 ETF를 먼저,
   오래된 snapshot을 다음으로 처리하며 일일 기본 예산은 50 ETF다.
12. FMP split 원장과 FMP `symbol-change` 계보를 source-preserving
    version으로 갱신한다. 원 티커 가격은 덮어쓰지 않고 분석 연결만 한다.
13. `oracle_snapshot_seals`와
   `state/oracle_incremental_status.json`에 같은 receipt SHA-256을 기록한다.
14. AI Radar가 read-only overlay를 열 때 두 봉인의 일치, 목표일 전시장
   행 게이트, Flow 신선도, ETF 구성 가용일을 다시 검증한다.
15. `ai-radar daily`는 준비 단계가 반환한 Oracle source fingerprint와
   기존 리포트·이메일 proof의 fingerprint가 정확히 같을 때만 재사용한다.
   같은 날짜라도 봉인이 달라지면 기존 queue와 산출물을 `runs/<date>/revisions/`
   아래 보존하고 새 queue로 다시 분석한다.
16. green v2 리포트, 420px HTML, 첨부 HTML, Gmail API message ID가 모두
   있어야 일일 실행이 PASS다. 같은 fingerprint와 report SHA의 재시도는
   중복 발송하지 않으며, source revision은 `[AI Radar 정정]` 메일 한 번만
   별도 dedupe key로 발송한다.

## 신규 종목과 신규 ETF

- 신규 보통주·ETF: FMP 당일 활성 master와 FMP 일봉에 자동 등장한다.
- 신규 ETF는 Massive ETF Flow ticker 또는 FMP ETF profile로도 교차 발견된다.
- 신규 ETF 구성종목: snapshot이 전혀 없는 ETF가 stale ETF보다 먼저
  FMP refresh 예산을 사용한다.
- 구성종목 자료가 아직 공개되지 않은 신규 ETF는 ETF 자체 가격·Flow
  분석은 가능하지만 종목 전달효과는 생성하지 않는다. 관계를 추측하지
  않고 다음 refresh backlog에 남긴다.
- 상폐 종목: 과거 base 행을 삭제하지 않는다. 현재 거래일에 거래 가능한
  행이 없으면 당일 universe에서만 빠진다.

## 룩어헤드 차단

- 가격: 확정된 거래일이 `as_of_date` 이하인 행만 사용한다.
- 티커 변경: `event_date`와 최초 관측 `available_date`가 모두 기준일
  이하일 때만 구티커→신티커 계보를 사용한다.
- 분할·병합: 효력일과 최초 관측일이 모두 기준일 이하인 검증 이벤트만
  과거 가격/거래량 조정에 사용한다.
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
- FMP 레거시 bulk가 실패하면 당시 전시장 master를 안전하게 복원할 수
  없으므로 조용히 Massive 가격으로 대체하지 않고 fail-closed 한다.
- 레거시 bulk의 개별 누락은 FMP 종목별 EOD로 복구한다.
- FMP 공식 가격표상 Starter는 Bulk and Batch Delivery를 포함하지 않고
  Ultimate부터 포함한다. 현재 레거시 v4 bulk 접근은 실측 성공했지만
  정식 entitlement가 아니므로 `legacy_best_effort`로 기록하며, 실패 시
  종목별 EOD가 최종 폴백이다.
- FMP 구성종목 일부 실패는 backlog와 오류 receipt에 남긴다. 기존
  visible snapshot이 허용 lag 안에 있으면 시장 전체 봉인은 유지할 수
  있지만 실패한 신규 ETF의 종목 전달효과는 만들지 않는다.
- active master, symbol-change, 일별 coverage 중 경고가 하나라도 있으면
  완료 receipt를 만들지 않는다. 경고를 남긴 COMPLETE는 허용하지 않는다.
- 모델 queue 한 건이라도 `error`이면 시장 종합·메일을 만들지 않는다.
  다음 KST readiness/recovery 실행은 같은 source fingerprint queue를
  checkpoint부터 재개한다.
- Oracle source fingerprint가 바뀌었는데 날짜가 같다는 이유로 기존
  리포트나 이메일을 완료로 재사용하지 않는다.

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
    market_report_email_420.html
    revisions/<old-source>_to_<new-source>_<kst>/
  email/email_delivery_history.json
```

## 2026-07-31 DGX 표준화 실측 인수 결과

- 표준화 범위: 2026-07-15~2026-07-30, NYSE 거래일 12개
- 가격 source: FMP만 사용, 증분 Massive 일봉 0행
- 7/30 FMP active master: 11,444
- 7/30 active master와 당일 legacy BAR 합집합: 11,449
- active 디렉터리 밖 당일 BAR: HBRD, NTWOU, ONTX, SVREW, TROT
- 7/30 coverage: BAR 11,072, NO_BAR 373,
  QUARANTINED_INVALID_BAR 4, ERROR 0
- 12거래일 전체 격리: 12행, model-facing 비정상 OHLC 0행
- FMP symbol-change: 원본 5,418행, 유효 계보 5,417행,
  공급자 no-op `GOAI→GOAI` 1행은 사유와 함께 명시 제외
- active/symbol-change warnings: 0
- SQLite `integrity_check`: ok, snapshot receipt와 status SHA-256 일치
- Massive ETF Flow 최신 effective date: 2026-07-29
- 일일 AI Radar 2026-07-30: 선택 queue 253건 완료, Gmail v3 message ID 확보

이 결과는 레거시 endpoint가 영구 보장된다는 뜻이 아니다. 레거시가
성공한 뒤 활성 master의 미수집 심볼만 종목별 EOD로 전환하며, 레거시
자체가 실패해 전시장 기준을 만들 수 없으면 봉인하지 않는다.

## 재학습 여부

이번 변경은 원천 수집 소유권과 일일 feature 생성 경로를 정리한 것이다.
모델 입력의 `quant.analysis_packet.v3` 의미, PIT 정책, 학습 과제와 응답
계약은 바꾸지 않았으므로 즉시 재학습할 필요는 없다. 패킷 스키마나 Flow
가시성 정의, 구성종목 전달효과 공식을 바꿀 때만 새 dataset version,
동결 test, LoRA 재학습과 release 승인을 수행한다.
