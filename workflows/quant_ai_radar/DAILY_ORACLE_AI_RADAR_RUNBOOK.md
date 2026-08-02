# Oracle DB → Quant AI Radar → 이메일 일일 운영서

## 목적과 완료 정의

이 문서는 DGX Spark의 일일 시장 데이터 수집부터 Quant AI Radar 이메일
전송까지의 단일 운영 계약이다. 목표는 단순히 프로세스가 종료되는 것이
아니라 다음 네 증거가 같은 미국 시장 기준일과 같은 Oracle source
fingerprint로 결합되는 것이다.

1. Oracle 증분 SQLite가 `COMPLETE`이고 snapshot receipt SHA-256이
   SQLite seal과 상태 JSON에서 일치한다.
2. 모든 필수 심볼이 일별 coverage ledger에서 종결되고 `ERROR=0`,
   `warnings=[]`다.
3. 승인된 BF16 `Qwen3-8B-FLOW` 리포트가 green이며 Oracle fingerprint와
   일치한다.
4. 420px 이메일 HTML, 상세 HTML 첨부, Gmail message ID가 모두 기록된다.

날짜만 같거나 프로세스 exit code만 0인 상태는 완료가 아니다.

## 소스와 소유권

| 계층 | 유일한 역할 |
|---|---|
| 불변 base DB | 2026-07-14까지 학습·장기 분석 이력. 일일 실행은 쓰지 않는다. |
| FMP | NASDAQ·NYSE·AMEX·CBOE active master, 전시장 일봉, 종목별 일봉 폴백, ETF 구성종목, split, symbol-change |
| Massive | ETF Global fund flow와 split 발견 원장. 주식·ETF 일봉 가격 원장으로 쓰지 않는다. |
| Market Structure Oracle | 증분 DB의 유일한 writer, 전수 coverage와 snapshot seal 소유자 |
| Quant AI Radar | 봉인된 base+incremental overlay의 read-only 소비자 |
| Gmail delivery | green 리포트의 최종 배포 proof |
| ETF RADAR | 별도 제품. 이 경로의 데이터·완료 게이트가 아니다. |

## FMP 전수 수집 방식

### 당일 마스터

FMP `company-screener`를 NASDAQ, NYSE, AMEX, CBOE별로
`isActivelyTrading=true` 조건으로 끝 페이지까지 읽는다. `country=US`는
미국 거래소에 상장된 해외 본사 종목을 누락시키므로 사용하지 않는다.
active list, stock list, ETF list, legacy reference 목록은 분류·교차검증에
쓰며 원본 payload와 SHA-256을 남긴다.

페이지 반복, 상한 도달 의심, 중간 빈 페이지, 파싱 손실, shard 경고는
참고 경고가 아니다. `warnings=[]`가 아니면 active master가 불완전하므로
Oracle 봉인을 중단한다.

현재 목표일의 필수 집합은 다음과 같다.

```text
저장된 당일 FMP active master
UNION
같은 날 FMP legacy EOD에 실제 BAR가 존재하는 심볼
UNION
운영 protected symbol
```

active 디렉터리에 없지만 당일 BAR가 있는 종목은 버리지 않고
`same_day_fmp_legacy_bar_not_in_active_directory`로 기록한다.

### 일봉

1. FMP legacy `/api/v4/batch-historical-eod?date=YYYY-MM-DD`로 당일
   전시장 CSV를 원본 보존한다.
2. 필수 집합에서 bulk BAR가 없는 심볼만
   `/stable/historical-price-eod/full`을 날짜 범위 1일로 조회한다.
3. 호출은 전역 300/min 계약보다 낮은 240/min 운영 한도로 제한한다.
4. Massive grouped stock daily는 사용하지 않는다.

심볼별 terminal outcome은 네 가지뿐이다.

| outcome | 의미 | 분석 projection |
|---|---|---|
| `BAR` | 정상 OHLCV 확보 | 포함 |
| `NO_BAR` | 요청 성공, 해당일 행 없음 | 미포함, 정상 종결 |
| `QUARANTINED_INVALID_BAR` | bulk와 exact FMP가 같은 비정상 OHLC를 반환 | 원본·version 보존, current projection 제외 |
| `ERROR` | 요청·파싱·분류 실패 | 봉인 차단 |

각 일자의 ledger 행 수는 필수 집합 수와 같아야 하고, 심볼 중복이 없어야
하며, outcome 합계도 필수 집합 수와 같아야 한다. ledger SHA-256은 Oracle
receipt에 결합된다.

### 과거 2026-07-15~07-29의 특별 처리

당시 active master 파일을 매일 보관하지 않았기 때문에 현재 active 목록을
과거로 투영하면 룩어헤드가 된다. 이 구간은 같은 날 FMP legacy BAR가 실제
존재한 심볼만 historical master로 복원했다. 2026-07-30부터는 매일 저장한
active master가 point-in-time 기준이다.

## 분할·병합과 티커 변경

- Massive split date-range 원장으로 전시장 이벤트를 발견한다.
- FMP `/stable/splits` 종목별 결과로 비율을 교차 확인한다.
- 발행사·거래소 공식 공지가 있으면 그 근거를 별도 보존한다.
- AI 가격 조정은 공식 근거가 있거나 Massive와 FMP의 비율이 정확히
  일치한 이벤트만 적용한다.
- `effective_date`와 최초 관측 `available_date`가 모두 분석일 이하일
  때만 보인다.
- FMP `symbol-change`는 한 번에 충분한 상한으로 수집한다. 응답이 상한에
  닿으면 잘린 것으로 보고 실패한다. 페이지 반복을 정상 완료로 보지 않는다.
- 구티커와 신티커의 원시 가격은 덮어쓰지 않는다. 계보 projection만
  별도로 유지한다.
- 공급자 no-op `oldSymbol == newSymbol`은 조용히 삭제하지 않고
  `provider_no_op_same_symbol` 사유로 manifest에 명시 제외한다.

## 일일 실행 순서

Hermes의 표준 KST readiness 실행은 화~토요일 06:00, 12:00, 18:00이다.
06:00 실행이 미완료일 때를 위한 07:05 복구 실행이 별도로 있다.

```text
Hermes App Manager
  → ai-radar daily
  → prepare_shared_data
  → Oracle writer lock
  → FMP active master + FMP EOD 전수 coverage
  → Massive ETF Flow D+1/D+2 가시성
  → split/symbol lineage/ETF 구성 갱신
  → Oracle snapshot seal
  → read-only shared binding + fingerprint
  → 관계 인덱스 증분 갱신
  → 전 종목 정량 스캔
  → 최대 64 ETF + 192 종목 동적 선택
  → Qwen3-8B-FLOW queue 재개 또는 새 추론
  → 시장·섹터·종목 다단계 해설
  → green 품질 게이트
  → 420px Gmail + 상세 HTML 첨부
  → message ID receipt
```

06:00에 공급자 공개가 아직 끝나지 않았거나 추론 한 건이 실패하면
완료로 표시하지 않는다. 07:05, 12:00, 18:00 실행은 같은 fingerprint의
checkpoint queue를 이어서 처리한다. 이미 같은 fingerprint·report SHA의
메일 proof가 있으면 수집 검증만 하고 중복 생성·중복 발송하지 않는다.
07:05에 06:00 App Manager lock이 아직 살아 있으면 recovery wrapper는
중복 실행하지 않고 `WATCH_PENDING`, `complete=false`, 다음 12:00 확인을
출력한다. 이것은 발행 완료가 아니라 활성 주 실행을 지켜보는 상태다.

## 같은 날짜의 Oracle 개정

같은 `as_of_date`라도 Oracle seal이 바뀌면 이전 리포트는 현재 자료의
완료 증거가 아니다.

1. `prepare_shared_data`가 현재 `source_fingerprint_sha256`을 반환한다.
2. 최신 green report의 `shared_oracle_store.source_fingerprint_sha256`와
   정확히 비교한다.
3. 다르면 기존 queue와 산출물 사본을
   `runs/<date>/revisions/<old>_to_<new>_<KST>/`에 보존한다.
4. 새 fingerprint로 queue를 만들고 선택·추론·종합·렌더링을 다시 한다.
5. 기존 이메일과 report SHA가 같으면 발송하지 않는다.
6. source 또는 report SHA가 실제로 달라졌으면
   `[AI Radar 정정]` 제목과 revision dedupe key로 한 번만 발송한다.
7. 정정 메일도 Gmail message ID가 없으면 PASS가 아니다.

## 완료 게이트와 감시 상태

### PASS

- Oracle `status=COMPLETE`
- `missing_sessions=[]`
- active/symbol-change `warnings=[]`
- 모든 coverage ledger `ERROR=0`
- SQLite `integrity_check=ok`
- post-base `daily_observations.source`는 FMP뿐
- model-facing 비정상 OHLC 0
- snapshot receipt SHA 일치
- AI report Oracle fingerprint 일치
- queue의 `pending/running/error=0`
- quality green, 모든 점수 8/10 이상
- Gmail v3 `complete=true`, `message_id` 존재

### WAITING 또는 RETRY

공급자 공개 대기, 실행창 대기, 같은 fingerprint queue의 미완료 항목은
완료가 아니다. heartbeat에는 대기 이유, 남은 queue 수, 다음 KST 실행을
표시한다.

### FAIL

`warnings`, `ERROR`, source hash 불일치, 모델 queue error, Gmail 실패,
잘린 pagination, 비정상 OHLC의 projection 유입 중 하나라도 있으면
실패다. “경고가 있지만 COMPLETE” 상태는 허용하지 않는다.

## 장애·삽질 기록과 해결

### 1. Massive 일봉과 FMP 일봉의 역할 혼동

- 문제: 증분 가격 원장에 Massive grouped daily를 섞으면 학습 원천과
  일일 원천의 의미가 달라지고 누락 판정도 불명확해졌다.
- 해결: 주식·ETF 일봉은 FMP만 사용한다. Massive는 ETF Flow와 split
  발견에만 남겼다.
- 재발 방지: shared binding이 base 종료 뒤 Massive 일봉을 발견하면
  즉시 거부한다.

### 2. 현재 active 목록을 과거 날짜에 적용할 위험

- 문제: 07-15~07-29 당시 active snapshot이 없는데 현재 목록으로 채우면
  신규상장·상폐에 룩어헤드가 생긴다.
- 해결: 과거일은 당일 FMP legacy BAR 존재만으로 복원하고 현재 목록을
  소급하지 않았다.
- 제한: 이 구간은 당시 `NO_BAR`였던 active 심볼까지 재구성할 수는 없다.
  이후 날짜는 매일 active master를 보존한다.

### 3. symbol-change 1,000행 반복 pagination

- 문제: 지원되지 않는 page 방식이 같은 1,000행을 반복 반환할 수 있었다.
  경고만 남기면 잘린 계보가 COMPLETE가 된다.
- 해결: 단일 10,000행 요청으로 전환하고 응답이 상한에 닿으면 실패한다.
  현재 원본 5,418행, 유효 5,417행, no-op 1행 명시 제외다.

### 4. 비정상 OHLC 12행

- 문제: bulk 원천의 high/low 관계가 깨진 행을 그대로 투영할 수 있었다.
- 해결: exact FMP 종목별 endpoint로 다시 확인한다. 같은 비정상 행이면
  원본·version은 보존하되 `QUARANTINED_INVALID_BAR`로 projection에서
  제거한다.
- 재발 방지: model-facing OHLC 정합성 SQL이 0행이어야 봉인한다.

### 5. 현재 active directory 밖 당일 BAR 5개

- 문제: active master 11,444개만 필수 집합으로 쓰면 실제 당일 FMP BAR가
  있는 HBRD, NTWOU, ONTX, SVREW, TROT를 버리게 된다.
- 해결: 현재 목표일을 `active master ∪ same-day legacy BAR`로 바꿨다.
  7/30 필수 집합은 11,449개다.

### 6. 날짜만 같은 오래된 AI Radar 재사용

- 문제: Oracle seal이 바뀌어도 `as_of_date`가 같고 기존 메일이 있으면
  일일 앱이 생성 단계를 건너뛸 수 있었다.
- 해결: prepare fingerprint, report fingerprint, email proof fingerprint와
  report SHA를 모두 비교한다. source revision은 새 queue와 정정 메일을
  요구한다.

### 7. 2026-07-31 12:11 KST 추론 1건 실패

- 증거: queue 252 done, 1 error, 3 excluded로 시장 종합을 차단했고 Hermes
  cron receipt는 FAIL이었다.
- 처리: checkpoint를 유지한 재실행에서 253 done, 3 excluded로 회복했고
  12:47 KST Gmail v3 message ID를 확보했다.
- 강화: 06:00·12:00·18:00 readiness와 07:05 recovery를 유지하며,
  queue error가 0이 되기 전에는 heartbeat와 App Manager가 PASS를
  주장하지 않는다.

### 8. SKY 구조화 응답 길이 초과와 실패 원문 누락

- 문제: 최신 Oracle 지문 재판단에서 SKY가 초기 호출과 두 번의 계약 복구
  호출 뒤에도 JSON 파싱에 실패했다. 기존 오류 원장은 예외명만 저장해 첫
  실패의 실제 응답을 사후 판별할 수 없었다.
- 확인된 재현: 보강된 오류 원장에서 최종 응답은 `finish_reason=length`,
  `completion_tokens=1400`이었다. 모델이 `regime` 뒤에서 공백을 반복해 필수
  `unknowns`와 닫는 중괄호를 쓰기 전에 잘렸다.
- 해결: 모델 오류 원장에 세 번의 계약 오류, 응답 SHA-256, finish reason,
  token usage, 최종 실패 원문을 보존한다. 심볼 프롬프트와 복구 지시에는
  불필요한 공백 반복을 금지하고 일곱 필드를 모두 쓴 직후 압축 JSON 객체를
  닫도록 명시했다. 정상 응답 여유는 1,600 tokens로 정했다.
- 결과: 기존 252건을 다시 계산하지 않고 SKY만 재시도해 253 done,
  3 excluded, error 0으로 종결했다.

### 9. WMT 섹터 동의어 과잉 차단

- 문제: WMT 종목 해설은 Consumer Defensive 연결을 설명했지만 검증기가
  `Consumer Defensive`와 `방어소비재`만 허용해 표준 한국어 동의어
  `필수소비재`를 실패로 처리했다.
- 해결: 판단이나 숫자를 바꾸지 않고 Consumer Defensive의 검증 가능한
  한국어 동의어에 `필수소비재`를 추가했다. 성공한 섹터·종목 배치는
  checkpoint로 보존하고 실패 배치를 다시 생성했다.
- 결과: WMT 포함 종목 배치, 최종 편집장 종합, 전체 품질 게이트가 통과했다.

### 10. 진행 중 상태에 과거 메일 DONE이 함께 보이던 문제

- 문제: 실제 앱이 새 Oracle 지문으로 재판단 중인데 `ai-radar status`가 같은
  날짜의 과거 리포트와 메일을 `DONE`으로 함께 표시할 수 있었다.
- 해결: 상태 출력도 현재 prepare binding의 Oracle source fingerprint,
  report SHA, email ledger fingerprint를 정확히 비교한다. 재판단 중에는
  `running / STALE_ORACLE_SOURCE / MISSING`, 완결 후에만
  `confirmed / CURRENT / DONE`을 표시한다.

## 2026-07-31 기준 인수 수치

| 항목 | 결과 |
|---|---:|
| 표준화 거래일 | 12 |
| 범위 | 2026-07-15~2026-07-30 |
| 7/30 active master | 11,444 |
| 7/30 최종 필수 집합 | 11,449 |
| 7/30 BAR | 11,072 |
| 7/30 NO_BAR | 373 |
| 7/30 격리 | 4 |
| 7/30 ERROR | 0 |
| 12일 전체 격리 | 12 |
| model-facing 비정상 OHLC | 0 |
| symbol-change projection | 5,417 |
| 명시적 no-op 제외 | 1 |
| warnings | 0 |

## 2026-07-31 실제 종단간 발행 증거

```text
Oracle source fingerprint:
572a6ee2cc645f3b254eed8cab850225980faf9f4120869b9b741d7f1680735f

AI Radar report SHA-256:
ac21306ac0528c847801976860dbdffa3d4d8c0f2dfaf61d9b72704a27f135fc

Queue: done=253, excluded=3, pending=0, running=0, error=0
Quality: 8개 항목 모두 10/10, green
Email: REVISION_DONE
Gmail message ID: 19fb7ed3ece976fa
Hermes Operations run ID: 4c18f934de6d414d8575793303445c17
Hermes receipt: PASS, exit_code=0
완료 KST: 2026-07-31 20:26:46
중복 확인 KST: 2026-07-31 20:27:57, generation_skipped=true
```

## 운영 확인 명령

```bash
cd /home/zooh/Documents/GitHub/quant

# Oracle·모델·메일 현재 상태
scripts/ai-radar status
scripts/ai-radar preflight

# 관리 앱과 KST 스케줄 검증
hermes apps show quant-ai-radar --json
hermes apps verify quant-ai-radar --json
hermes cron status

# 일일 정상 경로. 직접 Python 모듈 대신 앱 경로를 사용한다.
hermes apps run quant-ai-radar --json
```

핵심 evidence:

```text
STOCKDATA/QUANT_AI_RADAR/oracle/incremental/state/oracle_incremental_status.json
STOCKDATA/QUANT_AI_RADAR/oracle/incremental/state/daily_coverage/
STOCKDATA/QUANT_AI_RADAR/runs/<as_of_date>/run_state.json
STOCKDATA/QUANT_AI_RADAR/runs/<as_of_date>/market_report.json
STOCKDATA/QUANT_AI_RADAR/email/email_delivery_history.json
~/.hermes/operations/runs/quant-ai-radar/latest.json
~/.hermes/cron/jobs.json
```

장애 복구는 원본·기존 queue·기존 email ledger를 삭제하는 방식으로 하지
않는다. 같은 source는 checkpoint부터 재개하고, 다른 source는 revision
archive를 만든 뒤 새 queue로 진행한다.
