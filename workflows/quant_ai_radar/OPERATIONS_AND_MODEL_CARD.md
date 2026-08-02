# Quant AI Radar 모델 카드·학습 레시피·운영 계약

이 문서는 DGX Spark에서 학습 완료된 `qwen3-8b-quant-lora-v1`과 이를
매일 활용하는 Quant AI Radar의 단일 운영 기준서다. 학습 데이터의 역할,
전처리, 모델 레시피, 승인 결과, 일일 데이터 흐름, 선택 정책, 산출물,
복구 절차와 예시를 함께 기록한다.

## 1. 궁극적 목표

Quant AI Radar는 주문 프로그램이 아니다. 목표는 다음 세 층을 분리해
감사 가능한 시장·종목 분석을 만드는 것이다.

1. Python 정량 계층이 Oracle 공통 DB의 FMP·Massive point-in-time 사실과
   수치를 계산한다.
2. 학습된 Qwen3-8B LoRA가 그 사실만 해석해 가격과 ETF Flow의 확인,
   불일치, 반대 증거, 미확인 사항과 시장 국면을 구조화한다.
3. 개별 종목 요청에서는 Python이 과거 point-in-time 유사사례의 실제
   후행 성과를 계산하고 Qwen 27B가 8B 패턴과 그 분포를 종합해 확률적
   전망을 작성한다. 일일 리포트에는 이 27B 계층을 사용하지 않는다.
4. 리포트 계층이 전체 시장, 중요 ETF, ETF 자금 전달의 영향을 받는
   종목을 사용자가 검색·검토할 수 있게 제공한다.

모델이 해서는 안 되는 일:

- 수치 재계산 또는 Python이 준 사실 변경
- `as_of_date`보다 미래인 정보 사용
- 현재 상장 목록으로 과거 상폐 종목 제거
- 근거 없는 통화·자금 규모·뉴스·원인 생성
- 매수·매도·주문 지시 또는 실주문 연결
- 승인되지 않은 다른 모델이나 고정 문구 fallback 사용

## 2. 학습 데이터의 실제 범위

### 학습 입력의 세 가지 과제

| 과제 | 의미 | 학습 train 수 |
|---|---|---:|
| `etf_own_flow_analysis` | ETF 자체 가격·유동성·Flow·NAV·구성종목 해석 | 20,000 |
| `stock_constituent_flow_analysis` | ETF 자금이 당시 구성종목에 전달되는 효과 해석 | 25,000 |
| `all_stock_control_analysis` | ETF 관계가 없는 종목의 대조군·불충분 증거 해석 | 15,000 |

검증셋은 과제별 512개, 총 1,536개다. 동결 테스트는 2025년 이후
21,106개이며 학습 dataloader에서 사용하지 않았다.

### 시간 분할

- train: `2017-01-03` ~ `2023-11-30`
- validation: `2024-01-01` ~ `2024-12-02`
- frozen test: `2025-01-01` ~ `2026-07-14`
- 분할 경계마다 20거래일 embargo
- 무작위 시간 혼합 없음

### 데이터 원천

| 원천 | 사용 내용 | 공개·가용 시점 계약 |
|---|---|---|
| FMP | 원시 일봉 OHLCV, 과거 ETF 구성종목 | 일봉 거래일, 구성종목 acceptance 이후 첫 미국 거래세션 |
| Massive ETF Global | ETF fund flow, NAV, shares outstanding, assets | effective+2 미국 세션과 processed+1 세션 중 늦은 날 |
| FMP active master + legacy EOD + 종목별 EOD | Oracle의 FMP 장기 이력 이후 모든 활성 주식·ETF 일봉 | BAR/NO_BAR/QUARANTINED_INVALID_BAR/ERROR 전수 원장과 확정 거래일 |
| Oracle 파생 feature | 전체 ETF Flow·가격, ETF-구성종목 전달효과, 섹터 회전 | Oracle snapshot seal과 source fingerprint SHA256 |

FMP의 재무·뉴스·기관·내부자 등 확장 역사 백필은 현재 일시중지 상태이며
이 승인 모델의 SFT 입력에 포함되지 않았다. 이를 포함한 것처럼 리포트에
표현하면 안 된다.

## 3. 룩어헤드·생존편향 차단

### ETF Flow

`massive_etf_flow_us_sessions_v1` 정책을 사용한다.

```text
flow_visible_date =
  max(second_US_session_after(effective_date),
      first_US_session_after(processed_date))
```

`flow_visible_date <= as_of_date`인 Flow만 패킷에 들어간다. 세션 달력은
거래량이 존재하는 SPY/QQQ 거래일로 만들며 주말 잡음 행이 D+1/D+2
시계를 앞당길 수 없다.

### ETF 구성종목

FMP acceptance 날짜 다음의 첫 미국 거래세션부터만 보이도록 한다.
구성종목은 현재 보유목록이 아니라 해당 시점에 가용했던 snapshot을 쓴다.

주식 `s`, ETF `e`, 날짜 `t`의 자금 전달은 다음처럼 계산한다.

```text
eligible_membership(e,s,t) =
  next_US_session_after(acceptance_date) <= t

allocated_flow(e,s,t) =
  visible_fund_flow(e,t) * visible_holding_weight(e,s,t) / 100
```

같은 ETF snapshot 안에서 동일 종목이 여러 행이면 weight를 먼저 합산하고
Flow는 한 번만 적용한다.

### 가격·유동성

- `quant.analysis_packet.v3`만 허용
- 당일 가격과 양의 거래량 필수
- 모든 종목 최소 5개 가격 세션
- 20일 수익률을 위해 최대 21개 가격 세션 보존
- ETF는 최근 20세션 중 최소 10세션, 양의 거래량 비율 75% 이상,
  중앙값 달러 거래대금 USD 1백만 이상
- 조정주가의 절대 레벨은 사용하지 않고 원시 close 사용
- PIT 기업행위 근거 없이 하루 45% 이상 가격 단절이면 제외
- 현재 active ticker 목록을 과거 필터로 사용하지 않음
- Flow 정규화는 당시까지 보이는 trailing median/MAD와 AUM 비율만 사용

## 4. SFT 데이터 계약

- 패킷: `quant.analysis_packet.v3`
- 학습 행: `quant.sft.example.v1`
- 압축 계약: `quant.qwen3_8b_sft_contract.v2`
- train: 60,000행, 513,423,365 bytes
- validation: 1,536행, 12,914,368 bytes
- 미래 수익률: prompt와 target 모두 없음
- 중복: packet content와 example ID 기준 제거
- truncation: 금지, 4,096 token 초과 시 학습 전 fail closed
- assistant target만 loss 계산

모델 응답 필수 키:

```json
{
  "facts": {},
  "interpretation": {},
  "counter_evidence": [],
  "unknowns": [],
  "regime": "structured regime",
  "confidence": 0.0,
  "conclusion": "evidence-bounded interpretation"
}
```

정량 수치는 target을 만드는 결정론적 Python에서 계산했다. 따라서 현재
모델은 인간 펀드매니저의 자유서술을 모사한 모델이 아니라, point-in-time
시장 증거를 일관된 구조로 해석하도록 학습된 감사 가능한 baseline이다.

## 5. 학습 레시피

| 항목 | 값 |
|---|---|
| Base | `/home/zooh/models/Qwen3-8B-bf16` |
| Framework | NVIDIA NeMo AutoModel `26.06.00` |
| Precision | BF16 |
| Method | answer-only LoRA SFT |
| LoRA target | `*_proj` |
| rank / alpha | 8 / 32 |
| context | 4,096 |
| local / global batch | 1 / 8 |
| epochs | 2 |
| optimizer | Adam, fused |
| LR | `1e-5`, cosine to `1e-6` |
| topology | 1 process, FSDP2, TP=1, CP=1 |
| seed | 1111 |
| checkpoints | 500 steps |
| validation | 100 steps |
| thinking | disabled, prompt `/no_think` |

실제 설정 파일:

- `training/quant_llm/configs/qwen3_8b_lora_spark.yaml`
- `training/quant_llm/TRAINING_PLAN.md`
- `training/quant_llm/README.md`

학습 순서:

1. 전체 FMP symbol-session을 전수 스캔해 시간·과제별 salted hash
   reservoir 생성
2. 선택 pair의 완전한 v3 패킷을 원본 DB에서 재생성
3. PIT·유동성·스키마·중복·날짜 분할 검증
4. 과제별 quota로 train/validation view 고정
5. Qwen tokenizer로 전 행 4,096 token 무절단 검사
6. smoke 학습·checkpoint reload
7. full BF16 LoRA 학습
8. 최종 adapter를 vLLM에 올리고 21,106개 동결 테스트 전수 추론
9. 평가 GREEN인 경우에만 release manifest 생성

## 6. 승인된 모델과 평가

모델 ID: `qwen3-8b-quant-lora-v1`

동결 테스트:

- expected/matched/predicted: `21,106 / 21,106 / 21,106`
- JSON schema valid: `0.99729935`
- exact deterministic facts: `0.99748887`
- regime accuracy: `0.98810765`
- structured signal accuracy: `0.98076376`
- counter-evidence recall: `0.99511987`
- unknown recall: `0.99746518`
- 미래정보·거래지시 위반: `0`
- 결과: `green`

핵심 승인 해시:

```text
release manifest:
761b4fd3d0a51711a56fe8bbe3dba59d8d2444db0971103f823a7dea47fc38e8

adapter set:
2b6c1c96d3536ab475e56530166aa630e0138553ce4af86656633680f3cb50d6

adapter weights:
3f1a92fb63a7c04ed9a36ea91a08219f1f6ed31e3eca64ced2a3f69f3a607ab1

adapter config:
ab7ee5bd35a88bdf44dc0de864c774e99f8a0ee85181363d3bc24fd231f9e8fa

dataset manifest:
429911ab3f8cf6364e9dfa4c58f6955a5d911e64f47178771bf74be826335d29

frozen test:
63e41c123677cbde3a3a6519f1b44852f3fe283d5a58a8eaa4e6e43321979080
```

release manifest가 adapter, dataset, test, predictions, evaluation을 모두
해시로 묶는다. 런타임은 모델 호출 전에 이 해시를 다시 검증한다.

## 7. vLLM 배포 계약

현재 endpoint:

```text
http://127.0.0.1:8018/v1/chat/completions
model=qwen3-8b-quant-lora-v1
base=qwen3-8b-base
container=quant-qwen3-lora-vllm-8018
restart=unless-stopped
```

배포 고정값:

- `temperature=0`
- `enable_thinking=false`
- 다른 모델 fallback 금지
- 분석 전용 scope
- response contract 오류는 한 번만 사실 고정 repair
- repair 실패 시 hardcoded 결과를 만들지 않고 queue error

## 8. 매일 활용하는 최적 구조

10년치를 매일 모델 프롬프트에 다시 넣지 않는다. 장기 이력은 학습과
결정론적 feature 계산의 기반이며, 모델의 일일 입력은 짧은 당일 패킷이다.
개별 종목 분석만 분석 시점마다 파생 인덱스에서 유사사례를 검색해 완결된
후행 성과 분포를 계산한다. 이것은 추가 학습이 아니라 읽기 전용 검색이다.

```text
Oracle single writer
  -> immutable base + FMP 일봉 증분 + Massive ETF Flow 증분
  -> COMPLETE Oracle snapshot seal 및 SHA 검증
  -> read-only shared point-in-time overlay
  -> ETF/종목 관계 인덱스 증분 갱신
  -> 당일 전 종목 정량 스캔
  -> 중요 ETF/종목 동적 선택
  -> 선택 대상 v3 패킷 생성
  -> 승인 LoRA 구조화 판단
  -> 전체 시장 synthesis
  -> 시장/종목 리포트 + coverage ledger
  -> 420px Gmail 전송 + message-id 완료 proof
```

같은 시장 날짜의 Oracle source fingerprint가 바뀌면 기존 AI queue를
재사용하지 않는다. 이전 queue·리포트는 revision archive로 보존하고 새
fingerprint로 전 단계를 다시 계산한다. 동일 fingerprint 재시도는
checkpoint를 재사용하고 메일을 중복 발송하지 않지만, 실제 원장 개정은
정정 메일 한 번을 별도 proof로 남긴다. 세부 절차와 장애 이력은
`DAILY_ORACLE_AI_RADAR_RUNBOOK.md`를 따른다.

### 관계 인덱스

FMP 구성종목 2천만 행은 최초 1회만 distinct relation으로 변환한다.
이후 `rowid`/version `id` marker보다 큰 새 행만 반영한다.

인덱스는 다음과 결합되어야 소비할 수 있다.

- Oracle target `as_of_date`
- base/incremental DB 경로
- 모든 source marker
- shared source fingerprint
- index content SHA256

하나라도 맞지 않으면 추론을 시작하지 않는다.

### 전 종목 정량 스캔과 AI 선택

전체 ETF 관련 후보는 매일 정량적으로 검사한다. 그러나 모든 종목에
긴 생성 답변을 만드는 것은 느리고 정보가치가 낮다. 기본 AI budget은:

- ETF 최대 64개
- 구성종목 최대 192개
- 합계 최대 256개

이는 고정 종목 목록이나 매일 채워야 하는 quota가 아니라 처리량 ceiling이다.
선택 근거는 다음 Oracle 파생 증거를 매일 새로 점수화한다.

- integrated rotation cluster와 대표 ETF
- early accumulation
- Massive Flow fusion
- Flow/가격 불일치·outflow
- accumulation cluster
- ETF 관련 종목의 weight·반복 노출

모든 후보는 `coverage_ledger.jsonl`에 남는다.

- 선택: `selected_material_daily_evidence`
- 점수는 있으나 budget 아래: `capacity_ranked_below_daily_budget`
- 오늘 의미 있는 증거 없음: `no_material_oracle_evidence_today`
- 패킷 단계에서 탈락: queue의 PIT·유동성 exclusion reason

따라서 “AI가 답하지 않은 종목”도 스캔 누락이 아니다.

### 사용자 지정 종목

일일 64/192 ceiling은 자동 batch에만 적용한다. 사용자가 특정 ticker를
요청하면 선택 목록 밖이라도 `analyze_on_demand.py`가 같은 sealed Oracle
DB에서 승인 LoRA로 현재 패턴을 판독하고, 과거 유사사례의 완결된 후행
성과를 계산한 뒤 Qwen 27B가 확률적 전망을 종합한다. ETF 관계가 없으면
학습된 `all_stock_control_analysis`로 현재 패턴을 해석한다. 당일
가격·품질·PIT 증거, 유사사례, 또는 지정된 27B endpoint가 없으면 추측
답변이나 다른 모델 fallback 대신 fail-closed 사유를 저장한다.

일일 batch와 사용자 지정 분석은 corporate action에서도 같은 계약을
사용한다. Oracle은 Massive 발견본, FMP 교차확인본, 검증된 발행사·거래소
공지를 source-preserving version으로 저장한다. provider 행의
`available_date`는 최초 관측일이며 소급하지 않는다. 발행사·거래소 공지는
문서화된 공지일을 사용할 수 있다. 분석 기준일에는
`effective_date<=as_of`와 `available_date<=as_of`를 모두 만족하고,
공식 공지이거나 Massive와 FMP의 비율이 정확히 일치한 이벤트만 적용한다.
분할 효력일 이전의 가격은 `old_shares/new_shares`, 거래량은
`new_shares/old_shares`로 변환한 뒤에만 특징·학습 예시·추론 입력을 만든다.
따라서 현재 시점에 늦게 발견한 provider split을 과거 분석에 소급하는
lookahead는 허용되지 않는다.

```bash
PYTHONPATH=$PWD /usr/bin/python3 \
  -m workflows.quant_ai_radar.analyze_on_demand AAPL NVDA
```

## 9. 일일 산출물

```text
QUANT_AI_RADAR/
  shared/
    etf_relation_index.sqlite3
  runs/YYYY-MM-DD/
    universe_manifest.json
    candidates.jsonl
    selection_manifest.json
    selected_candidates.jsonl
    coverage_ledger.jsonl
    oracle_market_features.json
    selected_run_queue.sqlite3
    security_judgements.jsonl
    market_report.json
    run_state.json
  on_demand/YYYY-MM-DD/
    <SYMBOL>.json
    <SYMBOL>.html
    latest_request.json
  status/
    latest.json
    daily_cycle.json
```

사용자-facing 구조:

1. `market_report.json`: 시장 상태, 확인·반대 증거, 핵심 ETF·종목
2. `security_judgements.jsonl`: 선택된 ETF·종목의 완전한 AI 판단
3. `coverage_ledger.jsonl`: 전체 후보의 선택 여부·점수·사유
4. `market_report.html`: 시장 요약과 확인·반대 증거
5. `security_index.html`: ticker·regime·관계 검색
6. `security_reports/<SYMBOL>.html|json`: 선택 종목 상세 판단
7. `rendered_reports_manifest.json`: 모든 렌더 산출물 SHA256
8. `shadow_gate_audit.json`: queue/PIT/해시/비게시/처리시간 shadow 판정
9. `runtime_readiness_audit.json`: OOM/Xid/NVRM·vLLM·재부팅·서비스 판정

## 10. 입력·출력 예시

다음은 스키마 설명용 예시이며 특정 날짜의 실제 투자 판단이 아니다.

### ETF 입력 개념

```json
{
  "symbol": "AAA",
  "as_of_date": "2026-07-27",
  "price_sessions": "latest 21 visible sessions",
  "etf_flow": {
    "latest_effective_date": "2026-07-24",
    "visible_under_d_plus_1_d_plus_2_policy": true,
    "flow_to_assets_pct": 0.42,
    "robust_zscore": 2.3
  },
  "constituent_snapshot": {
    "effective_date": "2026-05-31",
    "available_date": "2026-07-07"
  },
  "quality": "pass"
}
```

### ETF 판단 예시

```json
{
  "facts": {
    "symbol": "AAA",
    "as_of_date": "2026-07-27"
  },
  "interpretation": {
    "price_signal": "positive",
    "etf_flow_signal": "positive",
    "relationship": "price_flow_positive_confirmation",
    "scope": "data_interpretation_not_trade_execution",
    "task_type": "etf_own_flow_analysis"
  },
  "counter_evidence": [
    "constituent breadth is weaker than aggregate ETF flow"
  ],
  "unknowns": [
    "same-day catalyst is not part of the approved training packet"
  ],
  "regime": "price_flow_positive_confirmation",
  "confidence": 0.73,
  "conclusion": "Visible price and ETF flow evidence agree, with breadth caveat."
}
```

### 구성종목 판단 예시

```json
{
  "facts": {
    "symbol": "AAPL",
    "as_of_date": "2026-07-27",
    "etf_flow_to_constituent": {
      "eligible_etf_count": 7,
      "net_weighted_flow_rate_contribution_pct": 0.001284
    }
  },
  "interpretation": {
    "price_signal": "mixed",
    "etf_flow_signal": "positive",
    "relationship": "positive_flow_without_price_confirmation",
    "scope": "data_interpretation_not_trade_execution",
    "task_type": "stock_constituent_flow_analysis"
  },
  "counter_evidence": [
    "same-session price did not confirm the weighted ETF inflow"
  ],
  "unknowns": [
    "non-ETF institution flow is outside this packet"
  ],
  "regime": "price_flow_divergence",
  "confidence": 0.61,
  "conclusion": "ETF transmission is positive but price confirmation is absent."
}
```

### 시장 synthesis 예시

```json
{
  "market_state": "rotation",
  "confidence": 0.68,
  "summary": "ETF Flow is concentrated in selected clusters rather than broad.",
  "confirmations": [
    {
      "evidence_id": "oracle.rotation_cluster.Software/Cyber",
      "interpretation": "The cluster has broad ETF participation."
    }
  ],
  "contradictions": [
    {
      "evidence_id": "aggregate.price_signal_counts",
      "interpretation": "Price confirmation is not uniform across selected stocks."
    }
  ],
  "unknowns": [
    "The approved packet does not contain a same-day news causal label."
  ],
  "leading_etfs": ["AAA"],
  "affected_stocks": ["AAPL"],
  "scope": "market_and_security_analysis_not_trade_execution"
}
```

## 11. 재개·실패·신선도 게이트

- queue는 SQLite WAL과 `synchronous=FULL` 사용
- 실행 중 재부팅되면 `running` 행을 `pending`으로 되돌림
- queue metadata는 as-of, source fingerprint, Oracle status hash,
  Oracle feature snapshot hash, model release hash, 선택 정책과 후보 수에 결합
- 다른 날짜·DB·release·model로 같은 queue를 재사용하면 실패
- model error는 원문을 보존하고 다음 실행에서 error row만 재시도
- pending/running/error가 하나라도 있으면 market report publish 금지
- stale constituent, stale Flow, Oracle missing session, row gate failure,
  artifact hash mismatch는 모두 fail closed

## 12. 처리량과 shadow 승인

과거 21,106개 전수 평가의 4-worker 실측은 약 176건/시간이었다. 따라서
10,885개 전체에 장문 답변을 만드는 방식은 약 62시간이 걸려 폐기했다.
기본 선택 상한 256개는 같은 보수적 속도에서 약 1.5시간이다.

자동 실행 활성화 전 필수:

1. 4/8/16 worker 동일 canary 비교
2. 선택 64/192의 실제 패킷 생성+추론 처리량 측정
3. 재시작 후 queue 이어받기 검증
4. frozen as-of 전체 selected shadow 1회
5. 연속 5회 daily shadow에서 stale/lookahead/hash/schema/queue error 0
6. 메모리·GPU OOM과 vLLM restart count 확인
7. 사용자 승인

shadow 전에는 timer를 enable하지 않는다.

전체 shadow는 `--shadow`로 실행한다. 이 모드는 모든 JSON/HTML을 만들고
queue를 완결하지만 `status/latest.json`을 갱신하지 않는다.

완료 후 실제 wall-clock 초를 넣어 fail-closed 감사를 실행한다.

```bash
PYTHONPATH=$PWD /usr/bin/python3 \
  -m workflows.quant_ai_radar.validate_shadow_run \
  --run-dir /home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/runs/YYYY-MM-DD \
  --elapsed-seconds <OBSERVED_SECONDS>
```

`shadow_gate_audit.json`은 queue 잔여/error, prompt/response hash 누락,
미래 날짜·거래 지시, Oracle source and feature hashes, HTML/JSON 렌더 해시,
`status/latest.json` 비게시, 운영창 내 종료 여부를 한 번에 판정한다.
한 번의 frozen-date PASS만으로 timer를 켜지 않는다. 서로 다른 실제
daily as-of 날짜에서 5회 연속 PASS와 사용자 승인이 필요하다.

각 shadow 직후에는 같은 실행창을 지정해 런타임 증거도 저장한다.

```bash
PYTHONPATH=$PWD /usr/bin/python3 \
  -m workflows.quant_ai_radar.validate_runtime_readiness \
  --run-dir /home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/runs/YYYY-MM-DD \
  --since "<shadow 시작 KST>"
```

`runtime_readiness_audit.json`은 Linux OOM-kill, NVIDIA Xid,
`NV_ERR_NO_MEMORY`, vLLM OOM/restart, Docker 부팅 복구, user linger,
Radar unit 비활성, FMP 역사 백필 중지를 서로 다른 상태로 기록한다.
vLLM이 정상이고 해시 검증이 통과한 상태의 graphics-context
`NV_ERR_NO_MEMORY`는 참고용 출력 자체를 무효화하지 않는다. 다만 해당
shadow는 timer 활성화용 연속 5회에는 포함하지 않는다.

## 13. 스케줄과 서비스

사용자 의도:

```text
KST: 화~토 14:30
Hermes 저장/실행값: 30 14 * * 2-6 (Asia/Seoul)
systemd 비상 폴백 UTC 저장값: Tue..Sat 05:30
reporting timezone: Asia/Seoul
```

서비스:

- `quant-ai-radar-daily.service`
- `quant-ai-radar-daily.timer`

기본 실행기는 Hermes Operations App Manager이며, 전체 일일 분석 뒤
품질 GREEN, 420px HTML, Gmail message id까지 모두 끝나야 PASS다. 같은
미국 기준일 재시도는 발송 원장으로 중복 메일을 막는다. Shadow/Smoke는
메일을 보내지 않는다. systemd timer는 비상 폴백으로
`disabled/inactive`를 유지한다. vLLM은
`quant-qwen3-lora-vllm-8018`의 `unless-stopped` 정책으로 복구한다.

FMP 역사 백필은 이 서비스가 절대 재개하지 않는다.

## 14. 가장 좋은 AI 활용법

이 모델의 최적 역할은 “숫자 계산기”도 “모든 종목에 문장을 쓰는 모델”도
아니다.

- Python: 전체 종목 전수 계산, PIT 필터, 정규화, 순위, 해시
- Oracle feature snapshot: ETF 전체 흐름·cluster·rotation 후보 생성
- 학습 LoRA: 중요한 당일 증거와 개별 종목의 현재 학습 패턴,
  확인/불일치/반대증거/unknown 해석
- historical analogue engine: 개별 종목과 같은 task/regime의 과거 사례
  검색, 완결된 후행 성과와 SPY 대비 분포 계산
- Qwen 27B: 개별 종목에서만 8B 패턴·유사사례 분포·현재 시장 문맥 종합
- 최종 market synthesis: 선택 결과와 전체 Oracle 시장 구조를 연결
- 사용자: 리포트를 참고해 실전 판단 여부를 결정

이 분업이 가장 빠르고, 룩어헤드를 막기 쉽고, 모델의 hallucination을
제한하며, 8B 모델의 해석 능력을 정보가치가 높은 곳에 집중한다.

## 15. 의사결정급 리포트 품질 계약

이번 품질 개선은 재학습이 아니라 프로그램 계층의 계약 강화다. 학습된
LoRA는 다음만 담당한다.

- 시장 상태 판단
- 확인/반대 근거 ID 선택
- 주요 ETF·영향 종목 선택
- 숫자를 다시 쓰지 않는 정성 요약

Python은 다음을 전담한다.

- 가격·Flow·ETF 구성종목의 PIT 사실과 가중 기여도
- 전체 breadth와 국면 집계
- Oracle rotation cluster·rotation 원본 값
- 리포트에 표시되는 모든 숫자와 비교
- 근거 ID 허용 목록, JSON 문법, 룩어헤드, 매매 지시, 해시 검증
- 현재 종목 전달용 Flow의 최대 10 calendar-day age, 절대
  flow-to-net-assets 100% plausibility, ETF 유동성 fail-closed gate

시장 synthesis는 현재 evidence catalog에서 만든 vLLM guided JSON Schema를
사용한다. 확인/반대 근거 ID는 enum 밖의 값을 생성할 수 없다. 모델이
근거 설명문을 임의로 다시 쓰지 않도록 최종 리포트에는 선택된 evidence
ID의 정확한 catalog 객체를 렌더링한다. 일일 자유문장에 숫자·날짜·미래
전망·가격과 Flow의 인과 과장·근거 없는 투자자 동기가 포함되면 문장을
삭제하거나 치환하지 않고 같은 8B에 계약 재응답을 요구한다. 비교 방향
역전은 semantic gate가 별도로 차단한다. 요약은 가격 폭, ETF 자금, 섹터
회전, 괴리·위험 중 최소
세 축을 실제 판단문으로 연결해야 하며, `한국어 제한`, `한국어로 해석`,
`근거를 제시하지 않음` 같은 메타 문구는 publish 전에 거부된다.

`quality_audit.json`의 다음 7개 점수는 각각 최소 8.0이어야 한다.

1. `data_integrity`
2. `numeric_faithfulness`
3. `flow_evidence_quality`
4. `security_analysis`
5. `market_structure`
6. `model_judgement_integration`
7. `report_usability`

하나라도 미달하면 reference publish는 fail closed다. 개별 종목 HTML은
근거 기반 결론, 가격 구조, ETF Flow 전달, 가격-Flow 관계, 학습 모델 해석,
확인/반대 증거, 다음 확인 조건을 모두 포함해야 한다.

CLI와 Hermes는 `action=ask`로 동일 앱을 호출한다. 명시 ticker 질문은
새 on-demand inference를 수행하고, 시장·후보·회전 질문은 Oracle 기준일과
일치하며 위 품질 감사가 green인 최신 AI Radar 리포트만 답변 근거로 쓴다.

## 16. 향후 확장 원칙

다음 데이터나 목표를 추가할 때 기존 release를 조용히 바꾸지 않는다.

- FMP 재무·뉴스·기관 데이터를 넣으려면 point-in-time available date가
  완성된 새 dataset version과 새 모델 release 필요
- 실제 미래 수익률은 SFT prompt/answer가 아니라 별도 sealed offline
  평가에만 사용
- expert human label을 추가하면 reviewer agreement와 별도 provenance 필요
- Nemotron 30B 등 비교 모델은 동일 frozen test·동일 schema로 별도 평가
- 실주문 연결은 이 분석 프로젝트의 범위 밖이며 별도 승인·리스크 시스템 필요

## 17. 개별 종목 inference-only 전망 계약

개별 종목의 최종 구조는 다음과 같다.

```text
sealed Oracle current packet
  -> Qwen3-8B-FLOW 현재 학습 패턴 판독
  -> 131,568개 SFT 후보 corpus의 동일 task/regime 유사사례 검색
  -> read-only 일봉 원장에서 5/20/60-session 실제 후행 성과 계산
  -> outcome_end_date <= analysis_as_of_date 누수 차단
  -> Qwen 27B가 패턴·성과 분포·동일 기준일 시장 문맥 종합
  -> 개별 JSON + 420px HTML
```

고정 수치와 계산은 Python이 소유한다. 유사사례는 한 종목당 최대 두 건으로
제한해 장수 종목이 표본을 독점하지 못하게 하며, adjusted close를 우선하고
SPY 초과성과를 별도 계산한다. Qwen 27B는 `매수 검토`, `보유 관찰`, `관망`,
`비중 축소 검토`, `회피` 중 하나를 직접 고르고 근거·반대 근거·무효화
조건을 작성한다. Python은 그 선택을 바꾸지 않는다.

27B 문장에 입력에 없는 숫자·날짜·가격 수준·내부 evidence ID·영문 상태
코드가 섞이면 같은 판단을 근거 안에서 다시 서술하도록 최대 세 번
재요청한다. 끝까지 계약을 통과하지 못하면 개별 분석 전체가 실패한다.
다른 모델이나 고정 문구 fallback은 없다. 상세 수용 기준은
`INFERENCE_ONLY_FORECAST_PRD.md`를 따른다.
