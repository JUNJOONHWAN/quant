# Quant Forecast v2

SPY와 Nasdaq-100(QQQ) point-in-time 구성 종목에 대해 5/20 거래일의
종가 수익률, 최대 상승 여력(MFE), 최대 손실(MAE)을 직접 회귀한다.

운영 타이밍은 다음 하나로 고정한다.

- 신호/거래 세션: `T`
- 가격·기술·재무·ETF 구성 공시: `T-1`까지
- Massive ETF Global Flow: 유효 거래일 `T-2`, 단 `T` 장전 캡처 확인 필수
- `T-2`가 없으면 `T-3`로 묵시적 대체하지 않는다.

`price_benchmark_flow_t3`는 기존 계약의 성능을 비교하기 위한 워크포워드
대조군일 뿐 운영 모델의 데이터 폴백이 아니다.

## 실행

```bash
python3 -m training.quant_forecast_v2 build-panel \
  --start-date 2018-01-02 --live-signal-date 2026-08-27 --replace

python3 -m training.quant_forecast_v2 evaluate --replace

python3 -m training.quant_forecast_v2 finalize

python3 -m training.quant_forecast_v2 report
```

패널·모델·평가 JSON은 `STOCKDATA/QUANT_FORECAST/v2` 아래에, 사용자용 HTML은
`DGX_Outputs/STOCK/리포트/AI_RADAR_FORECAST_V2` 아래에 생성한다. 어떤 명령도
주문, 메일, Google Sheets, systemd timer, AI RADAR 서비스 또는 Qwen 8B를
시작하거나 변경하지 않는다.
