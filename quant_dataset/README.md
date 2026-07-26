# Quant daily dataset pipeline (Layer 1 + Massive ETF Flow Phase 2)

This package builds the daily price backbone for a future FMP + Massive training
dataset. It is deliberately isolated from the existing quant runtime. The
default data root is:

```text
/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET
```

This is not the complete market dataset. It currently implements the Layer 1
price backbone plus one isolated Phase 2 source:

- Massive grouped U.S. stocks daily bars: one full-universe request per date.
- FMP stable historical EOD full bars: one request per symbol and date range.
- Massive ETF Global fund flows through endpoint id
  `partners_etf_fund_flows`, `GET /etf-global/v1/fund-flows`.

The current entitlement evidence is explicit in `preflight` and
`state/dataset_manifest.json`: FMP `/stable/eod-bulk` returned HTTP 402 and is
disabled. It is not silently retried or treated as available. If the FMP plan
changes, re-probe that entitlement before enabling it.

## Three storage layers

1. `raw/`: exact HTTP response bytes, gzip compressed, with an uncompressed
   SHA256 in a redacted metadata sidecar. Paths are content addressed and never
   overwritten. `capture_events` records every HTTP response event even when
   two responses have identical bytes.
2. `normalized/daily_observations.sqlite3`: source-specific normalized OHLCV.
   `daily_observation_versions` is append-only by
   `(source, symbol, trade_date, raw_artifact_id)`; `daily_observations` is the
   latest projection. FMP and Massive rows are never merged into a synthetic
   bar.
   ETF flow uses separate `etf_flow_versions` append-only records,
   `etf_flow_observations` revisions by `(provider,ticker,effective_date)`, and
   the `etf_flow_latest` ticker projection. Every version retains
   `effective_date`, `processed_date`, provider, `captured_at_utc`, raw/capture
   references, and explicit point-in-time fields.

Provider ticker case is preserved for Massive. Mixed-case tickers can identify
different securities (`TPC` and `TpC`, for example), so upper-casing Massive
symbols would silently collapse rows. FMP/user lookup symbols remain
upper-cased at the request boundary.
3. `training_packets/`: deterministic JSONL analysis inputs. Packets contain
   source rows, QC, raw checksums and capture times. They contain no future
   return, recommendation, class label, chain-of-thought, or fabricated expert
   answer.

FMP ETF constituent `acceptanceTime` is also date-only for training purposes.
Holdings and memberships become visible on the first positive-volume SPY/QQQ
session strictly after the provider acceptance date. Same-date inclusion is
forbidden because an after-close or timezone-ambiguous acceptance would leak
into that session.

Historical backfill is **not true point-in-time data**. Restricting a packet to
data proven available at or before `as_of_date` prevents direct future-row
leakage, but a
historical API response captured today may contain later corrections,
back-adjustments, ticker mapping changes, or survivorship bias. Use daily
capture events going forward for genuine as-observed history. Do not train a
point-in-time claim from a retrospective backfill without separate universe and
revision evidence.

## Authentication and preflight

Keys are read from process environment first, then
`~/.dgx-secrets/secrets.env`:

```text
FMP_API_KEY=...
MASSIVE_API_KEY=...
```

FMP authentication is sent in the `apikey` request header. Massive
authentication is sent as `Authorization: Bearer`. Keys are not placed in URL
query strings and authorization headers are not persisted in metadata.

```bash
cd /home/zooh/Documents/GitHub/quant
python3 -m quant_dataset preflight
```

Preflight performs no network request. It checks the data root, database,
credential presence, endpoint policy, and writes the manifest.

## Shared rate limits

Every HTTP attempt, including a retry, reserves a slot in a file-locked limiter
under `~/.cache/quant_dataset/rate_limits/`. This makes the limit shared across
concurrent dataset processes on the DGX instead of relying on a per-process
sleep. FMP is capped at 270 calls per rolling 60 seconds: 90% of the documented
300/minute limit, leaving headroom for other DGX research jobs using the same
credential. Massive REST is conservatively capped at two calls per second and
grouped full-market jobs remain serial.

## Collection and resume

```bash
# One day: Massive full universe plus selected FMP overlap universe
python3 -m quant_dataset capture-daily \
  --date 2026-07-10 \
  --symbols AAPL,MSFT,NVDA,QQQ

# Historical Layer 1 backfill
python3 -m quant_dataset backfill \
  --from 2020-01-02 \
  --to 2026-07-10 \
  --symbols AAPL,MSFT,NVDA,QQQ

# Massive grouped only does not require --symbols
python3 -m quant_dataset backfill \
  --source massive \
  --from 2026-07-01 \
  --to 2026-07-10
```

Jobs and work items are deterministic. FMP resumes at symbol/range granularity;
Massive resumes at date granularity. Completed checkpoints are skipped, failed
items remain retryable, and `jobs.contract_json` preserves the collection
contract and endpoint-registry version. Weekends are excluded from Massive
backfill scheduling; exchange holidays may still produce a valid empty grouped
response.

The HTTP client captures error bodies before raising, retries 429/5xx with a
bounded delay, honors `Retry-After`, and defaults to a 120-second timeout because
full-universe Massive grouped responses can be slow.

## Massive ETF fund-flow contract

This layer downloads ETF **fund flows**, NAV, and shares outstanding. It does
not treat fund flows as ETF holdings or constituent weights. The endpoint
contract was read from the live Spark STOCK registry:

```text
endpoint_id: partners_etf_fund_flows
method/path: GET /etf-global/v1/fund-flows
historical filters: processed_date.gte / processed_date.lte
ticker filter: composite_ticker.any_of
page limit: 1..5000
pagination: response next_url
auth: Authorization: Bearer header
```

Authorization is never put in a URL, manifest, checkpoint, or raw metadata.
Every page is first stored through the immutable `RawStore`; identical payloads
still create distinct `capture_events`. `etf_flow_runs` and
`etf_flow_run_pages` retain a sanitized next-page cursor so a failed invocation
resumes at the next unfinished page.

```bash
# Daily freshness capture: recent processed-date window ending at --date
python3 -m quant_dataset capture-etf-flows \
  --date 2026-07-14 \
  --lookback-days 7

# Optional filtered capture
python3 -m quant_dataset capture-etf-flows \
  --date 2026-07-14 \
  --tickers QQQ,SPY,IWM \
  --strict-freshness

# Historical endpoint-supported backfill; pagination resumes automatically
python3 -m quant_dataset backfill-etf-flows \
  --from 2020-01-01 \
  --to 2026-07-14
```

Daily capture compares both raw page hashes and normalized record-set hashes
with the preceding capture in the same series. It records `fresh`,
`unchanged_repeated_hash`, `stale_source_date`, `stale_repeated_hash`, or
`empty_requested_window`. `verify` treats a stale latest daily capture as a
freshness error. Historical backfill is labeled `historical_window_captured`
and is not falsely described as as-observed PIT history.

For packet date `D` and ticker `T`, Massive ETF Flow uses the fail-closed policy
`massive_etf_flow_us_sessions_v1`. A row is eligible only after the later of:

- the second observed U.S. equity trading session strictly after
  `effective_date`; and
- the first observed U.S. equity trading session strictly after
  `processed_date`.

This means a provider row marked effective and processed on the same day cannot
enter a D or D+1-session packet; its earliest use is D+2 sessions. A row first
processed on D+2 remains excluded until the next observed session because the
provider exposes only a date, not a verified intraday publication timestamp.
The U.S. session calendar is anchored to positive-volume SPY/QQQ daily rows;
isolated weekend rows from warrants, thin funds, or bad source dates cannot
advance the lag clock. If either future session is absent from this calendar,
the row is excluded rather than approximated with calendar days. Packets retain the raw
provider dates and expose the separate `training_available_session_date`,
policy id, and capture time. The 68 GB normalized source DB is not rewritten.

## Cross-source QC and verification

```bash
python3 -m quant_dataset verify --from 2026-07-01 --to 2026-07-10
```

Default QC tolerances are 0.5% or $0.02 for OHLC, 15% for volume, and 1% for
VWAP. A mismatch beyond five times the relevant tolerance is `fail`. Statuses:

- `pass`: both sources are structurally valid and within tolerances.
- `warn`: both are present with a soft discrepancy.
- `fail`: a hard cross-source discrepancy.
- `single_source`: only one source exists for that symbol/date.
- `invalid`: malformed OHLCV or range invariants.

`verify` re-hashes every raw gzip payload, validates metadata, scans sensitive
request fields generically for redaction, checks normalized OHLC invariants, and
recomputes QC. Source differences remain visible rather than being averaged
away.

## Deterministic unlabeled packets

```bash
python3 -m quant_dataset export-packets \
  --from 2026-07-01 \
  --to 2026-07-10 \
  --symbols AAPL,MSFT,NVDA,QQQ \
  --lookback-days 20
```

The exporter writes `quant.analysis_packet.v3`; v1/v2 packets predate the full
ETF Flow plus next-session constituent availability gates and must not be used
for model training. Running the export
twice against the same SQLite snapshot produces identical
bytes and SHA256. The default gate includes `pass`, `warn`, and
`single_source`; `fail` and `invalid` require an explicit override. These files
are evidence packets for later labeling or instruction construction, not SFT
answers by themselves.

## Remaining Phase 2 (not implemented)

The manifest registry reserves, but does not claim implementation for:

- universe and ticker reference history;
- dividends, splits, and ticker events;
- float, short interest, and short volume;
- news, SEC filing text, forms, and risk factors;
- market status, Treasury, and macro series;
- option contracts, previous-day option bars, and SMA/EMA/MACD/RSI.

Each Phase 2 category needs its own endpoint spec, entitlement probe, raw
dataset name, checkpoint granularity, normalization table, QC policy, and
packet inclusion rule. Until those collectors exist, Layer 1 must not be
described as the entire FMP/Massive training dataset.

## Tests

Tests use fake HTTP sessions and never call FMP or Massive:

```bash
PYTHONPYCACHEPREFIX=/tmp/quant_dataset_pycache \
python3 -m unittest tests.test_quant_dataset_pipeline
python3 -m unittest quant_dataset.tests.test_etf_flows
```
