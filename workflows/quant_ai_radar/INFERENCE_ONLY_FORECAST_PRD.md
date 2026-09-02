# Quant AI Radar inference-only forecast PRD

## Goal

Use the accepted Qwen3-8B-FLOW LoRA as a point-in-time price--ETF Flow
pattern reader, use the sealed ten-year corpus as an auditable historical
analogue memory, and use the running Qwen 27B service only to synthesize a
probabilistic individual-security outlook.  The daily report remains an
8B-only description of the current learned market structure.

This change does not train, merge, quantize, or otherwise mutate any model
weights.

## Product paths

### Individual security

1. Assemble the same point-in-time Oracle packet used by the accepted LoRA.
2. Ask Qwen3-8B-FLOW for its learned current pattern and counter-evidence.
3. Search the historical SFT materialization for comparable task/regime
   examples and calculate their realised 5, 20, and 60-session outcomes from
   the read-only daily-observation ledger.
4. Give Qwen 27B only the 8B pattern, computed analogue distributions, and an
   available same-date daily market context.
5. Publish the 8B pattern, deterministic historical evidence, 27B outlook,
   counter-evidence, invalidation conditions, source dates, hashes, and sample
   counts.  No order is placed.

### Daily report

Qwen3-8B-FLOW explains only the current market regime, ETF/sector rotation,
breadth, concentration, price--Flow confirmation or divergence,
counter-evidence, and next confirmation condition.  Daily security cards do
not ask the 8B model for future returns or buy/sell classifications.

Every daily report is already stored by as-of date.  Those outputs form a
pattern ledger for later evaluation without changing the trained model.

## Historical analogue contract

- The source SFT materialization and price ledgers are opened read-only.
- A compact derived index may be rebuilt atomically under the AI Radar output
  root.  It is disposable and never overwrites source data.
- Primary neighbours must share the current task type and learned regime.
- Distance uses only point-in-time price, volatility, drawdown, liquidity,
  ETF Flow, constituent-flow breadth, and exposure fields.
- Repeated observations from one symbol are capped so one long-lived ticker
  cannot dominate the result.
- A realised outcome is usable only when its horizon end date is on or before
  the current analysis as-of date.  This is the hard walk-forward leakage gate.
- Adjusted close is preferred; close is used only when adjusted close is not
  available.  SPY-relative outcomes are reported separately.
- Python owns neighbour selection and all arithmetic.  Neither language model
  may invent sample counts, returns, probabilities, or source dates.

## Acceptance criteria

1. No training process runs and no model/release file is modified.
2. Oracle, SFT materialization, and daily-observation source databases remain
   byte-for-byte read-only.
3. The existing D+1/D+2 Flow availability and point-in-time packet rules are
   preserved.
4. Every analogue outcome has `outcome_end_date <= analysis_as_of_date`.
5. The same packet and source ledgers produce the same neighbour statistics
   and 27B request hash.
6. The individual CLI fails closed if the accepted 8B, analogue engine, or
   configured 27B endpoint fails; it does not substitute another model or
   hard-coded action prose.
7. Individual JSON/HTML exposes the 8B learned pattern, 5/20/60-session
   historical distributions, and the 27B probabilistic outlook as separate
   layers.
8. New daily reports contain learned current-pattern fields and no security
   `action_view` or future-return request.
9. Unit tests cover feature extraction, leakage cutoff, deterministic
   aggregation, 27B response validation, CLI wiring, and daily narrative
   validation.
10. A live same-date smoke test confirms both model identities, source dates,
    output hashes, and an HTML render before operational use is declared.

## Non-goals

- Automated trading, position sizing, or order execution.
- Treating historical similarity as proof that the future repeats.
- Replacing the Oracle database, ETF Flow ingestion, or the accepted LoRA.
- Asking Qwen 27B to scan raw SQLite rows or calculate return statistics inside
  a prompt.
