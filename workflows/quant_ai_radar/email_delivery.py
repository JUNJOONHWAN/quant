"""Fail-closed 420px Gmail delivery for accepted Quant AI Radar reports."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import html
import json
import os
import re
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR"
)
DEFAULT_STATE_DIR = DEFAULT_OUTPUT_ROOT / "email"
DEFAULT_GMAIL_OAUTH_FILE = (
    Path.home() / ".dgx-secrets/files/google/etfradar-gmail-oauth.json"
)
DEFAULT_GMAIL_RECIPIENT_FILE = (
    Path.home() / ".dgx-secrets/files/google/etfradar-gmail-recipient"
)
EMAIL_CONTRACT_VERSION = "v2"
QUALITY_SCHEMA_VERSION = "quant.ai_radar_quality_audit.v2"
MAX_GMAIL_INLINE_BYTES = 90_000


class EmailDeliveryError(RuntimeError):
    """A production report cannot satisfy the email completion contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmailDeliveryError(f"invalid JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EmailDeliveryError(f"JSON artifact must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _recipients(value: str) -> list[str]:
    return [
        item.strip()
        for item in value.replace(";", ",").split(",")
        if "@" in item
    ]


def _oauth_path(environ: Mapping[str, str]) -> Path:
    return Path(
        environ.get(
            "QUANT_AI_RADAR_GMAIL_OAUTH_FILE",
            str(DEFAULT_GMAIL_OAUTH_FILE),
        )
    ).expanduser()


def _recipient_path(environ: Mapping[str, str]) -> Path:
    return Path(
        environ.get(
            "QUANT_AI_RADAR_GMAIL_RECIPIENT_FILE",
            str(DEFAULT_GMAIL_RECIPIENT_FILE),
        )
    ).expanduser()


def _recipient_list(environ: Mapping[str, str]) -> list[str]:
    explicit = environ.get("QUANT_AI_RADAR_EMAIL_TO", "").strip()
    file_value = ""
    try:
        file_value = _recipient_path(environ).read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return _recipients(explicit or file_value)


def email_transport_status(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return secret-free Gmail configuration readiness."""

    values = os.environ if environ is None else environ
    oauth = _oauth_path(values)
    recipients = _recipient_list(values)
    missing: list[str] = []
    oauth_keys: set[str] = set()
    if oauth.is_file():
        try:
            payload = json.loads(oauth.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                oauth_keys = {
                    key
                    for key in ("client_id", "client_secret", "refresh_token")
                    if str(payload.get(key) or "").strip()
                }
        except (OSError, json.JSONDecodeError):
            pass
    if oauth_keys != {"client_id", "client_secret", "refresh_token"}:
        missing.append("QUANT_AI_RADAR_GMAIL_OAUTH_FILE")
    if not recipients:
        missing.append("QUANT_AI_RADAR_EMAIL_TO")
    return {
        "status": "confirmed" if not missing else "failed",
        "configured": not missing,
        "transport": "gmail_api" if not missing else "none",
        "recipient_count": len(recipients),
        "missing": missing,
    }


def validate_mobile_report(path: Path) -> dict[str, Any]:
    """Validate the exact HTML artifact that will be delivered."""

    resolved = path.expanduser().resolve()
    text = ""
    failures: list[str] = []
    if not resolved.is_file():
        failures.append("html_missing")
    else:
        try:
            text = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            failures.append(f"html_read_failed:{type(exc).__name__}")
    lowered = text.lower()
    size = resolved.stat().st_size if resolved.is_file() else 0
    checks = {
        "minimum_size": size >= 2500,
        "gmail_inline_size": size <= MAX_GMAIL_INLINE_BYTES,
        "doctype": text.lstrip().lower().startswith("<!doctype html"),
        "html_close": "</html>" in lowered,
        "viewport": "width=device-width" in lowered,
        "max_width_420": bool(
            re.search(r"max-width\s*:\s*420px", text, flags=re.IGNORECASE)
        ),
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "status": "DONE" if not failures else "FAILED",
        "complete": not failures,
        "contract": "quant-ai-radar-mobile420-html-v1",
        "html_path": str(resolved),
        "bytes": size,
        "sha256": _sha256(resolved) if resolved.is_file() else "",
        "checks": checks,
        "failures": failures,
    }


def _validate_quality(report: Mapping[str, Any]) -> dict[str, Any]:
    quality = report.get("quality_audit")
    quality = quality if isinstance(quality, Mapping) else {}
    scores = quality.get("scores")
    scores = scores if isinstance(scores, Mapping) else {}
    failed = [
        str(name)
        for name, value in scores.items()
        if not isinstance(value, (int, float)) or float(value) < 8.0
    ]
    complete = (
        quality.get("schema_version") == QUALITY_SCHEMA_VERSION
        and quality.get("status") == "green"
        and quality.get("publishable_reference_report") is True
        and bool(scores)
        and not failed
        and report.get("selected_model_scope_complete") is True
        and report.get("deployment_mode") == "reference_publish"
    )
    return {
        "status": "green" if complete else "failed",
        "complete": complete,
        "failed_scores": failed,
        "scores": dict(scores),
    }


def _ledger_path(state_dir: Path) -> Path:
    return state_dir / "email_delivery_history.json"


def _read_ledger(state_dir: Path) -> dict[str, Any]:
    try:
        value = json.loads(_ledger_path(state_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {"sent": {}}
    if not isinstance(value, dict):
        value = {"sent": {}}
    if not isinstance(value.get("sent"), dict):
        value["sent"] = {}
    return value


def _dedupe_key(as_of_date: str) -> str:
    return (
        f"quant-ai-radar-daily:{as_of_date}:"
        f"{EMAIL_CONTRACT_VERSION}"
    )


def email_delivery_status(
    as_of_date: str,
    *,
    state_dir: Path = DEFAULT_STATE_DIR,
) -> dict[str, Any]:
    key = _dedupe_key(as_of_date)
    record = (_read_ledger(state_dir).get("sent") or {}).get(key)
    record = record if isinstance(record, Mapping) else {}
    message_id = str(record.get("message_id") or "")
    complete = bool(message_id)
    return {
        "status": "DONE" if complete else "MISSING",
        "complete": complete,
        "contract": "quant-ai-radar-daily-email-v2",
        "dedupe_key": key,
        "message_id": message_id,
        "recipient_count": int(record.get("recipient_count") or 0),
        "state_path": str(_ledger_path(state_dir)),
    }


def _gmail_access_token(path: Path) -> str:
    payload = _read_json(path)
    credentials = {
        key: str(payload.get(key) or "").strip()
        for key in ("client_id", "client_secret", "refresh_token")
    }
    if not all(credentials.values()):
        raise EmailDeliveryError("Gmail OAuth file is incomplete")
    body = urlencode(
        {**credentials, "grant_type": "refresh_token"}
    ).encode("utf-8")
    request = Request(
        "https://oauth2.googleapis.com/token",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urlopen(request, timeout=30) as response:
        token_payload = json.loads(response.read().decode("utf-8"))
    token = str(token_payload.get("access_token") or "")
    if not token:
        raise EmailDeliveryError("Google OAuth refresh returned no access token")
    return token


def _send_gmail_api(message: EmailMessage, oauth_file: Path) -> str:
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
    request = Request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
        data=json.dumps({"raw": raw}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_gmail_access_token(oauth_file)}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload.get("id") or "")


def _build_message(
    *,
    subject: str,
    html_body: str,
    plain_body: str,
    recipients: list[str],
    attachment: Path,
) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject
    message["To"] = ", ".join(recipients)
    message.set_content(plain_body)
    message.add_alternative(html_body, subtype="html")
    message.add_attachment(
        attachment.read_bytes(),
        maintype="text",
        subtype="html",
        filename=attachment.name,
    )
    return message


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def _candidate_rows(rows: Any) -> str:
    values = rows if isinstance(rows, list) else []
    body = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            _e(row.get("symbol")),
            _e(row.get("regime")),
            _e(_number(row.get("confidence"), 2)),
        )
        for row in values[:6]
        if isinstance(row, Mapping)
    )
    return body or '<tr><td colspan="3">해당 후보 없음</td></tr>'


def _number(value: Any, digits: int = 1) -> Any:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return value


def _email_html(report: Mapping[str, Any]) -> str:
    market = report.get("market_judgement")
    market = market if isinstance(market, Mapping) else {}
    dashboard = report.get("market_dashboard")
    dashboard = dashboard if isinstance(dashboard, Mapping) else {}
    breadth = dashboard.get("breadth")
    breadth = breadth if isinstance(breadth, Mapping) else {}
    lanes = dashboard.get("candidate_lanes")
    lanes = lanes if isinstance(lanes, Mapping) else {}
    selection = report.get("selection")
    selection = selection if isinstance(selection, Mapping) else {}
    quality = report.get("quality_audit")
    quality = quality if isinstance(quality, Mapping) else {}
    source = report.get("source_status")
    source = source if isinstance(source, Mapping) else {}
    oracle = source.get("shared_oracle_store")
    oracle = oracle if isinstance(oracle, Mapping) else {}
    rotations = dashboard.get("rotation_clusters")
    rotations = rotations if isinstance(rotations, list) else []
    rotation_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            _e(row.get("cluster")),
            _e(row.get("state")),
            _e(_number(row.get("score"))),
        )
        for row in rotations[:8]
        if isinstance(row, Mapping)
    ) or '<tr><td colspan="3">회전 cluster 없음</td></tr>'
    score_pills = "".join(
        '<span class="pill">{} {}/10</span>'.format(_e(name), _e(score))
        for name, score in (quality.get("scores") or {}).items()
    )
    positive = list(lanes.get("positive_confirmation_etfs") or []) + list(
        lanes.get("positive_confirmation_stocks") or []
    )
    negative = list(lanes.get("negative_confirmation_etfs") or []) + list(
        lanes.get("negative_confirmation_stocks") or []
    )
    divergence = list(lanes.get("divergence_etfs") or []) + list(
        lanes.get("divergence_stocks") or []
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quant AI Radar Daily</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#0b1020;color:#eef3ff;
font:15px/1.5 Arial,"Noto Sans KR",sans-serif}}main{{width:100%;max-width:420px;
margin:0 auto;padding:12px}}.card{{background:#131b31;border:1px solid #26324f;
border-radius:14px;padding:14px;margin:10px 0}}h1,h2,h3{{line-height:1.2}}
.meta,.muted{{color:#9ba9c8}}.metric{{font-size:24px;font-weight:700}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}.mini{{background:#0d1528;
border-radius:10px;padding:10px}}table{{width:100%;table-layout:fixed;
border-collapse:collapse}}th,td{{padding:7px;border-bottom:1px solid #26324f;
text-align:left;overflow-wrap:anywhere}}th{{color:#9ba9c8}}.pill{{display:inline-block;
padding:3px 7px;border:1px solid #26324f;border-radius:999px;margin:2px;
color:#72e0bd}}@media(max-width:420px){{main{{padding:10px}}}}
</style></head><body><main>
<h1>Quant AI Radar</h1>
<p class="meta">미국 기준일 {_e(report.get("as_of_date"))} · 생성
{_e(report.get("generated_at_kst"))}<br>참고용 분석 · 실주문 미연결</p>
<section class="card"><div class="muted">Market state</div>
<div class="metric">{_e(market.get("market_state"))}</div>
<p>{_e(market.get("summary"))}</p></section>
<section class="card"><h2>시장 구조</h2><div class="grid">
<div class="mini"><div class="muted">Confidence</div><div class="metric">{_e(market.get("confidence"))}</div></div>
<div class="mini"><div class="muted">AI 완료</div><div class="metric">{_e(quality.get("security_report_count", selection.get("selected_count")))}</div></div>
<div class="mini"><div class="muted">가격 양수</div><div class="metric">{_e(breadth.get("price_positive_pct"))}%</div></div>
<div class="mini"><div class="muted">Flow 양수</div><div class="metric">{_e(breadth.get("etf_flow_positive_pct"))}%</div></div>
</div></section>
<section class="card"><h2>섹터·테마 회전</h2><table><thead><tr>
<th>Cluster</th><th>State</th><th>Score</th></tr></thead>
<tbody>{rotation_rows}</tbody></table></section>
<section class="card"><h2>강세 확인 관찰</h2><table><thead><tr>
<th>Symbol</th><th>Regime</th><th>신뢰도</th></tr></thead>
<tbody>{_candidate_rows(positive)}</tbody></table></section>
<section class="card"><h2>약세 확인 위험</h2><table><thead><tr>
<th>Symbol</th><th>Regime</th><th>신뢰도</th></tr></thead>
<tbody>{_candidate_rows(negative)}</tbody></table></section>
<section class="card"><h2>가격–Flow 괴리</h2><table><thead><tr>
<th>Symbol</th><th>Regime</th><th>신뢰도</th></tr></thead>
<tbody>{_candidate_rows(divergence)}</tbody></table></section>
<section class="card"><h2>품질·신선도</h2><p>{score_pills}</p>
<p class="muted">Oracle 기준일 {_e(oracle.get("target_as_of_date"))} ·
Flow 최신 유효일 {_e(oracle.get("latest_flow_effective_date"))}<br>
전체 상세 근거와 종목 링크는 첨부 HTML에 포함됩니다.</p></section>
</main></body></html>"""


def render_email_html(report: Mapping[str, Any], path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(f"{resolved.suffix}.tmp")
    temporary.write_text(_email_html(report), encoding="utf-8")
    os.replace(temporary, resolved)
    return resolved


def deliver_daily_report(
    report_path: Path,
    *,
    state_dir: Path = DEFAULT_STATE_DIR,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Deliver one accepted report per US as-of date and persist Gmail proof."""

    values = os.environ if environ is None else environ
    resolved_report = report_path.expanduser().resolve()
    report = _read_json(resolved_report)
    as_of_date = str(report.get("as_of_date") or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of_date):
        raise EmailDeliveryError("report as_of_date is missing or invalid")
    if resolved_report.parent.name != as_of_date:
        raise EmailDeliveryError(
            "report directory does not match as_of_date: "
            f"{resolved_report.parent.name} != {as_of_date}"
        )
    quality = _validate_quality(report)
    if not quality["complete"]:
        raise EmailDeliveryError(
            "report quality gate is not publishable: "
            f"{json.dumps(quality, ensure_ascii=False, sort_keys=True)}"
        )
    attachment_path = resolved_report.with_name("market_report.html")
    if not attachment_path.is_file():
        raise EmailDeliveryError(
            f"full market report attachment is missing: {attachment_path}"
        )
    html_path = render_email_html(
        report,
        resolved_report.with_name("market_report_email_420.html"),
    )
    mobile = validate_mobile_report(html_path)
    if not mobile["complete"]:
        raise EmailDeliveryError(
            "420px HTML contract failed: "
            f"{json.dumps(mobile, ensure_ascii=False, sort_keys=True)}"
        )
    transport = email_transport_status(values)
    if not transport["configured"]:
        raise EmailDeliveryError(
            "Gmail transport is not configured: "
            f"{transport['missing']}"
        )

    state_dir = state_dir.expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "email_delivery.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = email_delivery_status(as_of_date, state_dir=state_dir)
        if existing["complete"]:
            return {
                **existing,
                "send_status": "SKIP_ALREADY_SENT",
                "html_contract": mobile,
                "quality_contract": quality,
                "transport": "gmail_api",
            }

        market = report.get("market_judgement")
        market = market if isinstance(market, Mapping) else {}
        selection = report.get("selection")
        selection = selection if isinstance(selection, Mapping) else {}
        market_state = str(market.get("market_state") or "UNKNOWN")
        selected_count = int(
            (report.get("quality_audit") or {}).get(
                "security_report_count",
                selection.get("selected_count"),
            )
            or 0
        )
        subject = (
            f"[AI Radar] {as_of_date} · {market_state} · "
            f"AI 분석 {selected_count}건"
        )
        plain_body = (
            "Quant AI Radar 일일 리포트\n"
            f"미국 시장 기준일: {as_of_date}\n"
            f"시장 상태: {market_state}\n"
            f"AI 상세 분석: {selected_count}건\n"
            "분석 전용이며 실주문과 연결되지 않습니다.\n"
        )
        recipients = _recipient_list(values)
        message = _build_message(
            subject=subject,
            html_body=html_path.read_text(encoding="utf-8"),
            plain_body=plain_body,
            recipients=recipients,
            attachment=attachment_path,
        )
        message_id = _send_gmail_api(message, _oauth_path(values))
        if not message_id:
            raise EmailDeliveryError("Gmail API returned no message id")

        ledger = _read_ledger(state_dir)
        ledger["sent"][_dedupe_key(as_of_date)] = {
            "message_id": message_id,
            "recipient_count": len(recipients),
            "subject": subject,
            "as_of_date": as_of_date,
            "report_path": str(resolved_report),
            "report_sha256": _sha256(resolved_report),
            "html_path": str(html_path),
            "html_sha256": mobile["sha256"],
            "attachment_path": str(attachment_path),
            "attachment_sha256": _sha256(attachment_path),
            "sent_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(_ledger_path(state_dir), ledger)
        return {
            **email_delivery_status(as_of_date, state_dir=state_dir),
            "send_status": "DONE",
            "html_contract": mobile,
            "quality_contract": quality,
            "transport": "gmail_api",
        }
