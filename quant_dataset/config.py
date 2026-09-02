"""Configuration and secret loading for the dataset pipeline."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


DEFAULT_DATA_ROOT = Path("/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET")
DEFAULT_SECRETS_PATH = Path("~/.dgx-secrets/secrets.env").expanduser()
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class CredentialSet:
    """Resolved credentials without exposing their values in representations."""

    fmp_api_key: Optional[str]
    massive_api_key: Optional[str]
    fmp_source: str = "missing"
    massive_source: str = "missing"

    def status(self) -> dict:
        return {
            "fmp": {"configured": bool(self.fmp_api_key), "source": self.fmp_source},
            "massive": {
                "configured": bool(self.massive_api_key),
                "source": self.massive_source,
            },
        }

    def __repr__(self) -> str:
        return (
            "CredentialSet(fmp_api_key=<redacted>, massive_api_key=<redacted>, "
            "fmp_source={!r}, massive_source={!r})"
        ).format(self.fmp_source, self.massive_source)


def _strip_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    # Treat a whitespace-prefixed hash as an inline comment. API keys containing
    # a literal hash with no preceding whitespace remain intact.
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    return value


def read_env_file(path: Path) -> dict:
    """Read a small dotenv-compatible file without importing python-dotenv."""

    values = {}
    try:
        text = path.expanduser().read_text(encoding="utf-8")
    except FileNotFoundError:
        return values
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or not _ENV_KEY.match(key):
            continue
        values[key] = _strip_env_value(raw_value)
    return values


def load_credentials(
    environ: Optional[Mapping[str, str]] = None,
    secrets_path: Path = DEFAULT_SECRETS_PATH,
) -> CredentialSet:
    """Resolve API keys, preferring the process environment over secrets.env."""

    env = dict(os.environ if environ is None else environ)
    file_values = read_env_file(Path(secrets_path))

    fmp_env = env.get("FMP_API_KEY", "").strip()
    massive_env = env.get("MASSIVE_API_KEY", "").strip()
    fmp_file = file_values.get("FMP_API_KEY", "").strip()
    massive_file = file_values.get("MASSIVE_API_KEY", "").strip()

    return CredentialSet(
        fmp_api_key=fmp_env or fmp_file or None,
        massive_api_key=massive_env or massive_file or None,
        fmp_source="environment" if fmp_env else ("secrets.env" if fmp_file else "missing"),
        massive_source=(
            "environment" if massive_env else ("secrets.env" if massive_file else "missing")
        ),
    )


def resolve_data_root(
    explicit: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Path:
    env = os.environ if environ is None else environ
    candidate = explicit or env.get("QUANT_DATASET_ROOT")
    return Path(candidate).expanduser() if candidate else DEFAULT_DATA_ROOT
