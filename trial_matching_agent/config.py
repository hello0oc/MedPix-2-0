"""Configuration for the trial matching agent.

Credential resolution mirrors the pattern from medgemma_gui/app.py:
  .streamlit/secrets.toml  →  environment variables
"""
from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# ── Model identifiers ────────────────────────────────────────────────────
GEMINI_MODEL_ID = "gemini-2.5-pro"
MEDGEMMA_MODEL_ID = "google/medgemma-1-5-4b-it-hae"
MEDGEMMA_ENDPOINT_URL = (
    "https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud"
)

# ── Credential key candidates (checked in order) ─────────────────────────
GEMINI_CANDIDATE_KEYS: Tuple[str, ...] = (
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY",
)
HF_CANDIDATE_KEYS: Tuple[str, ...] = (
    "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_API_KEY",
)

# ── Gemini thinking-model budgets ─────────────────────────────────────────
GEMINI_THINKING_BUDGET = 2048
GEMINI_MIN_OUTPUT_TOKENS = 1024

# ── NSCLC keyword patterns (shared with build_nsclc_dataset.py) ──────────
import re

NSCLC_PATTERNS = [
    re.compile(r"non.?small.?cell", re.IGNORECASE),
    re.compile(r"\bnsclc\b", re.IGNORECASE),
    re.compile(r"adenocarcinoma\s+of\s+(the\s+)?lung", re.IGNORECASE),
    re.compile(r"lung\s+adenocarcinoma", re.IGNORECASE),
    re.compile(r"squamous\s+cell\s+carcinoma\s+of\s+(the\s+)?lung", re.IGNORECASE),
    re.compile(r"lung\s+squamous\s+cell", re.IGNORECASE),
    re.compile(r"large\s+cell\s+carcinoma", re.IGNORECASE),
    re.compile(r"large.cell\s+lung", re.IGNORECASE),
]


def _matches_nsclc(text: str) -> bool:
    """Return True if *text* contains an NSCLC-related term."""
    return any(pat.search(text) for pat in NSCLC_PATTERNS)


# ── Credential resolution ────────────────────────────────────────────────
def _load_secrets() -> dict:
    """Load .streamlit/secrets.toml if it exists."""
    secrets_path = REPO_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return {}
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            data: dict = {}
            for line in secrets_path.read_text("utf-8").splitlines():
                m = re.match(r'^\s*(\w+)\s*=\s*["\'](.+?)["\']\s*$', line)
                if m:
                    data[m.group(1)] = m.group(2)
            return data
    with secrets_path.open("rb") as fh:
        return tomllib.load(fh)


_SECRETS_CACHE: Optional[dict] = None


def resolve_credential(candidates: Tuple[str, ...]) -> Optional[str]:
    """Try *candidates* against secrets.toml, then env vars. Return first hit."""
    global _SECRETS_CACHE
    if _SECRETS_CACHE is None:
        _SECRETS_CACHE = _load_secrets()
    for key in candidates:
        val = _SECRETS_CACHE.get(key) or os.environ.get(key)
        if val:
            return val
    return None


# ── Main configuration dataclass ─────────────────────────────────────────
@dataclass
class AgentConfig:
    """All tuneable knobs for the trial matching agent."""

    # API credentials (resolved lazily if not set)
    gemini_api_key: Optional[str] = None
    hf_token: Optional[str] = None

    # Model settings
    gemini_model: str = GEMINI_MODEL_ID
    medgemma_model: str = MEDGEMMA_MODEL_ID
    medgemma_endpoint: str = MEDGEMMA_ENDPOINT_URL

    # Token budgets
    gemini_max_output_tokens: int = 4096
    gemini_thinking_budget: int = GEMINI_THINKING_BUDGET
    medgemma_max_tokens: int = 1024

    # Timeouts and retries
    timeout_sec: int = 180
    retries: int = 3

    # Agent loop
    max_tool_rounds: int = 15

    # Data
    corpus: str = "sigir"
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR

    # Retrieval
    bm25_top_n: int = 50
    max_candidate_trials: int = 10

    # EHR processing
    ehr_char_limit: int = 3000

    def resolve_keys(self) -> "AgentConfig":
        """Populate API keys from secrets/env if not already set."""
        if not self.gemini_api_key:
            self.gemini_api_key = resolve_credential(GEMINI_CANDIDATE_KEYS)
        if not self.hf_token:
            self.hf_token = resolve_credential(HF_CANDIDATE_KEYS)
        return self

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create a fully-resolved config from environment / secrets."""
        return cls().resolve_keys()
