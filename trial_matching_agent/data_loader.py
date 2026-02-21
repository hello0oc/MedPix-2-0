"""Data loaders for the trial matching agent.

Loads:
- NSCLC patient cases from nsclc-dataset/nsclc_dataset.jsonl
- Pre-built Gemini trial profiles from nsclc-dataset/nsclc_trial_profiles.json
- TrialGPT corpus data (trial_info.json, queries.jsonl, qrels, retrieved_trials)
- Local BM25 search index for trial retrieval
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import REPO_ROOT, DATA_DIR, _matches_nsclc
from .schemas import (
    KeyFact,
    PatientProfile,
    TrialInfo,
)

# ── Shared text utilities (imported from existing benchmark code) ────────
sys.path.insert(0, str(REPO_ROOT))

try:
    from medgemma_benchmark.run_medgemma_benchmark import (
        build_ehr_text as _build_ehr_text,
        collect_image_paths,
        resolve_image_path,
        sanitize_text,
        truncate_text,
        load_jsonl,
    )
except ImportError:
    # Minimal fallback if benchmark module not importable
    def load_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def sanitize_text(value: Optional[str]) -> str:
        if not value:
            return ""
        return re.sub(r"\s+", " ", value).strip()

    def truncate_text(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "\u2026"

    def _build_ehr_text(record: Dict[str, Any]) -> str:
        pieces: List[str] = []
        for field in ("history", "exam", "findings"):
            val = sanitize_text(record.get(field))
            if val:
                label = {"history": "Clinical History",
                         "exam": "Physical Exam",
                         "findings": "Imaging Findings"}[field]
                pieces.append(f"{label}: {val}")
        return "\n".join(pieces).strip()

    def collect_image_paths(record: Dict[str, Any]) -> List[str]:
        paths: List[str] = []
        for image in record.get("images", []):
            raw_path = image.get("file_path")
            if raw_path:
                paths.append(str(raw_path))
        return paths

    def resolve_image_path(workspace_root: Path, raw_path: str) -> Optional[Path]:
        candidate = workspace_root / raw_path
        if candidate.exists():
            return candidate
        if raw_path.startswith("MedPix-2-0/"):
            fallback = workspace_root / raw_path.split("MedPix-2-0/", 1)[1]
            if fallback.exists():
                return fallback
        return None


# ── Public API ────────────────────────────────────────────────────────────

def build_ehr_text(record: Dict[str, Any], strip_findings: bool = True) -> str:
    """Build EHR text from a patient record.

    Parameters
    ----------
    record : dict
        A single patient case record.
    strip_findings : bool
        If True, exclude the ``findings`` field so the LLM must interpret
        images independently (matches medgemma_gui/app.py behaviour).
    """
    if strip_findings:
        record = {k: v for k, v in record.items() if k != "findings"}
    return _build_ehr_text(record)


def get_resolved_images(
    case: Dict[str, Any],
    workspace: Optional[Path] = None,
) -> List[Path]:
    """Resolve all image paths for a case. Returns only existing files."""
    ws = workspace or REPO_ROOT
    paths: List[Path] = []
    for raw in collect_image_paths(case):
        resolved = resolve_image_path(ws, raw)
        if resolved and resolved.exists():
            paths.append(resolved)
    return paths


# ═══════════════════════════════════════════════════════════════════════
# NSCLC Cases
# ═══════════════════════════════════════════════════════════════════════
NSCLC_DATASET_PATH = REPO_ROOT / "nsclc-dataset" / "nsclc_dataset.jsonl"
NSCLC_PROFILES_PATH = REPO_ROOT / "nsclc-dataset" / "nsclc_trial_profiles.json"


def load_nsclc_cases(
    path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load NSCLC patient cases from JSONL."""
    p = path or NSCLC_DATASET_PATH
    if not p.exists():
        raise FileNotFoundError(f"NSCLC dataset not found: {p}")
    return [c for c in load_jsonl(p) if c.get("uid")]


def load_prebuilt_profiles(
    path: Optional[Path] = None,
) -> Dict[str, PatientProfile]:
    """Load pre-built Gemini trial profiles keyed by topic_id."""
    p = path or NSCLC_PROFILES_PATH
    if not p.exists():
        return {}

    data = json.loads(p.read_text(encoding="utf-8"))
    profiles: Dict[str, PatientProfile] = {}

    for item in data.get("profiles", []):
        topic_id = item.get("topic_id", "")
        if not topic_id:
            continue
        profiles[topic_id] = PatientProfile.from_dict(item)

    return profiles


# ═══════════════════════════════════════════════════════════════════════
# TrialGPT Corpus Data
# ═══════════════════════════════════════════════════════════════════════

def load_trial_info(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Load the parsed ClinicalTrials.gov data from trial_info.json.

    Returns dict: {nct_id: trial_data_dict}
    """
    p = path or (DATA_DIR / "trial_info.json")
    if not p.exists():
        raise FileNotFoundError(
            f"trial_info.json not found: {p}\n"
            "Run: wget -O trial_matching_agent/data/trial_info.json "
            "https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trial_info.json"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def load_trialgpt_queries(
    corpus: str = "sigir",
    data_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Load patient queries from queries.jsonl.

    Returns dict: {patient_id: patient_text}
    """
    d = data_dir or DATA_DIR
    p = d / f"{corpus}_queries.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Queries file not found: {p}")

    queries: Dict[str, str] = {}
    for entry in load_jsonl(p):
        qid = entry.get("_id", "")
        text = entry.get("text", "")
        if qid and text:
            queries[qid] = text
    return queries


def load_trialgpt_qrels(
    corpus: str = "sigir",
    data_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, int]]:
    """Load ground truth relevance labels from qrels TSV.

    Returns dict: {patient_id: {nct_id: relevance_label}}
    where relevance_label is 0, 1, or 2.
    """
    d = data_dir or DATA_DIR
    p = d / f"{corpus}_qrels.tsv"
    if not p.exists():
        raise FileNotFoundError(f"Qrels file not found: {p}")

    qrels: Dict[str, Dict[str, int]] = {}
    with p.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            # TREC format: query_id  0  doc_id  relevance
            qid, _, doc_id, rel = row[0], row[1], row[2], row[3]
            try:
                rel_int = int(rel)
            except ValueError:
                continue
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = rel_int
    return qrels


def load_trialgpt_retrieved(
    corpus: str = "sigir",
    data_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load pre-retrieved trials with ground truth labels.

    Each entry has:
      patient_id, patient (text), and label keys "0", "1", "2"
      each mapping to a list of trial dicts with "NCTID" and criteria.
    """
    d = data_dir or DATA_DIR
    p = d / f"{corpus}_retrieved_trials.json"
    if not p.exists():
        raise FileNotFoundError(f"Retrieved trials not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_trialgpt_corpus(
    corpus: str = "sigir",
    data_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load the trial corpus JSONL (for BM25 indexing).

    Each entry has: _id, title, text, metadata (diseases_list, etc.)
    """
    d = data_dir or DATA_DIR
    p = d / f"{corpus}_corpus.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Corpus file not found: {p}")
    return load_jsonl(p)


# ═══════════════════════════════════════════════════════════════════════
# BM25 Search (lightweight, no GPU required)
# ═══════════════════════════════════════════════════════════════════════

def _tokenize_simple(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


class BM25Index:
    """Lightweight BM25 search over the TrialGPT trial corpus.

    Avoids the heavy ``rank_bm25`` dependency — implements Okapi BM25
    directly with the same weighting strategy as TrialGPT:
    3x title, 2x conditions, 1x text.
    """

    def __init__(self, trial_info: Dict[str, Dict[str, Any]]):
        self.nct_ids: List[str] = []
        self.doc_lengths: List[int] = []
        self.tf: List[Dict[str, int]] = []  # term freq per doc
        self.df: Dict[str, int] = {}  # doc freq per term
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        self._trial_info = trial_info

        self._build_index(trial_info)

    def _build_index(self, trial_info: Dict[str, Dict[str, Any]]) -> None:
        for nct_id, info in trial_info.items():
            # TrialGPT weighting: 3x title, 2x condition, 1x text
            title = info.get("brief_title", "")
            diseases = info.get("diseases_list", [])
            if isinstance(diseases, str):
                diseases = [diseases]
            text = info.get("brief_summary", "")

            tokens = (
                _tokenize_simple(title) * 3
                + sum((_tokenize_simple(d) * 2 for d in diseases), [])
                + _tokenize_simple(text)
            )

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            self.nct_ids.append(nct_id)
            self.tf.append(tf)
            self.doc_lengths.append(len(tokens))

            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

        self.n_docs = len(self.nct_ids)
        if self.n_docs > 0:
            self.avgdl = sum(self.doc_lengths) / self.n_docs

    def search(
        self,
        query: str,
        top_n: int = 50,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[Tuple[str, float]]:
        """Search for trials matching the query string.

        Returns list of (nct_id, score) sorted descending.
        """
        query_tokens = _tokenize_simple(query)
        scores: List[float] = [0.0] * self.n_docs

        import math
        for token in query_tokens:
            if token not in self.df:
                continue
            df = self.df[token]
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

            for i in range(self.n_docs):
                tf_val = self.tf[i].get(token, 0)
                if tf_val == 0:
                    continue
                dl = self.doc_lengths[i]
                numerator = tf_val * (k1 + 1)
                denominator = tf_val + k1 * (1 - b + b * dl / (self.avgdl + 1e-9))
                scores[i] += idf * numerator / denominator

        # Sort and return top N
        ranked = sorted(
            zip(self.nct_ids, scores),
            key=lambda x: -x[1],
        )
        return ranked[:top_n]

    def multi_condition_search(
        self,
        conditions: List[str],
        top_n: int = 50,
        k: int = 20,
    ) -> List[Tuple[str, float]]:
        """Reciprocal rank fusion across multiple condition queries.

        Mirrors TrialGPT's hybrid_fusion_retrieval approach (BM25 component).
        Each condition contributes 1/(rank+k) * 1/(condition_idx+1).
        """
        nctid2score: Dict[str, float] = {}

        for cond_idx, condition in enumerate(conditions):
            results = self.search(condition, top_n=top_n * 2)
            for rank, (nct_id, _bm25_score) in enumerate(results):
                if nct_id not in nctid2score:
                    nctid2score[nct_id] = 0.0
                nctid2score[nct_id] += (1.0 / (rank + k)) * (1.0 / (cond_idx + 1))

        ranked = sorted(nctid2score.items(), key=lambda x: -x[1])
        return ranked[:top_n]

    def get_trial_info(self, nct_id: str) -> Optional[TrialInfo]:
        """Retrieve full trial info for an NCT ID."""
        data = self._trial_info.get(nct_id)
        if not data:
            return None
        return TrialInfo.from_trialgpt(nct_id, data)


# ═══════════════════════════════════════════════════════════════════════
# NSCLC-specific trial filtering
# ═══════════════════════════════════════════════════════════════════════

def filter_nsclc_trials(
    trial_info: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Filter trial_info to NSCLC-related trials only."""
    nsclc_trials: Dict[str, Dict[str, Any]] = {}
    for nct_id, info in trial_info.items():
        diseases = info.get("diseases_list", [])
        if isinstance(diseases, str):
            diseases = [diseases]
        combined = " ".join(diseases) + " " + info.get("brief_title", "")
        if _matches_nsclc(combined):
            nsclc_trials[nct_id] = info
    return nsclc_trials


def filter_nsclc_qrels(
    qrels: Dict[str, Dict[str, int]],
    nsclc_trial_ids: set,
) -> Dict[str, Dict[str, int]]:
    """Filter qrels to only include NSCLC trial IDs."""
    filtered: Dict[str, Dict[str, int]] = {}
    for patient_id, trials in qrels.items():
        nsclc_matches = {
            tid: rel for tid, rel in trials.items()
            if tid in nsclc_trial_ids
        }
        if nsclc_matches:
            filtered[patient_id] = nsclc_matches
    return filtered
