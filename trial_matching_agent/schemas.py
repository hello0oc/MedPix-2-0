"""Pydantic data models for the trial matching agent.

Aligned with both TrialGPT data formats (for ground-truth evaluation)
and the existing trial-profile schema from medgemma_gui/app.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# ──────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────
class InclusionLabel(str, Enum):
    INCLUDED = "included"
    NOT_INCLUDED = "not included"
    NOT_APPLICABLE = "not applicable"
    NOT_ENOUGH_INFO = "not enough information"


class ExclusionLabel(str, Enum):
    EXCLUDED = "excluded"
    NOT_EXCLUDED = "not excluded"
    NOT_APPLICABLE = "not applicable"
    NOT_ENOUGH_INFO = "not enough information"


class RelevanceLevel(int, Enum):
    """TrialGPT 3-level relevance scale from qrels."""
    NOT_RELEVANT = 0
    PARTIALLY_ELIGIBLE = 1
    ELIGIBLE = 2


class FeedbackPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ──────────────────────────────────────────────────────────────────────────
# Trial information  (mirrors trial_info.json from TrialGPT)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class TrialInfo:
    """A clinical trial with eligibility criteria.

    Schema follows TrialGPT's ``trial_info.json``:
      nct_id, brief_title, phase, drugs_list, diseases_list,
      enrollment, inclusion_criteria, exclusion_criteria, brief_summary
    """
    nct_id: str
    brief_title: str = ""
    phase: str = ""
    drugs_list: List[str] = field(default_factory=list)
    diseases_list: List[str] = field(default_factory=list)
    enrollment: str = ""
    inclusion_criteria: str = ""
    exclusion_criteria: str = ""
    brief_summary: str = ""

    # Original raw string fields (sometimes present in retrieved_trials.json)
    drugs: str = ""
    diseases: str = ""

    @classmethod
    def from_trialgpt(cls, nct_id: str, data: Dict[str, Any]) -> "TrialInfo":
        """Build from a trial_info.json entry or retrieved_trials item."""
        dl = data.get("drugs_list", [])
        if isinstance(dl, str):
            # Sometimes stored as repr string: "['a', 'b']"
            try:
                import ast
                dl = ast.literal_eval(dl)
            except Exception:
                dl = [dl]
        dsl = data.get("diseases_list", [])
        if isinstance(dsl, str):
            try:
                import ast
                dsl = ast.literal_eval(dsl)
            except Exception:
                dsl = [dsl]
        return cls(
            nct_id=data.get("NCTID", nct_id),
            brief_title=data.get("brief_title", ""),
            phase=data.get("phase", ""),
            drugs_list=dl,
            diseases_list=dsl,
            enrollment=str(data.get("enrollment", "")),
            inclusion_criteria=data.get("inclusion_criteria", ""),
            exclusion_criteria=data.get("exclusion_criteria", ""),
            brief_summary=data.get("brief_summary", ""),
            drugs=data.get("drugs", ""),
            diseases=data.get("diseases", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nct_id": self.nct_id,
            "brief_title": self.brief_title,
            "phase": self.phase,
            "drugs_list": self.drugs_list,
            "diseases_list": self.diseases_list,
            "enrollment": self.enrollment,
            "inclusion_criteria": self.inclusion_criteria,
            "exclusion_criteria": self.exclusion_criteria,
            "brief_summary": self.brief_summary,
        }


# ──────────────────────────────────────────────────────────────────────────
# Patient profile  (reuses existing trial-profile schema)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class KeyFact:
    field: str
    value: Any = None
    evidence_span: Optional[str] = None
    required: bool = False
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "evidence_span": self.evidence_span,
            "required": self.required,
            "notes": self.notes,
        }


@dataclass
class PatientProfile:
    topic_id: str
    profile_text: str = ""
    key_facts: List[KeyFact] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)

    def get_fact(self, field_name: str) -> Optional[KeyFact]:
        for kf in self.key_facts:
            if kf.field == field_name:
                return kf
        return None

    @property
    def missing_info(self) -> List[str]:
        mi = self.get_fact("missing_info")
        if mi and isinstance(mi.value, list):
            return mi.value
        return []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "profile_text": self.profile_text,
            "key_facts": [kf.to_dict() for kf in self.key_facts],
            "ambiguities": self.ambiguities,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatientProfile":
        kfs = [
            KeyFact(**kf) if isinstance(kf, dict) else kf
            for kf in data.get("key_facts", [])
        ]
        return cls(
            topic_id=data.get("topic_id", ""),
            profile_text=data.get("profile_text", ""),
            key_facts=kfs,
            ambiguities=data.get("ambiguities", []),
        )


# ──────────────────────────────────────────────────────────────────────────
# Criterion-level matching  (TrialGPT format)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class CriterionMatch:
    """One criterion's assessment — mirrors TrialGPT's 3-element list.

    TrialGPT format:  [reasoning, [sentence_ids], label]
    We add:  imaging_relevant (bool) — whether the criterion could benefit
    from direct image review by MedGemma.
    """
    criterion_idx: str
    reasoning: str = ""
    evidence_sentence_ids: List[int] = field(default_factory=list)
    label: str = ""  # e.g. "included", "excluded", ...
    imaging_relevant: bool = False

    def to_trialgpt_format(self) -> list:
        """Return [reasoning, sentence_ids, label] for TrialGPT scoring."""
        return [self.reasoning, self.evidence_sentence_ids, self.label]


@dataclass
class MatchingResult:
    """Criterion-level matching output for one patient–trial pair."""
    inclusion: Dict[str, CriterionMatch] = field(default_factory=dict)
    exclusion: Dict[str, CriterionMatch] = field(default_factory=dict)

    def to_trialgpt_format(self) -> Dict[str, Dict[str, list]]:
        """Convert to TrialGPT dict format for scoring."""
        return {
            "inclusion": {
                idx: cm.to_trialgpt_format()
                for idx, cm in self.inclusion.items()
            },
            "exclusion": {
                idx: cm.to_trialgpt_format()
                for idx, cm in self.exclusion.items()
            },
        }

    def imaging_relevant_criteria(self) -> List[CriterionMatch]:
        """Return criteria flagged as imaging-relevant."""
        result = []
        for cm in list(self.inclusion.values()) + list(self.exclusion.values()):
            if cm.imaging_relevant:
                result.append(cm)
        return result


# ──────────────────────────────────────────────────────────────────────────
# Aggregation / scoring  (TrialGPT format)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class AggregationResult:
    """Trial-level aggregation — TrialGPT-Ranking output."""
    relevance_explanation: str = ""
    relevance_score_R: float = 0.0
    eligibility_explanation: str = ""
    eligibility_score_E: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevance_explanation": self.relevance_explanation,
            "relevance_score_R": self.relevance_score_R,
            "eligibility_explanation": self.eligibility_explanation,
            "eligibility_score_E": self.eligibility_score_E,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregationResult":
        return cls(
            relevance_explanation=data.get("relevance_explanation", ""),
            relevance_score_R=float(data.get("relevance_score_R", 0)),
            eligibility_explanation=data.get("eligibility_explanation", ""),
            eligibility_score_E=float(data.get("eligibility_score_E", 0)),
        )


# ──────────────────────────────────────────────────────────────────────────
# Image analysis  (MedGemma output)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ImageAnalysis:
    """Structured output from MedGemma image analysis."""
    modality: str = ""
    body_part: str = ""
    findings: List[str] = field(default_factory=list)
    raw_text: str = ""
    # Per-criterion imaging assessments  (criterion text → verdict)
    imaging_criteria_assessments: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "body_part": self.body_part,
            "findings": self.findings,
            "raw_text": self.raw_text,
            "imaging_criteria_assessments": self.imaging_criteria_assessments,
        }


# ──────────────────────────────────────────────────────────────────────────
# Feedback request  (to patient)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class FeedbackRequest:
    """A request for additional patient information."""
    field: str
    question: str
    reason: str
    source_criteria: List[str] = field(default_factory=list)
    priority: str = "medium"  # high / medium / low

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "question": self.question,
            "reason": self.reason,
            "source_criteria": self.source_criteria,
            "priority": self.priority,
        }


# ──────────────────────────────────────────────────────────────────────────
# Ranked trial result
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class RankedTrial:
    """One trial in the ranked output list."""
    nct_id: str
    title: str = ""
    matching_score: float = 0.0
    aggregation_score: float = 0.0
    total_score: float = 0.0
    matching_result: Optional[MatchingResult] = None
    aggregation_result: Optional[AggregationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "nct_id": self.nct_id,
            "title": self.title,
            "matching_score": self.matching_score,
            "aggregation_score": self.aggregation_score,
            "total_score": self.total_score,
        }
        if self.aggregation_result:
            d["aggregation"] = self.aggregation_result.to_dict()
        return d


# ──────────────────────────────────────────────────────────────────────────
# Agent state  (full reasoning trace)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class StepRecord:
    """One step in the agent reasoning trace."""
    step: int
    tool: str
    input_summary: str = ""
    output_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "tool": self.tool,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
        }


@dataclass
class AgentState:
    """Complete state of one trial-matching agent run."""
    patient_id: str
    patient_text: str = ""
    patient_profile: Optional[PatientProfile] = None
    image_analysis: Optional[ImageAnalysis] = None
    keyword_summary: str = ""
    search_conditions: List[str] = field(default_factory=list)
    retrieved_trials: List[TrialInfo] = field(default_factory=list)
    matching_results: Dict[str, MatchingResult] = field(default_factory=dict)
    aggregation_results: Dict[str, AggregationResult] = field(default_factory=dict)
    ranked_trials: List[RankedTrial] = field(default_factory=list)
    feedback_requests: List[FeedbackRequest] = field(default_factory=list)
    step_history: List[StepRecord] = field(default_factory=list)
    iteration: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "patient_text": self.patient_text[:500] + "..." if len(self.patient_text) > 500 else self.patient_text,
            "patient_profile": self.patient_profile.to_dict() if self.patient_profile else None,
            "image_analysis": self.image_analysis.to_dict() if self.image_analysis else None,
            "keyword_summary": self.keyword_summary,
            "search_conditions": self.search_conditions,
            "n_retrieved_trials": len(self.retrieved_trials),
            "ranked_trials": [rt.to_dict() for rt in self.ranked_trials],
            "feedback_requests": [fr.to_dict() for fr in self.feedback_requests],
            "step_history": [s.to_dict() for s in self.step_history],
            "iteration": self.iteration,
        }


# ──────────────────────────────────────────────────────────────────────────
# Evaluation report
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class EvaluationReport:
    """Metrics from evaluating agent output against ground truth."""
    n_patients: int = 0
    n_trials_evaluated: int = 0
    criterion_accuracy: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    three_class_accuracy: float = 0.0
    macro_f1: float = 0.0
    cohens_kappa: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_patients": self.n_patients,
            "n_trials_evaluated": self.n_trials_evaluated,
            "criterion_accuracy": round(self.criterion_accuracy, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "precision_at_5": round(self.precision_at_5, 4),
            "precision_at_10": round(self.precision_at_10, 4),
            "three_class_accuracy": round(self.three_class_accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "cohens_kappa": round(self.cohens_kappa, 4),
        }
