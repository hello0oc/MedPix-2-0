"""Agentic Clinical Trial Matching System.

Uses Gemini 2.5 Pro as the orchestrator with native function-calling
and MedGemma as a specialist worker for medical image analysis.
Evaluation grounded in the TrialGPT dataset (SIGIR corpus) with
expert-annotated 3-level relevance labels.
"""

__version__ = "0.1.0"

from .agent import TrialMatchingAgent
from .config import AgentConfig
from .schemas import (
    AgentState,
    AggregationResult,
    CriterionMatch,
    FeedbackRequest,
    ImageAnalysis,
    MatchingResult,
    PatientProfile,
    RankedTrial,
    TrialInfo,
)

__all__ = [
    "TrialMatchingAgent",
    "AgentConfig",
    "AgentState",
    "AggregationResult",
    "CriterionMatch",
    "FeedbackRequest",
    "ImageAnalysis",
    "MatchingResult",
    "PatientProfile",
    "RankedTrial",
    "TrialInfo",
]