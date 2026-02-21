"""Unit tests for the trial matching agent — schemas, prompts, tools, agent.

All tests are OFFLINE: API calls are mocked.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from trial_matching_agent.config import AgentConfig, DATA_DIR
from trial_matching_agent.schemas import (
    AggregationResult,
    AgentState,
    CriterionMatch,
    ExclusionLabel,
    FeedbackRequest,
    ImageAnalysis,
    InclusionLabel,
    MatchingResult,
    PatientProfile,
    RankedTrial,
    RelevanceLevel,
    StepRecord,
    TrialInfo,
)
from trial_matching_agent.prompts import (
    build_keyword_generation_prompt,
    build_matching_system_prompt,
    build_matching_user_prompt,
    number_patient_sentences,
    parse_criteria,
    format_trial_for_prompt,
)


# ══════════════════════════════════════════════════════════════════════
# Schema tests
# ══════════════════════════════════════════════════════════════════════

class TestTrialInfo:
    def test_from_trialgpt(self):
        raw = {
            "brief_title": "A Study of Drug X in NSCLC",
            "phase": "Phase 2",
            "drugs_list": ["Drug X"],
            "diseases_list": ["NSCLC"],
            "inclusion_criteria": "Age >= 18",
            "exclusion_criteria": "Prior therapy",
            "brief_summary": "Study of Drug X",
        }
        ti = TrialInfo.from_trialgpt("NCT001", raw)
        assert ti.nct_id == "NCT001"
        assert ti.brief_title == "A Study of Drug X in NSCLC"
        assert "Drug X" in ti.drugs_list
        assert ti.inclusion_criteria == "Age >= 18"

    def test_to_dict_roundtrip(self):
        ti = TrialInfo(nct_id="NCT002", brief_title="Test", phase="Phase 1")
        d = ti.to_dict()
        assert d["nct_id"] == "NCT002"
        assert d["phase"] == "Phase 1"


class TestPatientProfile:
    def test_from_dict(self):
        d = {
            "topic_id": "P001",
            "profile_text": "65yo male with NSCLC",
            "key_facts": [
                {"field": "condition", "value": "NSCLC", "evidence_span": "EHR"}
            ],
        }
        p = PatientProfile.from_dict(d)
        assert p.topic_id == "P001"
        assert len(p.key_facts) == 1
        assert p.key_facts[0].field == "condition"

    def test_to_dict(self):
        p = PatientProfile(topic_id="P002", profile_text="test")
        d = p.to_dict()
        assert d["topic_id"] == "P002"


class TestMatchingResult:
    def test_to_trialgpt_format(self):
        mr = MatchingResult()
        mr.inclusion["0"] = CriterionMatch(
            criterion_idx="0",
            reasoning="Patient is 65",
            evidence_sentence_ids=[0, 1],
            label="included",
        )
        mr.exclusion["0"] = CriterionMatch(
            criterion_idx="0",
            reasoning="No prior therapy mentioned",
            evidence_sentence_ids=[],
            label="not enough information",
        )
        fmt = mr.to_trialgpt_format()
        assert "inclusion" in fmt
        assert "exclusion" in fmt
        assert fmt["inclusion"]["0"][2] == "included"
        assert fmt["exclusion"]["0"][2] == "not enough information"

    def test_imaging_relevant_flag(self):
        cm = CriterionMatch(
            criterion_idx="1",
            reasoning="Measurable disease per RECIST",
            evidence_sentence_ids=[],
            label="not enough information",
            imaging_relevant=True,
        )
        assert cm.imaging_relevant is True


class TestAggregationResult:
    def test_from_dict(self):
        d = {
            "relevance_score_R": 85,
            "eligibility_score_E": 70,
            "relevance_explanation": "Meets most criteria",
            "eligibility_explanation": "Partially eligible",
        }
        agg = AggregationResult.from_dict(d)
        assert agg.relevance_score_R == 85
        assert agg.eligibility_score_E == 70

    def test_to_dict(self):
        agg = AggregationResult(relevance_score_R=90, eligibility_score_E=80)
        d = agg.to_dict()
        assert d["relevance_score_R"] == 90
        assert d["eligibility_score_E"] == 80


class TestAgentState:
    def test_to_dict_truncates_text(self):
        state = AgentState(
            patient_id="P001",
            patient_text="X" * 1000,
        )
        d = state.to_dict()
        assert len(d["patient_text"]) < 1000


class TestEnums:
    def test_inclusion_labels(self):
        assert InclusionLabel.INCLUDED.value == "included"
        assert InclusionLabel.NOT_ENOUGH_INFO.value == "not enough information"

    def test_exclusion_labels(self):
        assert ExclusionLabel.EXCLUDED.value == "excluded"

    def test_relevance_levels(self):
        assert RelevanceLevel.ELIGIBLE.value == 2
        assert RelevanceLevel.NOT_RELEVANT.value == 0


# ══════════════════════════════════════════════════════════════════════
# Prompt construction tests
# ══════════════════════════════════════════════════════════════════════

class TestPrompts:
    def test_keyword_generation_prompt(self):
        prompt = build_keyword_generation_prompt("65yo male with lung cancer")
        assert "65yo male" in prompt

    def test_matching_system_prompt_inclusion(self):
        prompt = build_matching_system_prompt("inclusion")
        assert "inclusion" in prompt.lower() or "included" in prompt.lower()

    def test_matching_system_prompt_exclusion(self):
        prompt = build_matching_system_prompt("exclusion")
        assert "exclusion" in prompt.lower() or "excluded" in prompt.lower()

    def test_number_patient_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        numbered = number_patient_sentences(text)
        # Uses "idx. sentence" format (TrialGPT convention)
        assert "0." in numbered or "0. " in numbered
        assert "1." in numbered or "1. " in numbered

    def test_parse_criteria(self):
        text = "1. Age >= 18\n2. Confirmed NSCLC\n3. ECOG <= 2"
        criteria = parse_criteria(text)
        assert len(criteria) >= 2  # Should find numbered items

    def test_format_trial_for_prompt(self):
        trial_dict = {
            "nct_id": "NCT001",
            "brief_title": "Test Trial",
            "brief_summary": "A test study",
            "inclusion_criteria": "1. Age >= 18",
            "exclusion_criteria": "1. Prior therapy",
            "diseases_list": ["NSCLC"],
            "drugs_list": ["Drug X"],
        }
        formatted = format_trial_for_prompt(trial_dict, "inclusion")
        assert "Test Trial" in formatted
        assert "Age >= 18" in formatted


# ══════════════════════════════════════════════════════════════════════
# Tool tests (mocked API calls)
# ══════════════════════════════════════════════════════════════════════

class TestToolsOffline:
    """Test tool dispatch and parsing with mocked LLM calls."""

    @pytest.fixture
    def config(self):
        return AgentConfig(
            gemini_api_key="fake-key",
            hf_token="fake-token",
            timeout_sec=10,
            retries=1,
        )

    @patch("trial_matching_agent.tools._call_gemini")
    def test_extract_patient_profile(self, mock_gemini, config):
        mock_gemini.return_value = json.dumps({
            "profile_text": "65-year-old male with stage IIIA NSCLC",
            "key_facts": [
                {"field": "age", "value": "65"},
                {"field": "condition", "value": "NSCLC stage IIIA"},
            ],
        })
        from trial_matching_agent.tools import extract_patient_profile
        profile = extract_patient_profile("Some EHR text", "P001", config)
        assert profile.topic_id == "p001"
        assert "NSCLC" in profile.profile_text

    @patch("trial_matching_agent.tools._call_gemini")
    def test_generate_search_keywords(self, mock_gemini, config):
        mock_gemini.return_value = json.dumps({
            "summary": "65yo male NSCLC",
            "conditions": ["non-small cell lung cancer", "NSCLC stage IIIA"],
        })
        from trial_matching_agent.tools import generate_search_keywords
        result = generate_search_keywords("Patient text", config)
        assert "conditions" in result
        assert len(result["conditions"]) == 2

    @patch("trial_matching_agent.tools._call_gemini")
    def test_match_criteria(self, mock_gemini, config):
        # Simulate two calls (inclusion + exclusion)
        mock_gemini.side_effect = [
            json.dumps({
                "0": ["Patient is 65, meets age criterion", [0], "included", False],
                "1": ["NSCLC confirmed", [1, 2], "included", False],
            }),
            json.dumps({
                "0": ["No prior immunotherapy mentioned", [], "not enough information", False],
            }),
        ]
        from trial_matching_agent.tools import match_criteria
        trial = TrialInfo(
            nct_id="NCT001",
            inclusion_criteria="1. Age >= 18\n2. Confirmed NSCLC",
            exclusion_criteria="1. No prior immunotherapy",
        )
        mr = match_criteria("65yo male with NSCLC", trial, config)
        assert len(mr.inclusion) == 2
        assert len(mr.exclusion) == 1
        assert mr.inclusion["0"].label == "included"
        assert mr.exclusion["0"].label == "not enough information"

    @patch("trial_matching_agent.tools._call_gemini")
    def test_aggregate_and_score(self, mock_gemini, config):
        mock_gemini.return_value = json.dumps({
            "relevance_score_R": 85,
            "eligibility_score_E": 70,
            "relevance_explanation": "Patient meets most inclusion criteria.",
            "eligibility_explanation": "Partially eligible.",
        })
        from trial_matching_agent.tools import aggregate_and_score
        trial = TrialInfo(nct_id="NCT001")
        mr = MatchingResult()
        mr.inclusion["0"] = CriterionMatch(
            criterion_idx="0", reasoning="meets", evidence_sentence_ids=[0], label="included"
        )
        agg = aggregate_and_score("Patient text", trial, mr, config)
        assert agg.relevance_score_R == 85
        assert agg.eligibility_score_E == 70

    @patch("trial_matching_agent.tools._call_gemini")
    def test_identify_missing_info(self, mock_gemini, config):
        mock_gemini.return_value = json.dumps([
            {
                "field": "ECOG performance status",
                "question": "What is the patient's ECOG score?",
                "reason": "Required by 3 trials",
                "source_criteria": ["NCT001_inc_2"],
                "priority": "high",
            }
        ])
        from trial_matching_agent.tools import identify_missing_info
        mr = MatchingResult()
        mr.inclusion["0"] = CriterionMatch(
            criterion_idx="0", reasoning="no info", evidence_sentence_ids=[], label="not enough information"
        )
        feedback = identify_missing_info(
            "Patient summary",
            {"NCT001": mr},
            {"NCT001": TrialInfo(nct_id="NCT001", brief_title="Test")},
            config,
        )
        assert len(feedback) == 1
        assert feedback[0].field == "ECOG performance status"
        assert feedback[0].priority == "high"

    def test_rank_and_build_results(self):
        from trial_matching_agent.tools import rank_and_build_results
        mr1 = MatchingResult()
        mr1.inclusion["0"] = CriterionMatch(
            criterion_idx="0", reasoning="", evidence_sentence_ids=[], label="included"
        )
        mr2 = MatchingResult()
        mr2.inclusion["0"] = CriterionMatch(
            criterion_idx="0", reasoning="", evidence_sentence_ids=[], label="not included"
        )

        agg1 = AggregationResult(relevance_score_R=90, eligibility_score_E=80)
        agg2 = AggregationResult(relevance_score_R=30, eligibility_score_E=20)

        ranked = rank_and_build_results(
            matching_results={"NCT001": mr1, "NCT002": mr2},
            aggregation_results={"NCT001": agg1, "NCT002": agg2},
            trial_infos={
                "NCT001": TrialInfo(nct_id="NCT001", brief_title="Trial 1"),
                "NCT002": TrialInfo(nct_id="NCT002", brief_title="Trial 2"),
            },
        )
        assert ranked[0].nct_id == "NCT001"
        assert ranked[0].total_score > ranked[1].total_score


# ══════════════════════════════════════════════════════════════════════
# JSON parser tests
# ══════════════════════════════════════════════════════════════════════

class TestParseJson:
    def test_direct_json(self):
        from trial_matching_agent.tools import _parse_json_from_text
        result = _parse_json_from_text('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced(self):
        from trial_matching_agent.tools import _parse_json_from_text
        text = '```json\n{"key": "value"}\n```'
        result = _parse_json_from_text(text)
        assert result == {"key": "value"}

    def test_embedded_json(self):
        from trial_matching_agent.tools import _parse_json_from_text
        text = 'Here is the output:\n\n{"key": "value"}\n\nDone.'
        result = _parse_json_from_text(text)
        assert result == {"key": "value"}

    def test_invalid_returns_none(self):
        from trial_matching_agent.tools import _parse_json_from_text
        result = _parse_json_from_text("not json at all")
        assert result is None


# ══════════════════════════════════════════════════════════════════════
# Config tests
# ══════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_from_env_no_crash(self):
        """Config.from_env() should not crash even without API keys."""
        config = AgentConfig.from_env()
        assert config.gemini_model == "gemini-2.5-pro"
        assert config.retries == 3

    def test_resolve_keys_idempotent(self):
        config = AgentConfig(gemini_api_key="test-key")
        config.resolve_keys()
        assert config.gemini_api_key == "test-key"
