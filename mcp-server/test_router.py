"""Tests for the intelligent model router."""

import pytest
from router import classify_query, RoutingDecision


MODELS = {
    "hivecoder": {
        "endpoint": "http://127.0.0.1:8089",
        "display_name": "HiveCoder (Qwen3-14B)",
        "capabilities": ["code", "shell", "tools", "reasoning", "analysis", "explanation", "general"],
    },
}


class TestExplicitHint:
    def test_valid_hint(self):
        d = classify_query("hello", model_hint="hivecoder", available_models=MODELS)
        assert d.model_id == "hivecoder"
        assert "explicit_hint" in d.reason
        assert d.confidence == 1.0

    def test_invalid_hint_falls_back(self):
        d = classify_query("hello", model_hint="nonexistent", available_models=MODELS)
        assert d.model_id == "hivecoder"
        assert "hint_not_found" in d.reason


class TestReasonMode:
    def test_reason_mode_uses_same_model(self):
        d = classify_query("what is python", reason_mode=True, available_models=MODELS)
        assert d.model_id == "hivecoder"
        assert d.reason == "reason_mode"
        assert d.confidence == 1.0

    def test_reason_mode_beats_code_signals(self):
        d = classify_query("write a python function", reason_mode=True, available_models=MODELS)
        assert d.model_id == "hivecoder"
        assert d.reason == "reason_mode"

    def test_reason_mode_flags_decision(self):
        """reason_mode sets the flag for http_server to toggle Qwen3 thinking mode."""
        d = classify_query("explain architecture", reason_mode=True, available_models=MODELS)
        assert d.model_id == "hivecoder"
        assert d.reason == "reason_mode"
        # Same model as non-reason mode, but flagged
        d2 = classify_query("explain architecture", reason_mode=False, available_models=MODELS)
        assert d2.model_id == d.model_id
        assert d2.reason != "reason_mode"


class TestHeuristics:
    def test_code_query(self):
        d = classify_query(
            "write a python function to parse JSON",
            available_models=MODELS,
        )
        assert d.model_id == "hivecoder"
        assert "code_signals" in d.reason

    def test_reasoning_query(self):
        d = classify_query(
            "explain why comparing btrfs vs ext4, what are the pros and cons and trade-offs",
            available_models=MODELS,
        )
        assert d.model_id == "hivecoder"
        assert "reasoning_signals" in d.reason

    def test_code_block_bonus(self):
        d = classify_query(
            "explain this:\n```python\ndef foo(): pass\n```",
            available_models=MODELS,
        )
        assert d.model_id == "hivecoder"

    def test_ambiguous_defaults_to_fast(self):
        d = classify_query("hello world", available_models=MODELS)
        assert d.model_id == "hivecoder"
        assert d.reason == "no_signals"

    def test_mixed_signals_tie_goes_to_fast(self):
        d = classify_query("explain this python code", available_models=MODELS)
        assert d.model_id == "hivecoder"

    def test_import_pattern_bonus(self):
        d = classify_query("import numpy as np", available_models=MODELS)
        assert d.model_id == "hivecoder"

    def test_file_path_bonus(self):
        d = classify_query("check /var/log/syslog for errors", available_models=MODELS)
        assert d.model_id == "hivecoder"


class TestNoModelsConfig:
    def test_no_models_uses_default(self):
        d = classify_query("anything", available_models=None)
        assert d.model_id == "hivecoder"

    def test_empty_models_uses_default(self):
        d = classify_query("anything", available_models={})
        assert d.model_id == "hivecoder"


class TestRoutingDecision:
    def test_dataclass_fields(self):
        d = RoutingDecision(model_id="test", reason="test", confidence=0.5)
        assert d.model_id == "test"
        assert d.reason == "test"
        assert d.confidence == 0.5
