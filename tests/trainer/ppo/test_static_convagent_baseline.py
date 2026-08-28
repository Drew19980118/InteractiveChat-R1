import pytest

from verl.utils.reward_score.static_convagent import (
    direct_evidence_coverage,
    monitor_plateau_reached,
    static_convagent_allowed_actions,
)


def test_static_convagent_keeps_mixed_action_supervision_explicit():
    ground_truth = [
        {"action": "answer", "response": "Paris"},
        {"action": "clarify", "response": "Which France?"},
    ]
    assert static_convagent_allowed_actions(ground_truth, data_source="inscit") == {
        "answer",
        "clarify",
    }


def test_static_convagent_qrecc_uses_implicit_answer_action():
    assert static_convagent_allowed_actions(
        [{"response": "A reference answer"}], data_source="qrecc"
    ) == {"answer"}


def test_direct_evidence_coverage_requires_full_short_factoid_phrase():
    passages = [{"passage_text": "Paris is the capital city of France."}]
    assert direct_evidence_coverage("Paris", passages) == 1.0
    assert direct_evidence_coverage("capital France", passages) == 0.0


def test_direct_evidence_coverage_uses_best_long_passage_f1():
    passages = [
        {"passage_text": "This passage is unrelated."},
        {"passage_text": "Marie Curie discovered radium and polonium."},
    ]
    assert direct_evidence_coverage("Marie Curie discovered radium", passages) == pytest.approx(8 / 9)


def test_monitor_requires_no_recent_improvement_and_stability():
    assert not monitor_plateau_reached(
        [0.10, 0.20, 0.30, 0.31],
        patience=2,
        min_delta=0.002,
        stability_window=3,
        stability_tolerance=0.005,
    )
    assert monitor_plateau_reached(
        [0.20, 0.30, 0.301, 0.300, 0.301],
        patience=2,
        min_delta=0.002,
        stability_window=3,
        stability_tolerance=0.005,
    )
