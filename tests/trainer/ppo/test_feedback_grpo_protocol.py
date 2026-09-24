"""CPU-only checks of the alternating-feedback reward contract.

The loader avoids importing verl's GPU-training package initializer. It loads
the real shared terminal parser and reward helpers; only the unused OpenAI
client import is stubbed while scoring these local strings.
"""

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def protocol(monkeypatch):
    for package in (
        "verl", "verl.utils", "verl.utils.reward_score", "verl.recipe",
        "verl.recipe.feedback_grpo",
    ):
        module = types.ModuleType(package)
        module.__path__ = [str(ROOT.joinpath(*package.split(".")))]
        monkeypatch.setitem(sys.modules, package, module)
    unused_openai = types.ModuleType("openai")
    unused_openai.OpenAI = object
    monkeypatch.setitem(sys.modules, "openai", unused_openai)

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    load("verl.utils.reward_score.static_convagent", "verl/utils/reward_score/static_convagent.py")
    load("verl.utils.reward_score.static_chatr1", "verl/utils/reward_score/static_chatr1.py")
    load("verl.utils.reward_score.ground_truth", "verl/utils/reward_score/ground_truth.py")
    load("verl.utils.reward_score.info_gain", "verl/utils/reward_score/info_gain.py")
    return load("verl.recipe.feedback_grpo.protocol", "verl/recipe/feedback_grpo/protocol.py")


def response(action, content=""):
    return f"<think>private reasoning</think><{action}>{content}</{action}>"


ANSWER_GT = [
    {"action": "answer", "response": "Paris France"},
    {"action": "answer", "response": "The city of Paris"},
]


def test_system_reward_uses_max_answer_ref_and_exact_action_weight(protocol):
    result = protocol.assess_response(response("answer", "The city of Paris"), ANSWER_GT)
    assert result.format_valid
    assert result.f1 == 1.0
    assert result.action_score == 1.0
    assert result.reward == 1.5


def test_clarify_and_nonanswer_never_receive_textual_f1(protocol):
    mixed = ANSWER_GT + [
        {"action": "clarify", "response": "Which city?"},
        {"action": "nonanswer", "response": ""},
    ]
    for action, text in (("answer", "Paris France"), ("clarify", "Which city?"), ("nonanswer", "")):
        result = protocol.assess_response(response(action, text), mixed)
        assert result.action_score == 1.0
        assert result.f1 == (1.0 if action == "answer" else 0.0)
        assert result.reward == (1.5 if action == "answer" else 0.5)


def test_json_wrapper_and_reference_separator_are_supported_only_in_gt(protocol):
    gold = {"ground_truth": [{"action": "answer", "response": "London<|answer_split|>Paris"}]}
    result = protocol.assess_response(response("answer", "Paris"), json.dumps(gold))
    assert result.reward == 1.5
    malformed = protocol.assess_response(response("answer", "London<|answer_split|>Paris"), gold)
    assert not malformed.format_valid
    assert malformed.reward == -0.25


def test_answer_only_dataset_accepts_plain_reference(protocol):
    result = protocol.assess_response(response("answer", "Paris"), "Paris", "qrecc")
    assert result.reward == 1.5


@pytest.mark.parametrize("malformed", [
    "unstructured response",
    "<answer>Paris France</answer>",
    "<think>x</think><nonanswer>forbidden text</nonanswer>",
    "<think>x</think><clarify></clarify>",
    "<think>x</think><answer>Paris</answer><answer>France</answer>",
    "<think>x</think><answer>Paris",
    "<think>Need evidence</think><search>Paris</search>",
])
def test_malformed_has_negative_action_reward_and_no_text_reward(protocol, malformed):
    result = protocol.assess_response(malformed, ANSWER_GT)
    assert not result.format_valid
    assert result.action_score == -0.5
    assert result.f1 == 0
    assert result.reward == -0.25
    assert result.content == ""


def test_user_full_delta_only_when_both_terminal_actions_are_answer(protocol):
    first = protocol.assess_response(response("answer", "Paris"), ANSWER_GT)
    second = protocol.assess_response(response("answer", "Paris France"), ANSWER_GT)
    assert protocol.feedback_reward(first, second) == pytest.approx(1 / 3)
    assert protocol.feedback_reward(second, first) == pytest.approx(-1 / 3)


def test_changed_action_uses_only_action_delta_not_answer_f1(protocol):
    first = protocol.assess_response(response("clarify", "Which city?"), ANSWER_GT)
    second = protocol.assess_response(response("answer", "Paris France"), ANSWER_GT)
    assert first.format_valid and first.action_score == -0.5
    assert protocol.feedback_reward(first, second) == 0.75
    assert protocol.feedback_reward(second, first) == -0.75
    # Both terminal choices are allowed: no feedback reward for switching,
    # even though the second response also gets a nonzero answer F1.
    mixed = ANSWER_GT + [{"action": "clarify", "response": "Which city?"}]
    first = protocol.assess_response(response("clarify", "Which city?"), mixed)
    second = protocol.assess_response(response("answer", "Paris France"), mixed)
    assert protocol.feedback_reward(first, second) == 0.0


def test_same_nonanswer_or_clarify_action_yields_zero_feedback_reward(protocol):
    for action, text in (("clarify", "Which city?"), ("nonanswer", "")):
        gold = [{"action": action, "response": text}]
        first = protocol.assess_response(response(action, text), gold)
        assert protocol.feedback_reward(first, first) == 0


def test_filtering_happens_before_group_mean_and_keeps_legal_wrong_actions(protocol):
    first = protocol.assess_response(response("answer", "Paris"), ANSWER_GT)
    seconds = [
        protocol.assess_response(response("answer", "Paris France"), ANSWER_GT),
        protocol.assess_response("malformed", ANSWER_GT),
        protocol.assess_response(response("answer", "Paris"), ANSWER_GT),
        protocol.assess_response(response("clarify", "Which city?"), ANSWER_GT),
    ]
    group = protocol.filter_feedback_group(first, seconds)
    assert not group.skipped
    assert group.retained_indices == (0, 2, 3)
    assert group.rewards == pytest.approx((1 / 3, 0, -0.75))
    assert group.advantages == pytest.approx(protocol.group_advantages(group.rewards))
    assert sum(group.advantages) == pytest.approx(0, abs=1e-12)


def test_malformed_first_or_too_few_second_responses_skip_feedback_group(protocol):
    first = protocol.assess_response(response("answer", "Paris"), ANSWER_GT)
    invalid = protocol.assess_response("malformed", ANSWER_GT)
    group = protocol.filter_feedback_group(invalid, [first, first])
    assert group.skipped and group.reason == "invalid_first_response"
    assert group.rewards == group.advantages == ()
    group = protocol.filter_feedback_group(first, [invalid, first, invalid])
    assert group.skipped and group.retained_indices == (1,)
    assert group.reason == "fewer_than_two_valid_feedbacks"
    assert group.rewards == group.advantages == ()
    with pytest.raises(ValueError):
        protocol.feedback_reward(first, invalid)


def test_group_normalization_population_std_and_negative_relative_advantage(protocol):
    assert protocol.group_advantages([-2, -1]) == pytest.approx([-1, 1], abs=3e-6)
    assert protocol.group_advantages([-1, 0, 1]) == pytest.approx(
        [-1 / (math.sqrt(2 / 3) + 1e-6), 0, 1 / (math.sqrt(2 / 3) + 1e-6)]
    )
    assert protocol.group_advantages([0, 0, 0]) == [0, 0, 0]
    assert protocol.group_advantages([-0.25, -0.25]) == [0, 0]
    assert protocol.group_advantages([]) == []
    with pytest.raises(ValueError):
        protocol.group_advantages([0, float("nan")])


def test_zero_reward_group_is_retained_for_explicit_diagnostics(protocol):
    first = protocol.assess_response(response("answer", "Paris"), ANSWER_GT)
    group = protocol.filter_feedback_group(first, [first, first])
    assert not group.skipped
    assert group.zero_variance
    assert group.rewards == group.advantages == (0, 0)


def test_feedback_input_excludes_system_reasoning_tools_and_ground_truth(protocol):
    trajectory = (
        '<think>SECRET_REASONING</think><search>SECRET_SEARCH</search>'
        '\n<|im_start|>tool\nSECRET_EVIDENCE\n<|im_end|>'
        '\n<|im_start|>assistant\n<think>SECRET_REASONING_2</think>'
        '<answer>Paris</answer>'
    )
    first = protocol.assess_response(trajectory, [{"action": "answer", "response": "SECRET_GT"}])
    prompt = protocol.user_feedback_prompt("Original user history", "Which city?", first)
    assert prompt.endswith("Feedback:")
    assert "Original user history" in prompt and "Which city?" in prompt
    assert "System Response:\nParis" in prompt
    assert "SECRET" not in prompt
    assert "<think>" not in prompt and "<search>" not in prompt and "<answer>" not in prompt
    assert protocol.REANSWER_INSTRUCTION not in prompt


def test_empty_nonanswer_has_readable_feedback_input_and_invalid_is_rejected(protocol):
    first = protocol.assess_response(response("nonanswer"), [{"action": "nonanswer"}])
    assert "System Response:\nNo answer was provided." in protocol.user_feedback_prompt("", "Why?", first)
    invalid = protocol.assess_response("broken", ANSWER_GT)
    with pytest.raises(ValueError):
        protocol.user_feedback_prompt("", "Why?", invalid)
