import numpy as np
import pytest
import torch

from scrl.llm_agent.simulated_user import (
    AgentAction,
    UserSimulatorClient,
    build_user_simulator_request_payload,
    canonicalize_inscit_dialogue,
    keep_first_complete_action,
    parse_agent_action,
    parse_user_simulator_judgement,
)
from scrl.llm_agent.simulated_user_rollout import (
    SimulatedUserGenerationManager,
    SimulatedUserSettings,
    _Event,
    _State,
)
from verl.trainer.ppo.simulated_user_algos import compute_simulated_user_sparse_grpo_advantage
from verl.trainer.ppo.simulated_user_validation import _ndcg_at_3, select_answer_for_text_metrics


def test_canonical_inscit_uses_the_next_context_response_not_candidate_order():
    dialogue = canonicalize_inscit_dialogue(
        "d0",
        [
            {
                "context": ["q0"],
                "labels": [
                    {"responseType": "clarification", "response": "Which city?"},
                    {"responseType": "directAnswer", "response": "It is Paris."},
                ],
            },
            {
                "context": ["q0", "It is Paris.", "q1"],
                "labels": [{"responseType": "noAnswerNoRelevantInfo", "response": ""}],
            },
        ],
    )
    assert dialogue["subtasks"][0]["expected_action"] == "answer"
    assert dialogue["subtasks"][0]["gold_response"] == "It is Paris."
    assert dialogue["subtasks"][0]["selection_source"] == "next_context_match"


@pytest.mark.parametrize(
    "raw,kind",
    [
        ("<think>x</think><answer>yes</answer>", "answer"),
        ("<think>x</think><clarify>Which date?</clarify>", "clarify"),
        ("<think>x</think><nonanswer></nonanswer>", "nonanswer"),
        (
            '<think>x</think><tool_call>{"name":"web_search","arguments":{"query":["q"]}}</tool_call>',
            "tool_call",
        ),
    ],
)
def test_parser_accepts_exactly_the_four_legal_initial_actions(raw, kind):
    action = parse_agent_action(raw, allow_clarify=True, allow_nonanswer=True)
    assert action.valid
    assert action.kind == kind


def test_answer_locked_retry_allows_tool_but_rejects_nonanswer_and_empty_answer():
    tool = parse_agent_action(
        '<think>x</think><tool_call>{"name":"web_search","arguments":{"query":["q"]}}</tool_call>',
        allow_clarify=True,
        allow_nonanswer=True,
        force_answer=True,
        allow_tool_call_in_answer_locked_retry=True,
    )
    assert tool.valid and tool.kind == "tool_call"
    assert not parse_agent_action(
        "<think>x</think><nonanswer></nonanswer>",
        allow_clarify=True,
        allow_nonanswer=True,
        force_answer=True,
        allow_tool_call_in_answer_locked_retry=True,
    ).valid


def test_controlled_uci_to_f1_ablation_keeps_auxiliary_channels_but_skips_uci():
    """The UCI ablation must not silently become the broad F1-only baseline."""
    manager = object.__new__(SimulatedUserGenerationManager)
    manager.settings = SimulatedUserSettings(
        reward_mode="uci_replaced_by_answer_f1",
        enable_search_efficiency=False,
    )
    state = _State(
        rollout_index=0,
        dialogue={
            "subtasks": [
                {
                    "question": "What is the capital of France?",
                    "expected_action": "answer",
                    "gold_response": "Paris",
                }
            ]
        },
        initial_messages=[],
        public_messages=[],
        messages=[],
        subtask_initial_messages=[],
        pending_action_prompt=[],
    )
    pending_uci = []
    pending_judgements = []
    manager._process_terminal_action(
        state,
        AgentAction(valid=True, kind="answer", content="Paris"),
        raw_response="<think>x</think><answer>Paris</answer>",
        original_raw_response="<think>x</think><answer>Paris</answer>",
        trailing_content_discarded=False,
        pending_uci=pending_uci,
        pending_judgements=pending_judgements,
    )

    event = state.events[0]
    assert event.components["format"] == 0.0
    assert event.components["action"] == 1.0
    assert event.components["answer_f1"] == 1.0
    assert "uci" not in event.components
    assert pending_uci == []
    assert pending_judgements == [(state, event, "Paris")]
    assert not parse_agent_action(
        "<think>x</think><answer></answer>",
        allow_clarify=True,
        allow_nonanswer=True,
        force_answer=True,
        allow_tool_call_in_answer_locked_retry=True,
    ).valid


def test_first_complete_action_discards_chatml_or_second_action_suffix_only():
    raw = (
        '<think>x</think><tool_call>{"name":"web_search","arguments":{"query":["q"]}}</tool_call>'
        '<|im_start|>assistant\n<think>ignored</think><answer>ignored</answer>'
    )
    kept, discarded = keep_first_complete_action(raw)
    assert discarded
    assert kept == '<think>x</think><tool_call>{"name":"web_search","arguments":{"query":["q"]}}</tool_call>'
    assert parse_agent_action(kept, allow_clarify=True, allow_nonanswer=True).valid


def test_incomplete_action_is_not_silently_repaired():
    raw = '<think>x</think><tool_call>{"name":"web_search","arguments":{"query":["q"]}}'
    kept, discarded = keep_first_complete_action(raw)
    assert not discarded
    assert not parse_agent_action(kept, allow_clarify=True, allow_nonanswer=True).valid


def test_user_simulator_payload_is_limited_to_public_dialogue_and_final_answer():
    payload = build_user_simulator_request_payload(
        dialogue_id="d0",
        subtask_index=2,
        question="What happened next?",
        public_transcript="User: Earlier question\nAssistant: Earlier final answer",
        answer="The final answer visible to the user.",
    )
    assert payload == {
        "dialogue_id": "d0",
        "subtask_index": 2,
        "current_question": "What happened next?",
        "public_dialogue_history": "User: Earlier question\nAssistant: Earlier final answer",
        "assistant_final_answer": "The final answer visible to the user.",
    }
    serialized = str(payload)
    assert "hidden_reference" not in serialized
    assert "tool_call" not in serialized
    assert "<think>" not in serialized


def test_user_simulator_parser_repairs_only_the_common_invalid_apostrophe_escape():
    level, feedback = parse_user_simulator_judgement(
        r'{"level": 2, "feedback": "Define b\'reishit before using it."}'
    )
    assert level == 2
    assert feedback == "Define b'reishit before using it."


def test_user_simulator_parser_recovers_a_missing_feedback_opening_quote_only():
    level, feedback = parse_user_simulator_judgement(
        '{"level": 2, "feedback": answer is too brief and lacks the requested detail}'
    )
    assert level == 2
    assert feedback == "answer is too brief and lacks the requested detail"


def test_user_simulator_parser_recovers_the_observed_feedback_key_colon_typo():
    level, feedback = parse_user_simulator_judgement(
        '{"level": 2, "feedback: "The answer is unclear and needs context"}'
    )
    assert level == 2
    assert feedback == "The answer is unclear and needs context"


class _FakeSimulatorResponse:
    def __init__(self, content: str, status_code: int = 200):
        self._content = content
        self.status_code = status_code
        self.text = content

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_user_simulator_retries_without_response_format_after_malformed_output(monkeypatch):
    import requests

    responses = iter(
        [
            _FakeSimulatorResponse('{"'),
            _FakeSimulatorResponse('{"level": 1, "feedback": "Clear."}'),
        ]
    )
    payloads = []

    def fake_post(*_args, **kwargs):
        payloads.append(kwargs["json"])
        return next(responses)

    monkeypatch.setattr(requests, "post", fake_post)
    judgement = UserSimulatorClient(base_url="http://simulator:8010", model="sim").judge_answer(
        dialogue_id="d0", subtask_index=0, question="q", public_transcript="", answer="a"
    )
    assert judgement.level == 1
    assert judgement.source == "retry_success"
    assert len(payloads) == 2
    assert "response_format" not in payloads[0]
    assert payloads[1]["max_tokens"] == 64


def test_user_simulator_uses_auditable_conservative_fallback_after_two_failures(monkeypatch):
    import requests

    responses = iter([_FakeSimulatorResponse('{"'), _FakeSimulatorResponse('{"')])
    monkeypatch.setattr(requests, "post", lambda *_args, **_kwargs: next(responses))
    judgement = UserSimulatorClient(base_url="http://simulator:8010", model="sim").judge_answer(
        dialogue_id="d0", subtask_index=0, question="q", public_transcript="", answer="a"
    )
    assert judgement.level == 2
    assert judgement.source == "fallback_after_invalid_output_then_invalid_output"


class _CharacterTokenizer:
    """Minimal deterministic tokenizer for live-context budget unit tests."""

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": list(str(text))}

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(token_ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        assert add_generation_prompt and not tokenize
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def test_live_context_adaptively_compacts_only_an_oversized_newest_topk():
    # Build the manager without its heavyweight actor/FSDP constructor. The
    # method under test only needs a tokenizer and a context-budget provider.
    manager = object.__new__(SimulatedUserGenerationManager)
    manager.tokenizer = _CharacterTokenizer()
    manager.settings = SimulatedUserSettings(tool_observation_token_cap=0)
    manager._prompt_token_budget = lambda: 512

    raw_retrieval = [
        {
            "search_query": "q",
            "web_page_info_list": [
                {"passage_id": f"p{rank}", "title": f"title {rank}", "quick_summary": "x" * 1000}
                for rank in range(1, 4)
            ],
        }
    ]
    full = manager._truncate_tool_observation(raw_retrieval)
    event = _Event(subtask_index=0, response_depth=1, action="tool_call", raw_response="")
    tool_message = {"role": "tool", "name": "web_search", "content": full}
    state = _State(
        rollout_index=0,
        dialogue={"subtasks": [{"question": "q"}]},
        initial_messages=[],
        public_messages=[],
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "current question"},
            tool_message,
        ],
        subtask_initial_messages=[],
        active_tool_observations=[
            {
                "message": tool_message,
                "raw_content": raw_retrieval,
                "full_content": full,
                "compact_content": manager._compact_tool_observation(raw_retrieval),
                "compacted": False,
                "latest_compacted": False,
                "event": event,
            }
        ],
    )

    manager._enforce_live_policy_context_budget(state)

    assert manager._policy_prompt_token_count(state) <= 504
    assert event.latest_retrieval_compacted
    assert 0 < event.latest_retrieval_visible_tokens < event.latest_retrieval_original_tokens


def test_single_query_mode_canonicalizes_an_accidental_multi_query_tool_call():
    action = parse_agent_action(
        '<think>x</think><tool_call>{"name":"web_search","arguments":{"query":["first", "second"]}}</tool_call>',
        allow_clarify=True,
        allow_nonanswer=True,
        max_search_queries=1,
    )
    assert action.valid and action.kind == "tool_call"
    assert action.tool_call["arguments"]["query"] == ["first"]


def test_static_gold_context_uses_source_prefix_and_never_carries_sampled_answers_forward():
    manager = object.__new__(SimulatedUserGenerationManager)
    manager.system_prompt = "system contract"
    manager.settings = SimulatedUserSettings(
        enable_user_feedback=False,
        use_static_gold_context=True,
    )
    dialogue = {
        "subtasks": [
            {
                "question": "q0",
                "source_context": ["q0"],
                "gold_response": "gold0",
            },
            {
                "question": "q1",
                "source_context": ["q0", "gold0", "q1"],
                "gold_response": "gold1",
            },
        ]
    }
    state = manager._new_state(rollout_index=0, dialogue=dialogue)
    assert state.messages == [
        {"role": "system", "content": "system contract"},
        {"role": "user", "content": "q0"},
    ]

    # A sampled answer is deliberately irrelevant to the next static task.
    state.messages.append({"role": "assistant", "content": "sampled and possibly wrong"})
    manager._move_to_next_subtask(state)
    assert state.subtask_index == 1
    assert state.answer_depth == 1
    assert state.messages == [
        {"role": "system", "content": "system contract"},
        {"role": "user", "content": "q0"},
        {"role": "assistant", "content": "gold0"},
        {"role": "user", "content": "q1"},
    ]


def test_text_metrics_fall_back_to_last_valid_answer_but_not_first_or_oracle_best():
    events = [
        {
            "predicted_action": "answer",
            "format_valid": True,
            "raw_response": "<think>x</think><answer>older answer</answer>",
        },
        {
            "predicted_action": "answer",
            "format_valid": True,
            "raw_response": "<think>x</think><answer>newer answer</answer>",
        },
        {
            "predicted_action": "invalid",
            "format_valid": False,
            "raw_response": "<think>x</think>",
        },
    ]
    event, answer, source = select_answer_for_text_metrics(events)
    assert event is events[1]
    assert answer == "newer answer"
    assert source == "last_valid_answer"


def test_text_metrics_use_terminal_answer_or_zero_when_no_valid_answer_exists():
    terminal = {
        "predicted_action": "answer",
        "format_valid": True,
        "raw_response": "<think>x</think><answer>terminal answer</answer>",
    }
    event, answer, source = select_answer_for_text_metrics([terminal])
    assert event is terminal
    assert answer == "terminal answer"
    assert source == "terminal"

    event, answer, source = select_answer_for_text_metrics(
        [{"predicted_action": "invalid", "format_valid": False, "raw_response": "<think>x</think>"}]
    )
    assert event is None
    assert answer == ""
    assert source == "missing"


def test_topiocqa_retrieval_ndcg_uses_normalized_passage_text_not_incompatible_ids():
    # TopiOCQA's ``wiki:...`` Gold_passage IDs differ from the local
    # retriever's ID namespace. Its content is the relevance label.
    gold_text = "The army personnel and thousands of Australian airmen took part."
    predicted_text = "  the ARMY personnel and thousands of Australian airmen took part.  "
    assert _ndcg_at_3(
        ["unrelated", predicted_text],
        [gold_text],
        match_by_text=True,
    ) == pytest.approx(1.0 / np.log2(3))
    assert _ndcg_at_3(
        ["local:42"],
        ["wiki:5498209"],
        match_by_text=False,
    ) == 0.0


def _sparse_adv(values, masks, boundaries, subtasks, depths):
    return compute_simulated_user_sparse_grpo_advantage(
        component_values={"action": values},
        component_masks={"action": masks},
        turn_boundary_mask=boundaries,
        subtask_ids=subtasks,
        response_depths=depths,
        response_mask=torch.ones_like(values),
        index=np.array(["d"] * values.size(0), dtype=object),
        gamma=1.0,
    )


def test_sparse_grpo_groups_only_active_rollouts_at_each_depth():
    # Row 2 never reaches depth 2. It must not be treated as a zero-reward
    # sibling in the depth-2 normalization group.
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])
    masks = torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.bool)
    boundaries = masks.clone()
    subtasks = torch.zeros_like(values, dtype=torch.long)
    depths = torch.tensor([[1, 2], [1, 2], [1, -1]])
    advantages, _, metrics = _sparse_adv(values, masks, boundaries, subtasks, depths)

    assert torch.allclose(
        advantages[:, 0], torch.tensor([-2.2247448, 1.0, 1.2247448]), atol=1e-5
    )
    # Depth-2 has two members (2 and 4), so it is normalized; row 2 is absent.
    assert torch.allclose(advantages[:2, 1], torch.tensor([-1.0, 1.0]), atol=1e-5)
    assert metrics["sim_user/normalization_groups"] == 2.0


def test_evidence_utility_is_independently_normalized_per_response_depth():
    # The raw evidence score is an absolute mean gold log-prob and therefore
    # normally negative.  Sparse GRPO must normalize it only against siblings
    # which reached the same original subtask and answer depth.
    values = torch.tensor([[-6.0, -2.0], [-4.0, -6.0], [-2.0, 0.0]])
    masks = torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.bool)
    boundaries = masks.clone()
    subtasks = torch.zeros_like(values, dtype=torch.long)
    depths = torch.tensor([[1, 2], [1, 2], [1, -1]])
    advantages, _, metrics = compute_simulated_user_sparse_grpo_advantage(
        component_values={"evidence_utility": values},
        component_masks={"evidence_utility": masks},
        turn_boundary_mask=boundaries,
        subtask_ids=subtasks,
        response_depths=depths,
        response_mask=torch.ones_like(values),
        index=np.array(["d", "d", "d"], dtype=object),
        gamma=1.0,
    )

    # Depth 1 uses all three active rollout scores [-6, -4, -2].  Its local
    # normalized values are [-1.2247, 0, 1.2247]; depth-2's [1, -1] return is
    # then propagated to the earlier action in each matching trajectory.
    assert torch.allclose(
        advantages[:, 0], torch.tensor([-0.2247448, -1.0, 1.2247448]), atol=1e-5
    )
    # Depth 2 contains only rows 0 and 1; row 2 is not a zero-reward sibling.
    assert torch.allclose(advantages[:2, 1], torch.tensor([1.0, -1.0]), atol=1e-5)
    assert advantages[2, 1] == 0.0
    assert metrics["sim_user/evidence_utility_groups"] == 2.0


def test_answer_f1_is_independently_normalized_per_response_depth():
    values = torch.tensor([[0.1, 0.8], [0.4, 0.2], [0.9, 0.0]])
    masks = torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.bool)
    boundaries = masks.clone()
    subtasks = torch.zeros_like(values, dtype=torch.long)
    depths = torch.tensor([[1, 2], [1, 2], [1, -1]])
    advantages, _, metrics = compute_simulated_user_sparse_grpo_advantage(
        component_values={"answer_f1": values},
        component_masks={"answer_f1": masks},
        turn_boundary_mask=boundaries,
        subtask_ids=subtasks,
        response_depths=depths,
        response_mask=torch.ones_like(values),
        index=np.array(["d", "d", "d"], dtype=object),
        gamma=1.0,
    )

    # Every active answer at depth 1 is a sibling; depth 2 excludes row 2.
    # The normalized depth-2 return [1, -1] is also credited to depth 1.
    assert torch.allclose(
        advantages[:, 0], torch.tensor([-0.1111112, -1.2020202, 1.3131313]), atol=1e-5        
    )
    assert torch.allclose(advantages[:2, 1], torch.tensor([1.0, -1.0]), atol=1e-5)
    assert advantages[2, 1] == 0.0
    assert metrics["sim_user/answer_f1_groups"] == 2.0


def test_search_efficiency_leaves_the_first_retrieval_free():
    reward = SimulatedUserGenerationManager._search_efficiency_reward
    assert reward(0) == 0.0
    assert reward(1) == 0.0
    assert reward(2) == -1.0
    assert reward(3) == -2.0
    assert reward(4) == -3.0


def test_search_efficiency_is_independently_normalized_per_response_depth():
    # One search is free (0); every later search increases the terminal
    # penalty. Only rollouts reaching the same response depth are siblings.
    values = torch.tensor([[0.0, -1.0], [-1.0, -3.0], [-2.0, 0.0]])
    masks = torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.bool)
    boundaries = masks.clone()
    subtasks = torch.zeros_like(values, dtype=torch.long)
    depths = torch.tensor([[1, 2], [1, 2], [1, -1]])
    advantages, _, metrics = compute_simulated_user_sparse_grpo_advantage(
        component_values={"search_efficiency": values},
        component_masks={"search_efficiency": masks},
        turn_boundary_mask=boundaries,
        subtask_ids=subtasks,
        response_depths=depths,
        response_mask=torch.ones_like(values),
        index=np.array(["d", "d", "d"], dtype=object),
        gamma=1.0,
    )

    # Depth 1: 0, -1, -2 -> trajectories using fewer searches are preferred.
    # The depth-2 normalized return [1, -1] is credited to the earlier action.
    assert torch.allclose(
        advantages[:, 0], torch.tensor([2.2247448, -1.0, -1.2247448]), atol=1e-5
    )
    # Depth 2 only includes the first two trajectories.
    assert torch.allclose(advantages[:2, 1], torch.tensor([1.0, -1.0]), atol=1e-5)
    assert advantages[2, 1] == 0.0
    assert metrics["sim_user/search_efficiency_groups"] == 2.0


def test_sparse_return_does_not_cross_original_subtask_boundary():
    values = torch.tensor([[0.0, 5.0], [0.0, 1.0]])
    masks = torch.tensor([[0, 1], [0, 1]], dtype=torch.bool)
    boundaries = torch.ones_like(masks)
    subtasks = torch.tensor([[0, 1], [0, 1]])
    depths = torch.ones_like(subtasks)
    advantages, _, _ = _sparse_adv(values, masks, boundaries, subtasks, depths)

    # The earlier subtask has no reward. The later reward must not leak across
    # the original user-task boundary into column zero.
    assert torch.equal(advantages[:, 0], torch.zeros(2))
    assert torch.allclose(advantages[:, 1], torch.tensor([1.0, -1.0]), atol=1e-5)


def test_event_rows_propagate_terminal_reward_to_earlier_tool_action():
    # Two sibling rollouts of one subtask. Each has a tool-call event (order
    # 0) followed by a terminal action reward (order 1). Exact prompt snapshots
    # may differ across rows, but the terminal normalized reward must still
    # return to its own preceding tool action.
    values = torch.tensor([[0.0], [1.0], [0.0], [3.0]])
    masks = torch.tensor([[0], [1], [0], [1]], dtype=torch.bool)
    boundaries = torch.ones_like(masks)
    subtasks = torch.zeros_like(values, dtype=torch.long)
    depths = torch.ones_like(subtasks)
    advantages, _, _ = compute_simulated_user_sparse_grpo_advantage(
        component_values={"action": values},
        component_masks={"action": masks},
        turn_boundary_mask=boundaries,
        subtask_ids=subtasks,
        response_depths=depths,
        response_mask=torch.ones_like(values),
        index=np.array(["d", "d", "d", "d"], dtype=object),
        rollout_ids=np.array([0, 0, 1, 1], dtype=np.int64),
        event_orders=torch.tensor([0, 1, 0, 1]),
        gamma=1.0,
    )
    assert torch.allclose(advantages[:, 0], torch.tensor([-1.0, -1.0, 1.0, 1.0]), atol=1e-5)


def test_event_rows_ignore_zero_masked_dispatch_padding():
    # DP dispatch may append copied rows only to make the global action-row
    # count divisible by the number of GPUs. They have no action boundary and
    # a zero response mask, so they must not alter a GRPO group or cause an
    # event-chain validation error.
    values = torch.tensor([[1.0], [3.0], [0.0]])
    masks = torch.tensor([[1], [1], [0]], dtype=torch.bool)
    boundaries = torch.tensor([[1], [1], [0]], dtype=torch.bool)
    subtasks = torch.tensor([[0], [0], [-1]])
    depths = torch.tensor([[1], [1], [-1]])
    advantages, _, metrics = compute_simulated_user_sparse_grpo_advantage(
        component_values={"action": values},
        component_masks={"action": masks},
        turn_boundary_mask=boundaries,
        subtask_ids=subtasks,
        response_depths=depths,
        response_mask=torch.tensor([[1.0], [1.0], [0.0]]),
        index=np.array(["d", "d", "padding"], dtype=object),
        rollout_ids=np.array([0, 1, -1], dtype=np.int64),
        event_orders=torch.tensor([0, 0, -1]),
        gamma=1.0,
    )
    assert torch.allclose(advantages[:, 0], torch.tensor([-1.0, 1.0, 0.0]), atol=1e-5)
    assert metrics["sim_user/events"] == 2.0
