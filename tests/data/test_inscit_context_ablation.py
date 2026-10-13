from __future__ import annotations

import json
import importlib.util
import logging
from pathlib import Path
import sys
from types import MethodType

import pandas as pd
import pytest
import requests

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = load_script("igpo_inscit_context_builder", "scripts/build_inscit_context_ablation.py")
SPLITTER = load_script("igpo_inscit_context_splitter", "scripts/prepare_inscit_context_ablation_splits.py")
SUMMARIZER = load_script("igpo_inscit_context_summarizer", "scripts/summarize_inscit_context_ablation.py")

from igpo_inscit_context_builder import (  # noqa: E402
    PermanentExternalError,
    LocalQwenDynamicBackend,
    ModelFormatError,
    OpenAIDynamicBackend,
    RetryContext,
    RetryController,
    direct_answer_turns,
    parse_args,
    process_dialogue,
    run,
    validate_pairs,
)
from igpo_inscit_context_splitter import split_paired_frames  # noqa: E402


def label(response_type: str, response: str) -> dict:
    return {
        "responseType": response_type,
        "response": response,
        "evidence": [{"passage_id": f"passage:{response}"}],
    }


def turn(question: str, labels: list[dict]) -> dict:
    return {"context": [question], "labels": labels, "prevEvidence": []}


class FakeBackend:
    def __init__(self, dependencies: dict[str, dict] | None = None) -> None:
        self.dependencies = dependencies or {}
        self.generated = []

    def generate_turn(self, *, public_history, current_query, context, max_feedback_rounds):
        history = [dict(value) for value in public_history]
        history.extend(
            [
                {"role": "User", "content": current_query},
                {"role": "Assistant", "content": "off topic"},
                {"role": "User", "content": "Please answer my original question directly."},
                {"role": "Assistant", "content": f"dynamic answer for {current_query}"},
            ]
        )
        self.generated.append(current_query)
        return {
            "public_history": history,
            "final_answer": f"dynamic answer for {current_query}",
            "attempts": [
                {
                    "answer": "off topic",
                    "judgement": {
                        "addresses_query": False,
                        "feedback": "Please answer my original question directly.",
                        "reason": "off topic",
                    },
                },
                {"answer": f"dynamic answer for {current_query}"},
            ],
            "accepted_unjudged": False,
        }

    def check_dependency(self, *, next_query, prior_gold_answers, dynamic_history, context):
        return self.dependencies.get(
            next_query,
            {
                "depends_on_prior_gold": False,
                "required_turn_indices": [],
                "dynamic_history_sufficient": True,
                "reason": "independent",
            },
        )


@pytest.mark.parametrize(
    ("judgements", "expected_feedback_rounds", "accepted_unjudged"),
    [
        ([True], 0, False),
        ([False, True], 1, False),
        ([False, False], 2, True),
    ],
)
def test_generation_loop_supports_zero_one_or_two_feedback_rounds(
    judgements: list[bool], expected_feedback_rounds: int, accepted_unjudged: bool
) -> None:
    backend = object.__new__(OpenAIDynamicBackend)
    answers = iter(["first", "second", "third"])
    decisions = iter(judgements)

    def one_answer(_self, messages, context, answer_index):
        answer = next(answers)
        return {"search_query": f"query-{answer_index}", "retrieved_passages": [], "answer": answer}

    def judge(_self, **kwargs):
        addresses_query = next(decisions)
        return {
            "addresses_query": addresses_query,
            "feedback": "Please answer the original query directly." if not addresses_query else "",
            "reason": "scripted",
        }

    backend._one_answer = MethodType(one_answer, backend)
    backend._judge = MethodType(judge, backend)
    result = backend.generate_turn(
        public_history=[],
        current_query="question",
        context=RetryContext("train", "dialogue", 0, "system_answer"),
        max_feedback_rounds=2,
    )
    feedback_messages = [
        message for message in result["public_history"] if message["role"] == "User"
    ][1:]
    assert len(feedback_messages) == expected_feedback_rounds
    assert result["accepted_unjudged"] is accepted_unjudged


def test_local_qwen_answer_uses_one_programmatic_retrieval() -> None:
    backend = object.__new__(LocalQwenDynamicBackend)
    backend.model = "qwen25-32b-inscit-context-builder"
    backend.base_url = "http://127.0.0.1:8011/v1"
    backend.format_retries = 2
    backend.max_output_tokens = 256
    outputs = iter(
        [
            '```json\n{"query": "standalone fungus question"}\n```',
            "<answer>Rhizopus is used.</answer>",
        ]
    )
    retrieval_queries: list[str] = []

    def cached(_self, *, kind, payload, context, request):
        return request()

    def chat(_self, **kwargs):
        return next(outputs)

    def retrieve(_self, query, context):
        retrieval_queries.append(query)
        return [{"passage_id": "p1", "passage_text": "Rhizopus is used."}]

    backend._cached_openai = MethodType(cached, backend)
    backend._chat = MethodType(chat, backend)
    backend._retrieve = MethodType(retrieve, backend)
    result = backend._one_answer(
        [{"role": "User", "content": "What fungus is used?"}],
        RetryContext("train", "dialogue", 0, "system_answer"),
        0,
    )
    assert retrieval_queries == ["standalone fungus question"]
    assert result["answer"] == "Rhizopus is used."
    assert result["retrieved_passages"][0]["passage_id"] == "p1"


def test_local_qwen_provider_selects_local_default_model() -> None:
    args = parse_args(["--provider", "local-qwen"])
    assert args.model == "qwen25-32b-inscit-context-builder"


def test_answer_prefix_accepts_mixed_label_and_truncates() -> None:
    retained, dropped = direct_answer_turns(
        "dialogue",
        [
            turn(
                "q1",
                [
                    label("clarification", "clarify"),
                    label("directAnswer", ""),
                    label("directAnswer", "a1"),
                ],
            ),
            turn("q2", [label("directAnswer", "a2")]),
            turn("q3", [label("noAnswerButRelevantInfo", "none")]),
            turn("q4", [label("directAnswer", "a4")]),
        ],
        split="train",
    )
    assert [value.question for value in retained] == ["q1", "q2"]
    assert retained[0].primary_answer == "a1"
    assert [value["source_turn_index"] for value in dropped] == [2, 3]


def test_first_nonanswer_drops_entire_conversation() -> None:
    retained, dropped = direct_answer_turns(
        "dialogue",
        [turn("q1", [label("clarification", "clarify")]), turn("q2", [label("directAnswer", "a2")])],
        split="test",
    )
    assert retained == []
    assert len(dropped) == 2


def test_dependency_failure_truncates_both_views_after_current_turn() -> None:
    dialogue = {
        "turns": [
            turn("q1", [label("directAnswer", "a1"), label("directAnswer", "alternative a1")]),
            turn("q2", [label("directAnswer", "a2")]),
            turn("q3", [label("directAnswer", "a3")]),
        ]
    }
    backend = FakeBackend(
        {
            "q3": {
                "depends_on_prior_gold": True,
                "required_turn_indices": [0],
                "dynamic_history_sufficient": False,
                "reason": "missing referent",
            }
        }
    )
    result = process_dialogue(
        dialogue_id="dialogue",
        dialogue=dialogue,
        split="train",
        backend=backend,
        max_feedback_rounds=2,
        config_hash="hash",
    )
    assert len(result.static_rows) == len(result.dynamic_rows) == 2
    assert result.static_rows[0]["reward_model"]["ground_truth"][1]["response"] == "alternative a1"
    static_prompt = result.static_rows[1]["prompt"][0]["content"]
    assert "Assistant: a1" in static_prompt
    assert "Assistant: alternative a1" not in static_prompt
    assert result.dynamic_rows[1]["extra_info"]["dependency_class"] == "independent"
    dynamic_prompt = result.dynamic_rows[1]["prompt"][0]["content"]
    assert "off topic" in dynamic_prompt
    assert "Please answer my original question directly." in dynamic_prompt
    assert "dynamic answer for q1" in dynamic_prompt
    assert result.dropped[-1]["pair_id"] == "train:dialogue:2"
    validate_pairs(result.static_rows, result.dynamic_rows)


def test_dependency_sufficient_retains_next_turn_and_marks_subset() -> None:
    dialogue = {
        "turns": [
            turn("q1", [label("directAnswer", "a1")]),
            turn("q2", [label("directAnswer", "a2")]),
        ]
    }
    backend = FakeBackend(
        {
            "q2": {
                "depends_on_prior_gold": True,
                "required_turn_indices": [0],
                "dynamic_history_sufficient": True,
                "reason": "referent is present",
            }
        }
    )
    result = process_dialogue(
        dialogue_id="dialogue",
        dialogue=dialogue,
        split="train",
        backend=backend,
        max_feedback_rounds=2,
        config_hash="hash",
    )
    assert len(result.dynamic_rows) == 2
    assert result.dynamic_rows[1]["extra_info"]["dependency_class"] == "dependent_sufficient"


def test_transient_failures_retry_until_success() -> None:
    sleeps = []
    controller = RetryController(
        logger=logging.getLogger("retry-test"),
        max_transient_retries=0,
        sleep=sleeps.append,
        jitter=lambda: 0.5,
    )
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            raise requests.ConnectionError("temporary")
        return "ok"

    assert controller.run(operation, RetryContext("train", "d", 0, "system_answer")) == "ok"
    assert attempts == 4
    assert sleeps == [2.0, 4.0, 8.0]
    assert controller.summary()["system_answer"]["recoveries"] == 1
    assert controller.summary_for("train", "d")["system_answer"]["transient_failures"] == 3


def test_permanent_http_error_fails_without_retry() -> None:
    class Unauthorized(RuntimeError):
        status_code = 401

    controller = RetryController(
        logger=logging.getLogger("retry-test"),
        max_transient_retries=0,
        sleep=lambda _: pytest.fail("must not sleep"),
    )
    with pytest.raises(PermanentExternalError, match="status=401"):
        controller.run(
            lambda: (_ for _ in ()).throw(Unauthorized("bad key")),
            RetryContext("train", "d", 0, "system_answer"),
        )


def test_format_error_is_bounded_and_written_as_pending(tmp_path: Path) -> None:
    class BrokenBackend(FakeBackend):
        def generate_turn(self, **kwargs):
            raise ModelFormatError("malformed answer")

    payload = {
        "dialogue": {
            "turns": [
                turn("q1", [label("directAnswer", "a1")]),
                turn("q2", [label("directAnswer", "a2")]),
            ]
        }
    }
    train_path = tmp_path / "train.json"
    test_path = tmp_path / "test.json"
    train_path.write_text(json.dumps(payload), encoding="utf-8")
    test_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "output"
    args = parse_args(
        [
            "--train-input", str(train_path),
            "--test-input", str(test_path),
            "--output-dir", str(output),
            "--workers", "1",
            "--log-file", str(tmp_path / "builder.log"),
        ]
    )
    with pytest.raises(ModelFormatError, match="malformed answer"):
        run(args, backend=BrokenBackend())
    pending = json.loads(
        (output / "audit" / "train" / "pending" / "dialogue.json").read_text(encoding="utf-8")
    )
    assert pending["status"] == "pending_format_error"
    assert pending["config_hash"]


def test_paired_split_uses_identical_conversation_assignment() -> None:
    static_rows = []
    dynamic_rows = []
    for conversation in range(20):
        for turn_index in range(2):
            pair_id = f"train:c{conversation}:{turn_index}"
            for variant, rows in (("static", static_rows), ("dynamic", dynamic_rows)):
                rows.append(
                    {
                        "extra_info": {
                            "pair_id": pair_id,
                            "conversation_id": f"c{conversation}",
                            "context_variant": variant,
                        }
                    }
                )
    result, manifest = split_paired_frames(
        pd.DataFrame(static_rows), pd.DataFrame(dynamic_rows), holdout_fraction=0.2, seed=42
    )
    for split_index in (0, 1):
        static_ids = [value["pair_id"] for value in result["static"][split_index]["extra_info"]]
        dynamic_ids = [value["pair_id"] for value in result["dynamic"][split_index]["extra_info"]]
        assert static_ids == dynamic_ids
    assert manifest["paired_views"] is True


def test_end_to_end_fake_build_writes_paired_parquets(tmp_path: Path) -> None:
    payload = {
        "dialogue": {
            "turns": [
                turn("q1", [label("directAnswer", "a1")]),
                turn("q2", [label("directAnswer", "a2")]),
            ]
        }
    }
    train_path = tmp_path / "train.json"
    test_path = tmp_path / "test.json"
    train_path.write_text(json.dumps(payload), encoding="utf-8")
    test_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "output"
    args = parse_args(
        [
            "--train-input",
            str(train_path),
            "--test-input",
            str(test_path),
            "--output-dir",
            str(output),
            "--workers",
            "1",
            "--log-file",
            str(tmp_path / "builder.log"),
        ]
    )
    backend = FakeBackend()
    manifest = run(args, backend=backend)
    assert manifest["status"] == "complete"
    assert manifest["counts"]["train"]["rows"] == 2
    assert manifest["build_statistics"]["train"]["source_turns"] == 2
    assert manifest["build_statistics"]["train"]["feedback_rounds"] == 1
    static = pd.read_parquet(output / "static" / "inscit_train.parquet")
    dynamic = pd.read_parquet(output / "dynamic" / "inscit_train.parquet")
    assert len(static) == len(dynamic) == 2
    assert static.iloc[1]["extra_info"]["pair_id"] == dynamic.iloc[1]["extra_info"]["pair_id"]
    assert "dynamic answer for q1" in dynamic.iloc[1]["prompt"][0]["content"]
    ground_truth = static.iloc[0]["reward_model"]["ground_truth"]
    assert ground_truth[0]["action"] == "answer"
    assert ground_truth[0]["passage_id"] == ["passage:a1"]
    assert ground_truth[0]["evidence"][0]["passage_id"] == "passage:a1"

    resume_args = parse_args(
        [
            "--train-input", str(train_path),
            "--test-input", str(test_path),
            "--output-dir", str(output),
            "--workers", "1",
            "--log-file", str(tmp_path / "builder.log"),
            "--resume",
        ]
    )
    generated_before_resume = list(backend.generated)
    run(resume_args, backend=backend)
    assert backend.generated == generated_before_resume


def test_2x2_summarizer_preserves_pairing_and_subsets(tmp_path: Path) -> None:
    cell_dirs = {}
    offsets = {"ss": 0.0, "sd": -0.1, "ds": 0.1, "dd": 0.2}
    for cell, offset in offsets.items():
        directory = tmp_path / cell
        directory.mkdir()
        rows = []
        for turn_index, dependency_class in ((0, "first_turn"), (1, "independent"), (2, "dependent_sufficient")):
            score = 0.4 + 0.05 * turn_index + offset
            rows.append(
                {
                    "pair_id": f"test:dialogue:{turn_index}",
                    "turn_index": turn_index,
                    "dependency_class": dependency_class,
                    "f1": score,
                    "bertscore_f1": score,
                    "ndcg_at_3": score,
                    "action_accuracy": 1.0,
                    "format_valid": turn_index != 2,
                    "has_terminal_answer": True,
                }
            )
        (directory / "metrics_per_sample.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        cell_dirs[cell] = directory

    output = tmp_path / "summary"
    SUMMARIZER.main(
        [
            "--static-train-static-test", str(cell_dirs["ss"]),
            "--static-train-dynamic-test", str(cell_dirs["sd"]),
            "--dynamic-train-static-test", str(cell_dirs["ds"]),
            "--dynamic-train-dynamic-test", str(cell_dirs["dd"]),
            "--output-dir", str(output),
            "--bootstrap-samples", "20",
        ]
    )
    summary = json.loads((output / "context_ablation_2x2.json").read_text(encoding="utf-8"))
    assert summary["subsets"]["all"]["cells"]["ss"]["sample_count"] == 3
    assert summary["subsets"]["history_bearing"]["cells"]["ss"]["sample_count"] == 2
    assert summary["subsets"]["dependency_required"]["cells"]["ss"]["sample_count"] == 1
    contrast = summary["subsets"]["all"]["contrasts"]["dynamic_vs_static_train_on_static_test"]
    assert contrast["f1"]["estimate"] == pytest.approx(0.1)
