# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import math
import re

import torch
import json
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.utils.reward_score.ground_truth import (
    select_expected_action,
    select_static_convagent_expected_actions,
)
from verl.utils.reward_score.info_gain import extract_terminal_action
from verl.utils.reward_score.igt import (
    IGTScore,
    IGTUserSimulatorScorer,
    extract_static_convagent_public_state,
)
from verl.utils.reward_score.sentence_bertscore import (
    SentenceBERTScore,
    SentenceBERTScoreReward,
)
from verl.utils.reward_score.answer_bertscore import (
    AnswerBERTScore,
    AnswerBERTScoreReward,
)
from verl.utils.reward_score.static_convagent import (
    static_convagent_answer_references,
    terminal_format_reward,
    token_set_f1,
)


def _last_assistant_answer(response: str) -> str:
    """Extract the final ``<answer>`` payload after validating the action."""
    action, format_valid = extract_terminal_action(
        response,
        allow_clarify=True,
        allow_search=True,
    )
    if not format_valid or action != "answer":
        return ""
    final_turn = response.rsplit("\n<|im_start|>assistant\n", 1)[-1]
    match = re.search(r"<answer>(.*?)</answer>", final_turn, re.DOTALL)
    return match.group(1).strip() if match else ""


def _max_finite(values) -> float:
    """Return a robust max for optional per-search reward vectors."""
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(finite) if finite else 0.0


class NaiveRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        use_action_reward: bool = False,
        static_convagent_mode: bool = False,
        static_chatr1_mode: bool = False,
        static_convagent_paper_reward: bool = False,
        static_convagent_retrieval_reward: bool = True,
        static_convagent_action_reward: bool = True,
        static_convagent_format_reward: bool = False,
        static_convagent_format_valid_reward: float = 0.1,
        static_convagent_format_invalid_reward: float = -0.1,
        static_convagent_igt_reward: bool = False,
        static_convagent_igt_base_url: str | None = None,
        static_convagent_igt_model: str | None = None,
        static_convagent_igt_timeout_seconds: int = 300,
        static_convagent_igt_batch_size: int = 32,
        static_convagent_igt_tau: float = 0.5,
        static_convagent_igt_reward_floor: float = 0.0,
        static_convagent_igt_reward_ceiling: float = 1.2,
        static_convagent_sentence_bertscore_reward: bool = False,
        static_convagent_sentence_bertscore_additive_f1: bool = False,
        static_convagent_sentence_bertscore_model: str = "roberta-large",
        static_convagent_sentence_bertscore_device: str = "cpu",
        static_convagent_sentence_bertscore_batch_size: int = 32,
        static_convagent_bertscore_reward: bool = False,
        static_convagent_bertscore_additive_f1: bool = False,
        static_convagent_bertscore_model: str = "roberta-large",
        static_convagent_bertscore_device: str = "cpu",
        static_convagent_bertscore_batch_size: int = 32,
        static_chatr1_paper_reward: bool = False,
        static_chatr1_intent_weight: float = 1.0,
        action_incorrect_reward: float = -1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        # Terminal-action supervision is an experiment-level switch. Both
        # baseline IGPO and Rewrite-Bound enable it for a fair comparison.
        self.use_action_reward = use_action_reward
        self.static_convagent_mode = static_convagent_mode
        self.static_chatr1_mode = static_chatr1_mode
        self.static_convagent_paper_reward = static_convagent_paper_reward
        self.static_convagent_retrieval_reward = static_convagent_retrieval_reward
        self.static_convagent_action_reward = static_convagent_action_reward
        self.static_convagent_format_reward = bool(static_convagent_format_reward)
        self.static_convagent_format_valid_reward = float(
            static_convagent_format_valid_reward
        )
        self.static_convagent_format_invalid_reward = float(
            static_convagent_format_invalid_reward
        )
        if self.static_convagent_format_reward and not (
            self.static_convagent_mode and self.static_convagent_paper_reward
        ):
            raise ValueError(
                "static_convagent_format_reward requires static_convagent_mode=true "
                "and static_convagent_paper_reward=true"
            )
        if not (
            math.isfinite(self.static_convagent_format_valid_reward)
            and math.isfinite(self.static_convagent_format_invalid_reward)
        ):
            raise ValueError("static ConvAgent format reward values must be finite")
        if self.static_convagent_format_valid_reward < 0.0:
            raise ValueError("static_convagent_format_valid_reward must be >= 0")
        if self.static_convagent_format_invalid_reward > 0.0:
            raise ValueError("static_convagent_format_invalid_reward must be <= 0")
        self.static_convagent_igt_reward = bool(static_convagent_igt_reward)
        if self.static_convagent_igt_reward and not (
            self.static_convagent_mode and self.static_convagent_paper_reward
        ):
            raise ValueError(
                "static_convagent_igt_reward requires static_convagent_mode=true "
                "and static_convagent_paper_reward=true"
            )
        if self.static_convagent_igt_reward and self.static_convagent_retrieval_reward:
            raise ValueError(
                "IGT ConvAgent replaces answer F1 and must run with "
                "static_convagent_retrieval_reward=false"
            )
        self.static_convagent_igt_scorer = (
            IGTUserSimulatorScorer(
                tokenizer=tokenizer,
                base_url=str(static_convagent_igt_base_url or ""),
                model=str(static_convagent_igt_model or ""),
                timeout_seconds=static_convagent_igt_timeout_seconds,
                batch_size=static_convagent_igt_batch_size,
                tau=static_convagent_igt_tau,
                reward_floor=static_convagent_igt_reward_floor,
                reward_ceiling=static_convagent_igt_reward_ceiling,
            )
            if self.static_convagent_igt_reward
            else None
        )
        self.static_convagent_sentence_bertscore_reward = bool(
            static_convagent_sentence_bertscore_reward
        )
        self.static_convagent_sentence_bertscore_additive_f1 = bool(
            static_convagent_sentence_bertscore_additive_f1
        )
        if (
            self.static_convagent_sentence_bertscore_additive_f1
            and not self.static_convagent_sentence_bertscore_reward
        ):
            raise ValueError(
                "static_convagent_sentence_bertscore_additive_f1 requires "
                "static_convagent_sentence_bertscore_reward=true"
            )
        if self.static_convagent_sentence_bertscore_reward and not (
            self.static_convagent_mode and self.static_convagent_paper_reward
        ):
            raise ValueError(
                "static_convagent_sentence_bertscore_reward requires "
                "static_convagent_mode=true and static_convagent_paper_reward=true"
            )
        if self.static_convagent_sentence_bertscore_reward and self.static_convagent_igt_reward:
            raise ValueError(
                "Sentence BERTScore and IGT both replace the ConvAgent answer-F1 outcome; "
                "enable exactly one of them"
            )
        self.static_convagent_sentence_bertscore_scorer = (
            SentenceBERTScoreReward(
                model=static_convagent_sentence_bertscore_model,
                device=static_convagent_sentence_bertscore_device,
                batch_size=static_convagent_sentence_bertscore_batch_size,
            )
            if self.static_convagent_sentence_bertscore_reward
            else None
        )
        self.static_convagent_bertscore_reward = bool(static_convagent_bertscore_reward)
        self.static_convagent_bertscore_additive_f1 = bool(
            static_convagent_bertscore_additive_f1
        )
        if (
            self.static_convagent_bertscore_additive_f1
            and not self.static_convagent_bertscore_reward
        ):
            raise ValueError(
                "static_convagent_bertscore_additive_f1 requires "
                "static_convagent_bertscore_reward=true"
            )
        if self.static_convagent_bertscore_reward and not (
            self.static_convagent_mode and self.static_convagent_paper_reward
        ):
            raise ValueError(
                "static_convagent_bertscore_reward requires static_convagent_mode=true "
                "and static_convagent_paper_reward=true"
            )
        if self.static_convagent_bertscore_reward and (
            self.static_convagent_igt_reward
            or self.static_convagent_sentence_bertscore_reward
        ):
            raise ValueError(
                "choose exactly one static ConvAgent answer-outcome replacement: "
                "IGT, Sentence BERTScore, or whole-answer BERTScore"
            )
        self.static_convagent_bertscore_scorer = (
            AnswerBERTScoreReward(
                model=static_convagent_bertscore_model,
                device=static_convagent_bertscore_device,
                batch_size=static_convagent_bertscore_batch_size,
            )
            if self.static_convagent_bertscore_reward
            else None
        )
        self.static_chatr1_paper_reward = static_chatr1_paper_reward
        self.static_chatr1_intent_weight = float(static_chatr1_intent_weight)
        self.action_incorrect_reward = float(action_incorrect_reward)

    def __call__(self, data: DataProto, return_dict=False, val_type='f1', info_gain_rewards=None, is_validation=False):
        """We will expand this function gradually based on the available datasets"""
        data_str = str(data)
        if is_validation:
            f1_scores = []
            em_scores = []
            noformatf1_scores = []
            expected_actions = []
            predicted_actions = []
            action_corrects = []
            format_valids = []
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        # IGT candidates are deliberately collected over the whole n=8 group
        # and scored in batches. A per-sample HTTP call would make the frozen
        # simulator the bottleneck and would unnecessarily repeat the same
        # gold-reference baseline eight times.
        pending_static_igt: list[dict] = []
        static_igt_action_contributions = [0.0 for _ in range(len(data))]
        static_igt_diagnostics: dict[int, IGTScore] = {}
        # Sentence BERTScore is another frozen outcome scorer.  Its records
        # are collected first so all sentence pairs in this rollout can be
        # evaluated in one batched BERTScore call after rollout decoding.
        pending_static_sentence_bertscore: list[dict] = []
        static_sentence_bertscore_diagnostics: dict[int, SentenceBERTScore] = {}
        # Whole-answer BERTScore differs from sentence coverage: one answer is
        # compared against each released reference and the best reference wins.
        pending_static_answer_bertscore: list[dict] = []
        static_answer_bertscore_diagnostics: dict[int, AnswerBERTScore] = {}
        # Keep one format value per sampled trajectory so W&B can distinguish
        # the explicit format component from the paper MIA action reward.
        static_format_rewards = [0.0 for _ in range(len(data))]
        static_format_valids = [0.0 for _ in range(len(data))]

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            reward_model = data_item.non_tensor_batch["reward_model"]
            ground_truth = reward_model["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            expected_action = None
            predicted_action = None
            format_valid = None
            action_reward = None
            action_correct = None
            # Paper ConvAgent forms MIA in the raw trajectory reward.  It is
            # not a separately normalized component and is available only for
            # InsCiT, the dataset carrying mixed-initiative action labels.
            needs_action_parse = self.use_action_reward or (
                self.static_convagent_mode and self.static_convagent_paper_reward
            )
            if needs_action_parse:
                if self.static_convagent_mode:
                    expected_action = sorted(
                        select_static_convagent_expected_actions(reward_model, data_source=data_source)
                    )
                    predicted_action, format_valid = extract_terminal_action(
                        response_str,
                        allow_clarify=True,
                        allow_search=True,
                    )
                    if expected_action:
                        action_correct = bool(format_valid and predicted_action in expected_action)
                        if self.static_convagent_paper_reward:
                            # ConvAgent Eq. (2) defines MIA as +1 for a
                            # permitted action and -0.5 otherwise.  In this
                            # strict action grammar a malformed completion is
                            # an incorrect action, not a reward-neutral one.
                            # Leaving it at zero creates a GRPO exploit:
                            # malformed text outranks a valid-but-wrong action
                            # (-0.5), which can reinforce degenerate loops.
                            action_reward = 1.0 if action_correct else -0.5
                        else:
                            action_reward = 1.0 if action_correct else self.action_incorrect_reward
                else:
                    expected_action = select_expected_action(reward_model, data_source=data_source)
                    predicted_action, format_valid = extract_terminal_action(response_str)
                    if expected_action is not None:
                        action_correct = bool(format_valid and predicted_action == expected_action)
                        action_reward = 1.0 if action_correct else self.action_incorrect_reward
                # Legacy static ConvAgent supplies this independent channel to
                # separate-component GRPO. Paper mode has already folded MIA
                # into the terminal trajectory reward below.
                if self.use_action_reward and not self.static_convagent_paper_reward:
                    reward_extra_info["action_rewards"].append(action_reward)
            elif self.static_chatr1_mode:
                # ChatR1 has no action-supervision reward.  Parse its final
                # answer/search form only for validation health reporting;
                # this must not alter the paper's zero format-reward setting.
                predicted_action, format_valid = extract_terminal_action(
                    response_str,
                    allow_search=True,
                )

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # info_gain_reward - add null check
            info_gain_reward = info_gain_rewards[i] if info_gain_rewards is not None else []

            # The original ChatR1 reward is trajectory-level, so query F1 must
            # not be placed at intermediate search-token boundaries. The
            # original ConvAgent total is likewise formed before GRPO.
            score_info_gain_reward = (
                []
                if self.static_convagent_paper_reward or self.static_chatr1_paper_reward
                else info_gain_reward
            )
            score = self.compute_score(
                data_source=data_source,
                prompt_str = prompt_str,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                val_type=val_type,
                info_gain_reward=score_info_gain_reward,
                tokenizer=self.tokenizer,
                is_validation=is_validation,
                static_convagent_mode=self.static_convagent_mode,
                static_chatr1_mode=self.static_chatr1_mode,
            )

            if not is_validation and self.static_convagent_paper_reward:
                # R_outcome is answer-only max-F1 by default. The optional
                # sentence-level BERTScore coverage reward replaces just this
                # outcome term; clarify/nonanswer never receive text reward.
                answer = _last_assistant_answer(response_str)
                references = static_convagent_answer_references(ground_truth)
                use_igt_outcome = bool(self.static_convagent_igt_reward)
                use_sentence_bertscore_outcome = bool(
                    self.static_convagent_sentence_bertscore_reward
                )
                add_f1_to_sentence_bertscore = bool(
                    use_sentence_bertscore_outcome
                    and self.static_convagent_sentence_bertscore_additive_f1
                )
                use_answer_bertscore_outcome = bool(self.static_convagent_bertscore_reward)
                add_f1_to_answer_bertscore = bool(
                    use_answer_bertscore_outcome
                    and self.static_convagent_bertscore_additive_f1
                )
                outcome_reward = (
                    max(token_set_f1(answer, reference) for reference in references)
                    if not (
                        use_igt_outcome
                        or (
                            use_answer_bertscore_outcome
                            and not add_f1_to_answer_bertscore
                        )
                        or (
                            use_sentence_bertscore_outcome
                            and not add_f1_to_sentence_bertscore
                        )
                    )
                    and answer
                    and references
                    else 0.0
                )
                # The retrieval ablation retains the native tool interaction
                # and all retrieval evaluation, but removes R_IG from the
                # learning signal.  R_outcome and (on InsCiT) R_MIA remain
                # exactly as in the paper reward.
                information_gain_reward = (
                    _max_finite(info_gain_reward)
                    if self.static_convagent_retrieval_reward
                    else 0.0
                )
                mia_reward = (
                    float(action_reward or 0.0)
                    if self.static_convagent_action_reward
                    and str(data_source).strip().lower() == "inscit"
                    else 0.0
                )
                format_reward = (
                    terminal_format_reward(
                        bool(format_valid),
                        valid_reward=self.static_convagent_format_valid_reward,
                        invalid_reward=self.static_convagent_format_invalid_reward,
                    )
                    if self.static_convagent_format_reward
                    else 0.0
                )
                static_format_rewards[i] = float(format_reward)
                static_format_valids[i] = float(bool(format_valid))
                score = [0.0] * len(score)
                if score:
                    action_and_retrieval = 0.5 * (information_gain_reward + mia_reward)
                    if use_igt_outcome:
                        static_igt_action_contributions[i] = float(
                            action_and_retrieval + format_reward
                        )
                        # Only a format-valid, permissible terminal answer is
                        # eligible for IGT. A semantically plausible answer
                        # cannot compensate for a wrong/malformed action.
                        if bool(action_correct) and predicted_action == "answer" and answer and references:
                            history, query = extract_static_convagent_public_state(prompt_str)
                            pending_static_igt.append(
                                {
                                    "row_index": i,
                                    "history": history,
                                    "query": query,
                                    "predicted_answer": answer,
                                    "references": references,
                                }
                            )
                        score[-1] = float(action_and_retrieval + format_reward)
                    elif use_sentence_bertscore_outcome:
                        # Preserve the paper's action/retrieval terms and add
                        # sentence coverage after one batched frozen BERTScore
                        # pass.  Unlike IGT, this directly mirrors the native
                        # F1 branch: any terminal <answer> is text-scored, and
                        # an impermissible action remains penalized by MIA.
                        if answer and references:
                            pending_static_sentence_bertscore.append(
                                {
                                    "row_index": i,
                                    "predicted_answer": answer,
                                    "references": references,
                                }
                            )
                        score[-1] = float(
                            outcome_reward + action_and_retrieval + format_reward
                        )
                    elif use_answer_bertscore_outcome:
                        # Whole-answer BERTScore is the strict replacement of
                        # native max-F1: action and retrieval composition is
                        # unchanged, while the answer outcome is added after a
                        # batched frozen scorer pass.
                        if answer and references:
                            pending_static_answer_bertscore.append(
                                {
                                    "row_index": i,
                                    "predicted_answer": answer,
                                    "references": references,
                                }
                            )
                        score[-1] = float(
                            outcome_reward + action_and_retrieval + format_reward
                        )
                    else:
                        score[-1] = float(
                            outcome_reward + action_and_retrieval + format_reward
                        )

            if not is_validation and self.static_chatr1_paper_reward:
                # ChatR1 Appendix D/E: R = answer-F1 + alpha * max query-F1,
                # assigned only at the terminal trajectory token for PPO/GAE.
                if score:
                    score[-1] += self.static_chatr1_intent_weight * _max_finite(info_gain_reward)

            if is_validation:
                f1_scores.append(score['f1'])
                em_scores.append(score['em'])
                noformatf1_scores.append(score['noformatf1'])
                expected_actions.append(expected_action)
                predicted_actions.append(predicted_action)
                action_corrects.append(action_correct)
                format_valids.append(bool(format_valid))
                reward_tensor[i, :valid_response_length] = torch.tensor(score['scores'])
            else:
                reward_tensor[i, :valid_response_length] = torch.tensor(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and val_type == 'f1':
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[data_source]", data_source, "[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    # Validation mode: score is dict
                    for key, value in score.items():
                        if key != 'scores':  # Skip verbose token-level scores
                            print(f"[{key}]", value)
                else:
                    # Training mode: score is list (token-level rewards)
                    # Only print non-zero count and last value (usually F1 score)
                    if isinstance(score, list) and len(score) > 0:
                        non_zero_count = sum(1 for s in score if s != 0)
                        last_value = score[-1] if score else 0
                        print(f"[score] {non_zero_count} non-zero rewards, final={last_value:.4f}")
                    else:
                        print("[score]", score)
                
                # Print turn count and info_gain_reward (for both training and validation)
                if info_gain_reward:
                    num_turns = len(info_gain_reward) + 1
                    print(f"[turns]", num_turns)
                    print(f"[info_gain_reward]", info_gain_reward)

        if pending_static_igt:
            assert self.static_convagent_igt_scorer is not None
            static_igt_diagnostics = self.static_convagent_igt_scorer.score_records(pending_static_igt)
            for row_index, diagnostic in static_igt_diagnostics.items():
                prompt_ids = data[row_index].batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_response_length = int(
                    data[row_index].batch["attention_mask"][prompt_length:].sum().item()
                )
                if valid_response_length < 1:
                    raise RuntimeError("IGT received a rollout with no valid response tokens")
                reward_tensor[row_index, valid_response_length - 1] = float(
                    diagnostic.reward + static_igt_action_contributions[row_index]
                )
            igt_rewards = [value.reward for value in static_igt_diagnostics.values()]
            igt_gaps = [value.gap for value in static_igt_diagnostics.values()]
            print(
                "[StaticConvAgent IGT] "
                f"eligible={len(static_igt_diagnostics)}/{len(data)} "
                f"reward_mean={sum(igt_rewards) / len(igt_rewards):.4f} "
                f"gap_mean={sum(igt_gaps) / len(igt_gaps):.4f}",
                flush=True,
            )

        if pending_static_sentence_bertscore:
            assert self.static_convagent_sentence_bertscore_scorer is not None
            static_sentence_bertscore_diagnostics = (
                self.static_convagent_sentence_bertscore_scorer.score_records(
                    pending_static_sentence_bertscore
                )
            )
            for row_index, diagnostic in static_sentence_bertscore_diagnostics.items():
                prompt_ids = data[row_index].batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_response_length = int(
                    data[row_index].batch["attention_mask"][prompt_length:].sum().item()
                )
                if valid_response_length < 1:
                    raise RuntimeError(
                        "Sentence BERTScore received a rollout with no valid response tokens"
                    )
                reward_tensor[row_index, valid_response_length - 1] += float(
                    diagnostic.reward
                )
            sentence_rewards = [
                value.reward for value in static_sentence_bertscore_diagnostics.values()
            ]
            print(
                "[StaticConvAgent SentenceBERTScore] "
                f"eligible={len(static_sentence_bertscore_diagnostics)}/{len(data)} "
                f"coverage_mean={sum(sentence_rewards) / len(sentence_rewards):.4f}",
                flush=True,
            )

        if pending_static_answer_bertscore:
            assert self.static_convagent_bertscore_scorer is not None
            static_answer_bertscore_diagnostics = (
                self.static_convagent_bertscore_scorer.score_records(
                    pending_static_answer_bertscore
                )
            )
            for row_index, diagnostic in static_answer_bertscore_diagnostics.items():
                prompt_ids = data[row_index].batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_response_length = int(
                    data[row_index].batch["attention_mask"][prompt_length:].sum().item()
                )
                if valid_response_length < 1:
                    raise RuntimeError("Answer BERTScore received a rollout with no valid response tokens")
                reward_tensor[row_index, valid_response_length - 1] += float(diagnostic.reward)
            answer_rewards = [value.reward for value in static_answer_bertscore_diagnostics.values()]
            print(
                "[StaticConvAgent AnswerBERTScore] "
                f"eligible={len(static_answer_bertscore_diagnostics)}/{len(data)} "
                f"reward_mean={sum(answer_rewards) / len(answer_rewards):.4f}",
                flush=True,
            )

        if self.static_convagent_igt_reward and not is_validation:
            # Persist one value for every sampled response, including invalid
            # actions. This makes W&B rates and score distributions auditable.
            for row_index in range(len(data)):
                diagnostic = static_igt_diagnostics.get(row_index)
                reward_extra_info["static_convagent_igt_reward"].append(
                    float(diagnostic.reward) if diagnostic is not None else 0.0
                )
                reward_extra_info["static_convagent_igt_gap"].append(
                    float(diagnostic.gap) if diagnostic is not None else float("nan")
                )
                reward_extra_info["static_convagent_igt_gold_logprob"].append(
                    float(diagnostic.gold_logprob) if diagnostic is not None else float("nan")
                )
                reward_extra_info["static_convagent_igt_predicted_logprob"].append(
                    float(diagnostic.predicted_logprob) if diagnostic is not None else float("nan")
                )
                reward_extra_info["static_convagent_igt_eligible"].append(
                    float(diagnostic is not None)
                )

        if self.static_convagent_sentence_bertscore_reward and not is_validation:
            # Write one row per sampled response so W&B exposes both semantic
            # coverage and eligibility rates rather than only the mixed final
            # reward. Invalid/no-answer actions receive an explicit zero.
            for row_index in range(len(data)):
                diagnostic = static_sentence_bertscore_diagnostics.get(row_index)
                reward_extra_info["static_convagent_sentence_bertscore_reward"].append(
                    float(diagnostic.reward) if diagnostic is not None else 0.0
                )
                reward_extra_info[
                    "static_convagent_sentence_bertscore_predicted_sentences"
                ].append(
                    float(diagnostic.predicted_sentence_count)
                    if diagnostic is not None
                    else 0.0
                )
                reward_extra_info[
                    "static_convagent_sentence_bertscore_gold_sentences"
                ].append(
                    float(diagnostic.gold_sentence_count)
                    if diagnostic is not None
                    else 0.0
                )
                reward_extra_info[
                    "static_convagent_sentence_bertscore_references_scored"
                ].append(
                    float(diagnostic.references_scored)
                    if diagnostic is not None
                    else 0.0
                )
                reward_extra_info["static_convagent_sentence_bertscore_eligible"].append(
                    float(diagnostic is not None)
                )

        if self.static_convagent_bertscore_reward and not is_validation:
            for row_index in range(len(data)):
                diagnostic = static_answer_bertscore_diagnostics.get(row_index)
                reward_extra_info["static_convagent_bertscore_reward"].append(
                    float(diagnostic.reward) if diagnostic is not None else 0.0
                )
                reward_extra_info["static_convagent_bertscore_references_scored"].append(
                    float(diagnostic.references_scored) if diagnostic is not None else 0.0
                )
                reward_extra_info["static_convagent_bertscore_eligible"].append(
                    float(diagnostic is not None)
                )

        if self.static_convagent_format_reward and not is_validation:
            reward_extra_info["static_convagent_format_reward"].extend(
                static_format_rewards
            )
            reward_extra_info["static_convagent_format_valid"].extend(
                static_format_valids
            )

        if is_validation:
            return {
                "f1_scores": f1_scores,
                "em_scores": em_scores,
                "noformatf1_scores": noformatf1_scores,
                "expected_actions": expected_actions,
                "predicted_actions": predicted_actions,
                "action_corrects": action_corrects,
                "format_valids": format_valids,
                "reward_tensor": reward_tensor,
            }
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
