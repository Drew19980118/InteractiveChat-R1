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

import torch
import json
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.utils.reward_score.ground_truth import (
    select_expected_action,
    select_static_convagent_expected_actions,
)
from verl.utils.reward_score.info_gain import extract_terminal_action


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
            if self.use_action_reward:
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
                        action_reward = 1.0 if action_correct else self.action_incorrect_reward
                else:
                    expected_action = select_expected_action(reward_model, data_source=data_source)
                    predicted_action, format_valid = extract_terminal_action(response_str)
                    if expected_action is not None:
                        action_correct = bool(format_valid and predicted_action == expected_action)
                        action_reward = 1.0 if action_correct else self.action_incorrect_reward
                # Preserve one slot per sample. ``None`` marks unsupported
                # InSCIt clarification examples and is ignored downstream.
                reward_extra_info["action_rewards"].append(action_reward)

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # info_gain_reward - add null check
            info_gain_reward = info_gain_rewards[i] if info_gain_rewards is not None else []

            score = self.compute_score(
                data_source=data_source,
                prompt_str = prompt_str,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                val_type=val_type,
                info_gain_reward=info_gain_reward,
                tokenizer=self.tokenizer,
                is_validation=is_validation,
                static_convagent_mode=self.static_convagent_mode,
            )

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
