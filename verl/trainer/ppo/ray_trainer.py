# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
import glob
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Iterator, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, RandomSampler, Sampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.simulated_user_algos import compute_simulated_user_sparse_grpo_advantage
from verl.trainer.ppo.simulated_user_validation import validate_simulated_user
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.reward_score.ground_truth import (
    select_answer_ground_truth_with_passage_ids,
    select_static_chatr1_answer_ground_truth_with_passage_ids,
    select_static_convagent_answer_ground_truth_with_passage_ids,
)
from verl.utils.reward_score.static_convagent import monitor_plateau_reached
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from scrl.llm_agent.generation import LLMGenerationManager, GenerationConfig
from scrl.llm_agent.simulated_user import parse_dialogue_payload
from scrl.llm_agent.simulated_user_rollout import SimulatedUserGenerationManager, SimulatedUserSettings
from tools_server.util import MessageClient

WorkerType = Type[Worker]


class SimulatedUserExactContextBatchSampler(Sampler[list[int]]):
    """Pack whole dialogues into exact original-subtask context batches.

    A dialogue is indivisible at collection time, so every group of its eight
    sibling rollouts is fully scored before an actor update.  The sampler uses
    known canonical sub-task counts to pack complete dialogues to exactly the
    requested context count *before* online collection.  This is preferable to
    collecting an overflowing dialogue, updating once, and training stale
    leftovers after the policy has changed.
    """

    def __init__(
        self,
        context_counts: list[int],
        *,
        target_contexts: int,
        num_batches: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if target_contexts < 1:
            raise ValueError("simulated-user context batch size must be >= 1")
        if num_batches < 1:
            raise ValueError("simulated-user packed sampler needs at least one batch")
        if not context_counts:
            raise ValueError("simulated-user packed sampler received no dialogues")
        invalid = [count for count in context_counts if count < 1 or count > target_contexts]
        if invalid:
            raise ValueError(
                "every simulated-user dialogue must contain between 1 and the target number "
                f"of sub-tasks; target={target_contexts}, invalid_counts={invalid[:8]}"
            )
        self.context_counts = [int(count) for count in context_counts]
        self.target_contexts = int(target_contexts)
        self.num_batches = int(num_batches)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

    @staticmethod
    def _exact_subset_positions(
        indices: list[int], counts: list[int], target: int
    ) -> list[int] | None:
        """Return positions whose whole-dialogue counts sum exactly to target."""
        # predecessor[sum] = (previous_sum, position_in_indices).  The target
        # is 128 in the intended run, hence this O(num_dialogues * target)
        # dynamic program is negligible compared with one online rollout.
        predecessor: list[tuple[int, int] | None] = [None] * (target + 1)
        predecessor[0] = (-1, -1)
        for position, index in enumerate(indices):
            count = counts[index]
            for subtotal in range(target - count, -1, -1):
                if predecessor[subtotal] is None or predecessor[subtotal + count] is not None:
                    continue
                predecessor[subtotal + count] = (subtotal, position)
        if predecessor[target] is None:
            return None
        positions: list[int] = []
        current = target
        while current:
            previous, position = predecessor[current]  # target is reachable.
            positions.append(position)
            current = previous
        positions.reverse()
        return positions

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        available = list(range(len(self.context_counts)))
        if self.shuffle:
            rng.shuffle(available)

        for _ in range(self.num_batches):
            positions = self._exact_subset_positions(
                available, self.context_counts, self.target_contexts
            )
            if positions is None:
                # The remaining tail of an epoch cannot make an exact target.
                # Carry it forward and replenish only dialogues not already in
                # ``available``. This preserves whole-dialogue packing without
                # ever placing two copies of the same dialogue (and thus 16
                # siblings with one uid) in a single GRPO update.
                available_set = set(available)
                fresh = [
                    index for index in range(len(self.context_counts))
                    if index not in available_set
                ]
                if self.shuffle:
                    rng.shuffle(fresh)
                available.extend(fresh)
                positions = self._exact_subset_positions(
                    available, self.context_counts, self.target_contexts
                )
            if positions is None:
                raise RuntimeError(
                    "could not pack complete simulated-user dialogues to exactly "
                    f"{self.target_contexts} sub-task contexts; inspect sub-task counts"
                )
            selected_positions = set(positions)
            selected = [available[position] for position in positions]
            if len(selected) != len(set(selected)):
                raise RuntimeError("simulated-user packed batch contains a duplicate dialogue")
            available = [index for position, index in enumerate(available) if position not in selected_positions]
            if sum(self.context_counts[index] for index in selected) != self.target_contexts:
                raise RuntimeError("simulated-user packed sampler produced an inexact context batch")
            yield selected

    def __len__(self) -> int:
        return self.num_batches

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self._epoch}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self._epoch = int(state_dict.get("epoch", 0))


class SimulatedUserValidationContextBatchSampler(Sampler[list[int]]):
    """Cover validation dialogues once in batches of at most N sub-tasks.

    Unlike the training sampler, validation must never replay a dialogue merely
    to make an exact total.  This deterministic knapsack packing keeps every
    dialogue whole, packs each batch as close as possible to the requested
    context budget, and emits one final smaller batch if necessary.
    """

    def __init__(self, context_counts: list[int], *, target_contexts: int) -> None:
        if target_contexts < 1:
            raise ValueError("simulated-user validation context batch size must be >= 1")
        if not context_counts:
            raise ValueError("simulated-user validation sampler received no dialogues")
        invalid = [count for count in context_counts if count < 1 or count > target_contexts]
        if invalid:
            raise ValueError(
                "every simulated-user validation dialogue must contain between 1 and the "
                f"context target; target={target_contexts}, invalid_counts={invalid[:8]}"
            )
        self.context_counts = [int(count) for count in context_counts]
        self.target_contexts = int(target_contexts)

    @staticmethod
    def _largest_subset_positions(
        indices: list[int], counts: list[int], target: int
    ) -> list[int]:
        predecessor: list[tuple[int, int] | None] = [None] * (target + 1)
        predecessor[0] = (-1, -1)
        for position, index in enumerate(indices):
            count = counts[index]
            for subtotal in range(target - count, -1, -1):
                if predecessor[subtotal] is None or predecessor[subtotal + count] is not None:
                    continue
                predecessor[subtotal + count] = (subtotal, position)
        best = max(subtotal for subtotal, previous in enumerate(predecessor) if previous is not None)
        positions: list[int] = []
        while best:
            previous, position = predecessor[best]  # ``best`` is reachable.
            positions.append(position)
            best = previous
        positions.reverse()
        return positions

    def __iter__(self) -> Iterator[list[int]]:
        remaining = list(range(len(self.context_counts)))
        while remaining:
            positions = self._largest_subset_positions(
                remaining, self.context_counts, self.target_contexts
            )
            if not positions:
                raise RuntimeError("simulated-user validation packer could not select a dialogue")
            selected_positions = set(positions)
            selected = [remaining[position] for position in positions]
            if len(selected) != len(set(selected)):
                raise RuntimeError("simulated-user validation batch contains a duplicate dialogue")
            if sum(self.context_counts[index] for index in selected) > self.target_contexts:
                raise RuntimeError("simulated-user validation batch exceeds its context target")
            remaining = [
                index for position, index in enumerate(remaining) if position not in selected_positions
            ]
            yield selected

    def __len__(self) -> int:
        # Match ``__iter__`` exactly so validation progress never reports more
        # completed batches than its denominator.
        remaining = list(range(len(self.context_counts)))
        batches = 0
        while remaining:
            positions = self._largest_subset_positions(
                remaining, self.context_counts, self.target_contexts
            )
            if not positions:
                raise RuntimeError("simulated-user validation packer could not count a dialogue batch")
            selected_positions = set(positions)
            remaining = [
                index for position, index in enumerate(remaining) if position not in selected_positions
            ]
            batches += 1
        return batches


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        if "loss_mask" not in data.batch:
            loss_mask = data.batch["attention_mask"]
        else:
            loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    info_gain_norm_mode="joint",
    curriculum_f1_weight=1.0,
    curriculum_ig_weight=1.0,
    query_group_advantage="disabled",
    query_group_similarity_threshold=0.90,
    query_group_metrics=None,
    rewrite_bound_rewards=None,
    action_rewards=None,
    info_gain_weight=1.0,
    action_reward_weight=1.0,
    simulated_user_enabled=False,
    simulated_user_component_weights=None,
    simulated_user_metrics=None,
):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            if "loss_mask" not in data.batch:
                grpo_calculation_mask = data.batch["attention_mask"][:, -response_length:]
            else:
                grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        if simulated_user_enabled:
            component_names = (
                "action",
                "answer_f1",
                "evidence_utility",
                "search_efficiency",
                "clarity",
                "patience",
                "format",
                "clarify_f1",
            )
            advantages, returns, sparse_metrics = compute_simulated_user_sparse_grpo_advantage(
                component_values={name: data.batch[f"sim_user_{name}_values"] for name in component_names},
                component_masks={name: data.batch[f"sim_user_{name}_mask"] for name in component_names},
                turn_boundary_mask=data.batch["sim_user_turn_boundary_mask"],
                subtask_ids=data.batch["sim_user_subtask_ids"],
                response_depths=data.batch["sim_user_response_depths"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                rollout_ids=(
                    data.non_tensor_batch["uidr"]
                    if "sim_user_event_order" in data.batch
                    else None
                ),
                event_orders=(
                    data.batch["sim_user_event_order"]
                    if "sim_user_event_order" in data.batch
                    else None
                ),
                gamma=gamma,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                component_weights=simulated_user_component_weights,
            )
            if simulated_user_metrics is not None:
                simulated_user_metrics.update(sparse_metrics)
        else:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                gamma=gamma,
                info_gain_norm_mode=info_gain_norm_mode,
                curriculum_f1_weight=curriculum_f1_weight,
                curriculum_ig_weight=curriculum_ig_weight,
                query_group_advantage=query_group_advantage,
                query_group_similarity_threshold=query_group_similarity_threshold,
                ig_query_embeddings=data.non_tensor_batch.get("ig_query_embeddings"),
                query_group_metrics=query_group_metrics,
                rewrite_bound_rewards=rewrite_bound_rewards,
                action_rewards=action_rewards,
                info_gain_weight=info_gain_weight,
                action_reward_weight=action_reward_weight,
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()
        self.data_offset = 0
        
        
        self.wait_reward_step = []
        self.client = MessageClient(self.config.data.data_writing_path,
                               isconsumer = True,
                               oss_access_key_id=self.config.data.oss_access_key_id,
                               oss_access_key_secret=self.config.data.oss_access_key_secret,
                               oss_endpoint=self.config.data.oss_endpoint)

    def _simulated_user_enabled(self) -> bool:
        return bool(self.config.algorithm.get("simulated_user_enabled", False))

    def _simulated_user_context_counts(self, dataset=None) -> list[int]:
        """Read canonical sub-task counts without changing an RLHF dataset."""
        dataset = self.train_dataset if dataset is None else dataset
        dataframe = getattr(dataset, "dataframe", None)
        if dataframe is None:
            raise TypeError(
                "simulated-user exact context batching requires a dataset exposing a dataframe"
            )
        counts: list[int] = []
        for row_index in range(len(dataframe)):
            row = dataframe[row_index]
            reward_model = row.get("reward_model") if isinstance(row, dict) else None
            if isinstance(reward_model, dict):
                payload = reward_model.get("simulated_dialogue", reward_model.get("dialogue", reward_model))
            else:
                payload = reward_model
            dialogue = parse_dialogue_payload(payload, fallback_id=f"dialogue-{row_index}")
            counts.append(len(dialogue["subtasks"]))
        return counts

    def _make_generation_config(self, *, n: int) -> GenerationConfig:
        return GenerationConfig(
            max_turns=self.config.max_turns,
            num_gpus=self.config.trainer.n_gpus_per_node,
            data_writing_path=self.config.data.data_writing_path,
            model_name=self.config.actor_rollout_ref.model.path,
            n=n,
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            search_engine=self.config.search_engine,
            nnodes=self.config.trainer.nnodes,
            oss_access_key_id=self.config.data.oss_access_key_id,
            oss_access_key_secret=self.config.data.oss_access_key_secret,
            oss_endpoint=self.config.data.oss_endpoint,
            codeact_env_disabled=self.config.codeact_env_disabled,
            info_gain_type=getattr(self.config.algorithm, "info_gain_type", "prob_diff"),
            query_group_advantage=str(self.config.algorithm.get("query_group_advantage", "disabled")).lower(),
            rewrite_bound_advantage=bool(self.config.algorithm.get("rewrite_bound_advantage", False)),
            rewrite_bound_topk=int(self.config.algorithm.get("rewrite_bound_topk", 3)),
            max_search_queries=int(self.config.algorithm.get("max_search_queries", 3)),
            allow_nonanswer=bool(self.config.algorithm.get("allow_nonanswer_action", False)),
            use_vectorized_gt_logprob=bool(self.config.algorithm.get("use_vectorized_gt_logprob", False)),
            max_model_len=self.config.actor_rollout_ref.rollout.get(
                "max_model_len", self.config.data.get("max_model_len", None)
            ),
            max_response_length=self.config.data.max_response_length,
            static_convagent_mode=bool(self.config.algorithm.get("static_convagent_mode", False)),
            static_convagent_direct_evidence_reward=bool(
                self.config.algorithm.get("static_convagent_direct_evidence_reward", False)
            ),
            static_convagent_short_answer_tokens=int(
                self.config.algorithm.get("static_convagent_short_answer_tokens", 4)
            ),
            static_chatr1_mode=bool(self.config.algorithm.get("static_chatr1_mode", False)),
            static_chatr1_intent_reward=bool(
                self.config.algorithm.get("static_chatr1_intent_reward", False)
            ),
        )

    def _make_simulated_user_manager(self, *, n: int, is_validation: bool):
        settings = SimulatedUserSettings(
            allow_clarify=bool(self.config.algorithm.get("simulated_user_allow_clarify", True)),
            allow_nonanswer=bool(self.config.algorithm.get("allow_nonanswer_action", True)),
            reward_mode=str(self.config.algorithm.get("simulated_user_reward_mode", "full")),
            enable_evidence_utility=bool(
                self.config.algorithm.get("simulated_user_enable_evidence_utility", False)
            ),
            enable_search_efficiency=bool(
                self.config.algorithm.get("simulated_user_enable_search_efficiency", False)
            ),
            enable_user_feedback=bool(
                self.config.algorithm.get("simulated_user_enable_feedback", True)
            ),
            # The assessment-only path is intentionally validation-only.  It
            # reports satisfaction for the no-feedback ablation without
            # letting a simulator call alter training trajectories, rewards,
            # or the policy-visible dialogue.
            assess_user_satisfaction=bool(is_validation)
            and bool(
                self.config.algorithm.get("simulated_user_assess_satisfaction", False)
            ),
            use_static_gold_context=bool(
                self.config.algorithm.get("simulated_user_static_gold_context", False)
            ),
            max_tool_calls=int(self.config.algorithm.get("simulated_user_max_tool_calls", 4)),
            max_search_queries=int(self.config.algorithm.get("simulated_user_max_search_queries", 1)),
            search_top_k=int(self.config.algorithm.get("simulated_user_search_top_k", 3)),
            max_answer_depth=int(self.config.algorithm.get("simulated_user_max_answer_depth", 3)),
            tool_observation_token_cap=int(
                self.config.algorithm.get("simulated_user_tool_observation_token_cap", 512)
            ),
            simulator_mode=str(self.config.algorithm.get("simulated_user_mode", "openai")),
            simulator_base_url=self.config.algorithm.get("simulated_user_base_url", None),
            simulator_model=self.config.algorithm.get("simulated_user_model", None),
            simulator_timeout_seconds=int(self.config.algorithm.get("simulated_user_timeout_seconds", 120)),
            simulator_question_token_cap=int(
                self.config.algorithm.get("simulated_user_simulator_question_token_cap", 384)
            ),
            simulator_gold_response_token_cap=int(
                self.config.algorithm.get("simulated_user_simulator_gold_response_token_cap", 768)
            ),
            simulator_answer_token_cap=int(
                self.config.algorithm.get("simulated_user_simulator_answer_token_cap", 768)
            ),
            simulator_transcript_token_cap=int(
                self.config.algorithm.get("simulated_user_simulator_transcript_token_cap", 2048)
            ),
            simulator_source_context_token_cap=int(
                self.config.algorithm.get("simulated_user_simulator_source_context_token_cap", 1024)
            ),
            simulator_max_output_tokens=int(
                self.config.algorithm.get("simulated_user_simulator_max_output_tokens", 128)
            ),
        )
        return SimulatedUserGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=self._make_generation_config(n=n),
            is_validation=is_validation,
            client=self.client,
            settings=settings,
        )
    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        """
        Creates the train and validation dataloaders.
        """
        # make sure the batch size is divisible by the dp size
        from verl.utils.import_utils import load_extern_type

        if "custom_train_cls" in self.config.data and self.config.data.custom_train_cls.get("path", None) is not None:
            # Dynamically load the custom dataset class specified in config
            try:
                dataset_train_cls = load_extern_type(self.config.data.custom_train_cls.path, self.config.data.custom_train_cls.name)
                if not issubclass(dataset_train_cls, Dataset):
                    raise TypeError(f"The custom dataset class '{self.config.data.custom_train_cls.name}' from '{self.config.data.custom_train_cls.path}' must inherit from torch.utils.data.Dataset")
                print(f"Using custom train dataset class: {dataset_train_cls.__name__}")
            except Exception as e:
                print(f"Error loading custom train dataset class: {e}")
                raise e
        else:
            dataset_train_cls = RLHFDataset
            print(f"Using default train dataset class: {dataset_train_cls.__name__}")
        
        
        if "custom_valid_cls" in self.config.data and self.config.data.custom_valid_cls.get("path", None) is not None:
            # Dynamically load the custom dataset class specified in config
            try:
                dataset_valid_cls = load_extern_type(self.config.data.custom_valid_cls.path, self.config.data.custom_valid_cls.name)
                if not issubclass(dataset_valid_cls, Dataset):
                    raise TypeError(f"The custom dataset class '{self.config.data.custom_valid_cls.name}' from '{self.config.data.custom_valid_cls.path}' must inherit from torch.utils.data.Dataset")
                print(f"Using custom valid dataset class: {dataset_valid_cls.__name__}")
            except Exception as e:
                print(f"Error loading custom valid dataset class: {e}")
                raise e
        else:
            dataset_valid_cls = RLHFDataset
            print(f"Using default valid dataset class: {dataset_valid_cls.__name__}")
            
        if dataset_train_cls.__name__ == 'AsycRMDataset':
            self.train_dataset = dataset_train_cls(
                data_files=self.config.data.train_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                config=self.config,
                reward_fn=self.reward_fn,
                train_reward_type=self.config.reward_model.train_reward_type,
            )
            self.config.data.shuffle = False
        else:
            self.train_dataset = dataset_train_cls(
                data_files=self.config.data.train_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                config=self.config.data,
            )
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        train_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        exact_context_batching = bool(
            self.config.algorithm.get("simulated_user_exact_context_batch", True)
        )
        if self._simulated_user_enabled() and exact_context_batching:
            target_contexts = int(self.config.data.train_batch_size)
            context_counts = self._simulated_user_context_counts()
            configured_steps = self.config.trainer.get("total_training_steps", None)
            if configured_steps is None:
                sampler_batches = max(1, (sum(context_counts) + target_contexts - 1) // target_contexts)
            else:
                # Keep the requested epoch count meaningful: 15 total updates
                # and 3 epochs produce five exact packed batches per epoch.
                # The sampler can revisit dialogues across epochs, just as a
                # static baseline revisits its examples in later epochs.
                configured_epochs = max(1, int(self.config.trainer.get("total_epochs", 1)))
                sampler_batches = max(
                    1,
                    (int(configured_steps) + configured_epochs - 1) // configured_epochs,
                )
            packed_sampler = SimulatedUserExactContextBatchSampler(
                context_counts,
                target_contexts=target_contexts,
                num_batches=sampler_batches,
                shuffle=bool(self.config.data.shuffle),
                seed=int(self.config.data.get("seed", 1)),
            )
            print(
                "[SimUser] exact packed batches: "
                f"target={target_contexts} original-subtask contexts/update; "
                f"batches/epoch={len(packed_sampler)}; "
                f"dialogue sub-task range={min(context_counts)}..{max(context_counts)}"
            )
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_sampler=packed_sampler,
                num_workers=self.config.data.get("dataloader_num_workers", 8),
                collate_fn=collate_fn,
            )
        else:
            if self._simulated_user_enabled():
                # Compatibility fallback for tiny smoke tests whose requested
                # batch is smaller than one complete dialogue.
                configured_collector_size = self.config.algorithm.get(
                    "simulated_user_rollout_dialogue_batch_size", None
                )
                if configured_collector_size is None:
                    train_batch_size = max(1, (int(self.config.data.train_batch_size) + 7) // 8)
                else:
                    train_batch_size = int(configured_collector_size)
                    if train_batch_size < 1:
                        raise ValueError("simulated_user_rollout_dialogue_batch_size must be >= 1")
                print(
                    "[SimUser] approximate collector dialogue batch size="
                    f"{train_batch_size}; target original-subtask contexts/update="
                    f"{self.config.data.train_batch_size}"
                )
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_size=train_batch_size,
                num_workers=self.config.data.get("dataloader_num_workers", 8),
                drop_last=True,
                collate_fn=collate_fn,
                sampler=sampler,
            )

        self.val_dataset = dataset_valid_cls(
            data_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)
        validation_context_batching = bool(
            self.config.algorithm.get("simulated_user_validation_context_batching", True)
        )
        if self._simulated_user_enabled() and validation_context_batching:
            target_contexts = int(val_batch_size)
            val_context_counts = self._simulated_user_context_counts(self.val_dataset)
            val_packed_sampler = SimulatedUserValidationContextBatchSampler(
                val_context_counts,
                target_contexts=target_contexts,
            )
            print(
                "[SimUser] validation packed batches: "
                f"target<={target_contexts} original-subtask contexts/batch; "
                f"batches={len(val_packed_sampler)}; "
                f"dialogue sub-task range={min(val_context_counts)}..{max(val_context_counts)}"
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_sampler=val_packed_sampler,
                num_workers=self.config.data.get("dataloader_num_workers", 8),
                collate_fn=collate_fn,
            )
        else:
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=self.config.data.get("dataloader_num_workers", 8),
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(
        self,
        inputs,
        outputs,
        reward_extra_infos_dict,
        dump_path,
        rewards,
        info_gain_rewards,
        f1_scores=[],
        em_scores=[],
        noformatf1_scores=[],
        ground_truths=[],
        raw_responses=None,
        data_sources=None,
        ground_truth_answers=None,
        ground_truth_passage_ids=None,
        ground_truth_passage_texts=None,
        predicted_passage_ids=None,
        predicted_passage_texts=None,
        expected_actions=None,
        predicted_actions=None,
        action_corrects=None,
        format_valids=None,
    ):
        

        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")
        inputs = inputs[:len(outputs)]
        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
			"rewards": rewards,
			"info_gain_rewards": info_gain_rewards,
            "step": [self.global_steps] * n,
        }
        
        if len(ground_truths) == n:
            base_data['ground_truth'] = ground_truths

        # Explicit retrieval-evaluation fields.  ``raw_response`` preserves
        # the ChatML trajectory, while ``output`` keeps the existing readable
        # decoded output for backwards compatibility.
        if raw_responses is not None and len(raw_responses) == n:
            base_data["raw_response"] = raw_responses
        if data_sources is not None and len(data_sources) == n:
            base_data["data_source"] = data_sources
        # This writer is used by the legacy, non-simulated-user validation
        # path.  Keep it explicit so metric reporting never mistakes static
        # action-labelled data for online simulator traces.
        base_data["evaluation_mode"] = ["static"] * n
        if ground_truth_answers is not None and len(ground_truth_answers) == n:
            base_data["ground_truth_answer"] = ground_truth_answers
        if ground_truth_passage_ids is not None and len(ground_truth_passage_ids) == n:
            base_data["ground_truth_passage_ids"] = ground_truth_passage_ids
        if ground_truth_passage_texts is not None and len(ground_truth_passage_texts) == n:
            base_data["ground_truth_passage_texts"] = ground_truth_passage_texts
        if predicted_passage_ids is not None and len(predicted_passage_ids) == n:
            base_data["predicted_passage_ids"] = predicted_passage_ids
        if predicted_passage_texts is not None and len(predicted_passage_texts) == n:
            base_data["predicted_passage_texts"] = predicted_passage_texts
        has_action_labels = (
            expected_actions is not None
            and len(expected_actions) == n
            and any(
                (isinstance(action, (list, tuple, set)) and len(action) > 0)
                or action in {"answer", "clarify", "nonanswer"}
                for action in expected_actions
            )
        )
        if has_action_labels:
            if predicted_actions is not None and len(predicted_actions) == n:
                base_data["expected_action"] = expected_actions
                base_data["predicted_action"] = predicted_actions
            if action_corrects is not None and len(action_corrects) == n:
                base_data["action_correct"] = action_corrects
            if format_valids is not None and len(format_valids) == n:
                base_data["format_valid"] = format_valids
        if len(f1_scores) == n:
            base_data['f1_scores'] = f1_scores
        if len(em_scores) == n:
            base_data['em_scores'] = em_scores
        if len(noformatf1_scores) == n:
            base_data['noformatf1_scores'] = noformatf1_scores	

                        
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self, *, simulated_user_max_batches: Optional[int] = None):
        if self._simulated_user_enabled():
            return validate_simulated_user(
                self,
                max_batches_override=simulated_user_max_batches,
            )

        reward_tensor_lst = []
        em_reward_tensor_lst = []
        llm_reward_tensor_lst = []
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        ground_truths = []
        reward_tensor_all = []
        info_gain_rewards_all = []
        raw_responses = []
        ground_truth_passage_texts = []
        ground_truth_answers = []
        ground_truth_passage_ids = []
        predicted_passage_ids = []
        predicted_passage_texts = []
        expected_actions = []
        predicted_actions = []
        action_corrects = []
        format_valids = []

        try:
            validation_total_batches = len(self.val_dataloader)
        except TypeError:
            validation_total_batches = None
        validation_completed_samples = 0

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            num_gpus=self.config.trainer.n_gpus_per_node,
            data_writing_path=self.config.data.data_writing_path,
            model_name=self.config.actor_rollout_ref.model.path,
            n=1,  # only roll once
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            search_engine=self.config.search_engine,
            nnodes=self.config.trainer.nnodes,
            oss_access_key_id=self.config.data.oss_access_key_id,
            oss_access_key_secret=self.config.data.oss_access_key_secret,
            oss_endpoint=self.config.data.oss_endpoint,
            codeact_env_disabled=self.config.codeact_env_disabled,
            info_gain_type=getattr(self.config.algorithm, 'info_gain_type', 'prob_diff'),
            query_group_advantage=str(self.config.algorithm.get("query_group_advantage", "disabled")).lower(),
            rewrite_bound_advantage=bool(self.config.algorithm.get("rewrite_bound_advantage", False)),
            rewrite_bound_topk=int(self.config.algorithm.get("rewrite_bound_topk", 3)),
            max_search_queries=int(self.config.algorithm.get("max_search_queries", 3)),
            allow_nonanswer=bool(self.config.algorithm.get("allow_nonanswer_action", False)),
            use_vectorized_gt_logprob=bool(self.config.algorithm.get("use_vectorized_gt_logprob", False)),
            max_model_len=self.config.actor_rollout_ref.rollout.get(
                'max_model_len', self.config.data.get('max_model_len', None)
            ),
            max_response_length=self.config.data.max_response_length,
            static_convagent_mode=bool(self.config.algorithm.get("static_convagent_mode", False)),
            static_convagent_direct_evidence_reward=bool(
                self.config.algorithm.get("static_convagent_direct_evidence_reward", False)
            ),
            static_convagent_short_answer_tokens=int(
                self.config.algorithm.get("static_convagent_short_answer_tokens", 4)
            ),
            static_chatr1_mode=bool(self.config.algorithm.get("static_chatr1_mode", False)),
            static_chatr1_intent_reward=bool(
                self.config.algorithm.get("static_chatr1_intent_reward", False)
            ),
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation=True,
            client = self.client
        )
        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
                # repeat test batch
                test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
                test_batch.non_tensor_batch["uidr"] = np.tile(np.arange(self.config.actor_rollout_ref.rollout.val_kwargs.n), len(test_batch.batch))

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                    return {}

                # Store original inputs
                input_ids = test_batch.batch["input_ids"]
                # TODO: Can we keep special tokens except for padding tokens?
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_inputs" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
#                 if "raw_prompt" in test_batch.non_tensor_batch:
#                     non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                test_gen_batch = test_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                }
                print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                if not self.async_rollout_mode:
                    test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                else:
                    self.async_rollout_manager.wake_up()
                    test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                    self.async_rollout_manager.sleep()

                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print("validation generation end")

                # Store generated outputs
                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                result = self.val_reward_fn(test_batch, return_dict=True)
                reward_tensor = result["reward_tensor"]
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)

                reward_extra_infos_dict["reward"].extend(scores)
                if "reward_extra_info" in result:
                    for key, lst in result["reward_extra_info"].items():
                        reward_extra_infos_dict[key].extend(lst)

                data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
        else:
            f1_scores_lst = []
            em_scores_lst = []
            noformatf1_scores_lst = []
            for validation_batch_index, batch_dict in enumerate(self.val_dataloader, start=1):
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                total_text = str(validation_total_batches) if validation_total_batches is not None else "?"
                print(
                    f"[Validation] progress: starting batch {validation_batch_index}/{total_text}; "
                    f"completed_samples={validation_completed_samples}",
                    flush=True,
                )

                # Store original inputs
                input_ids = test_batch.batch['input_ids']
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw

                        # normalize length to be divisible by total GPU count
                        n_gpus = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                        if len(test_gen_batch) >= n_gpus:
                            norm_len = len(test_gen_batch) // n_gpus * n_gpus
                        else:
                            # If batch size < n_gpus, keep all data (let downstream handle it)
                            norm_len = len(test_gen_batch)
                        test_gen_batch = test_gen_batch[:norm_len]
                        sample_inputs.extend(input_texts[:norm_len])

                        batch_reward_models = list(test_batch.non_tensor_batch["reward_model"])[:norm_len]
                        batch_data_sources = list(
                            test_batch.non_tensor_batch.get("data_source", ["unknown"] * norm_len)
                        )[:norm_len]
                        static_convagent_mode = bool(
                            self.config.algorithm.get("static_convagent_mode", False)
                        )
                        static_chatr1_mode = bool(
                            self.config.algorithm.get("static_chatr1_mode", False)
                        )
                        if static_convagent_mode and static_chatr1_mode:
                            raise ValueError("static ConvAgent and ChatR1 modes are mutually exclusive")
                        answer_selector = (
                            select_static_chatr1_answer_ground_truth_with_passage_ids
                            if static_chatr1_mode
                            else (
                                select_static_convagent_answer_ground_truth_with_passage_ids
                                if static_convagent_mode
                                else select_answer_ground_truth_with_passage_ids
                            )
                        )
                        batch_answer_references = [
                            answer_selector(reward_model, data_source=data_source)
                            for reward_model, data_source in zip(batch_reward_models, batch_data_sources)
                        ]
                        if static_chatr1_mode:
                            # Query--rewrite intent reward needs the original
                            # nested candidates, not only the display reference.
                            batch_ground_truths = [
                                {
                                    "ground_truth": reward_model.get("ground_truth", reward_model)
                                    if isinstance(reward_model, dict)
                                    else reward_model,
                                    "data_source": data_source,
                                }
                                for reward_model, data_source in zip(batch_reward_models, batch_data_sources)
                            ]
                        else:
                            batch_ground_truths = [
                                {"ground_truth": answer}
                                for answer, _ in batch_answer_references
                            ]

                        _, final_gen_batch_output, info_gain_rewards, first_retrieved_passages = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            global_steps=-self.global_steps,  # negative value indicates validation
                            ground_truths=batch_ground_truths
                        )
                    test_batch = test_batch[:len(final_gen_batch_output)]
                    test_batch = test_batch.union(final_gen_batch_output)

                    # Store original outputs
                    output_ids = test_batch.batch['responses']
                    output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                    sample_outputs.extend(output_texts)
                    raw_responses.extend(
                        self.tokenizer.decode(ids, skip_special_tokens=False).replace("<|endoftext|>", "")
                        for ids in output_ids
                    )
                    ground_truths.extend(answer for answer, _ in batch_answer_references)
                    ground_truth_answers.extend(answer for answer, _ in batch_answer_references)
                    ground_truth_passage_ids.extend(passage_ids for _, passage_ids in batch_answer_references)
                    ground_truth_passage_texts.extend(
                        passage_ids if str(data_source).strip().lower() == "topiocqa" else []
                        for data_source, (_, passage_ids) in zip(batch_data_sources, batch_answer_references)
                    )

                    for data_source, retrieved_passages in zip(batch_data_sources, first_retrieved_passages):
                        # Preserve actual retriever IDs for every dataset.  The
                        # TopiOCQA source's gold "IDs" are passage text, so its
                        # relevance comparison is exported separately below.
                        predicted_passage_ids.append(
                            [passage.get("passage_id", "") for passage in retrieved_passages]
                        )
                        if str(data_source).strip().lower() == "topiocqa":
                            # TopiOCQA stores ground-truth passage *text*
                            # rather than a stable ID, so compare canonical
                            # retrieved text in the offline metrics script.
                            predicted_passage_texts.append(
                                [passage.get("passage_text", "") for passage in retrieved_passages]
                            )
                        else:
                            predicted_passage_texts.append([])
                        print(
                            f"[Validation] sample={len(predicted_passage_ids) - 1} "
                            f"data_source={data_source} predicted_passage_ids={predicted_passage_ids[-1]}",
                            flush=True,
                        )
                    
                    
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    try:
                        reward_dict = self.val_reward_fn(test_batch, info_gain_rewards=info_gain_rewards, is_validation=True)
                        f1_scores_lst.extend(reward_dict["f1_scores"])
                        em_scores_lst.extend(reward_dict["em_scores"])
                        noformatf1_scores_lst.extend(reward_dict["noformatf1_scores"])
                        expected_actions.extend(reward_dict.get("expected_actions", []))
                        predicted_actions.extend(reward_dict.get("predicted_actions", []))
                        action_corrects.extend(reward_dict.get("action_corrects", []))
                        format_valids.extend(reward_dict.get("format_valids", []))
                        reward_tensor = reward_dict["reward_tensor"]
                        reward_tensor_all.extend(reward_tensor.tolist())
                        info_gain_rewards_all.extend(info_gain_rewards)
                    except:
                        import traceback
                        print(f"----- {str(traceback.format_exc())}")
                        print('------',test_batch)
                        exit()

                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                validation_completed_samples += len(test_batch)
                progress_text = (
                    "?"
                    if validation_total_batches is None
                    else f"{100 * validation_batch_index / validation_total_batches:.1f}%"
                )
                print(
                    f"[Validation] progress: completed batch {validation_batch_index}/{total_text} "
                    f"({progress_text}); completed_samples={validation_completed_samples}",
                    flush=True,
                )
            f1_scores = f1_scores_lst
            em_scores = em_scores_lst
            noformatf1_scores = noformatf1_scores_lst

        data_sources = np.concatenate(data_source_lst, axis=0)
		
        data_source_reward = {}
        for i in range(len(data_sources)):
            data_source = data_sources[i]
            # Metric naming: f1 = noformatf1 (without format penalty), f1_format_penalty = original f1 (with format penalty)
            key_f1 = f"{data_source}_f1"  # uses noformatf1 value
            key_f1_format_penalty = f"{data_source}_f1_format_penalty"  # uses original f1 value
            key_em = f"{data_source}_em"
            if key_f1 not in data_source_reward:
                data_source_reward[key_f1] = []
            data_source_reward[key_f1].append(noformatf1_scores[i])  # f1 uses noformatf1 value
            if key_f1_format_penalty not in data_source_reward:
                data_source_reward[key_f1_format_penalty] = []
            data_source_reward[key_f1_format_penalty].append(f1_scores[i])  # f1_format_penalty uses original f1 value
            if key_em not in data_source_reward:
                data_source_reward[key_em] = []
            data_source_reward[key_em].append(em_scores[i])
		

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        # Action accuracy is only defined for converted action-labelled samples
        # (InsCiT / TopiOCQA). Static ConvAgent labels may contain an allowed
        # set of actions, whereas online data has one canonical action.
        action_values_by_source: dict[str, list[float]] = defaultdict(list)
        if len(expected_actions) == len(data_sources):
            for data_source, expected_action, action_correct in zip(
                data_sources, expected_actions, action_corrects
            ):
                has_action_label = (
                    isinstance(expected_action, (list, tuple, set))
                    and len(expected_action) > 0
                ) or expected_action in {"answer", "clarify", "nonanswer"}
                if has_action_label and action_correct is not None:
                    action_values_by_source[str(data_source)].append(float(bool(action_correct)))
        for data_source, values in action_values_by_source.items():
            metric_dict[f"val/test_score/{data_source}_action_accuracy"] = np.mean(values)
        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
				rewards=reward_tensor_all if reward_tensor_all else reward_tensor.tolist(),
				info_gain_rewards=info_gain_rewards_all if info_gain_rewards_all else info_gain_rewards,
				f1_scores=f1_scores,
				em_scores=em_scores,
                noformatf1_scores=noformatf1_scores,
                ground_truths=ground_truths,
                raw_responses=raw_responses,
                data_sources=data_sources.tolist(),
                ground_truth_answers=ground_truth_answers,
                ground_truth_passage_ids=ground_truth_passage_ids,
                ground_truth_passage_texts=ground_truth_passage_texts,
                predicted_passage_ids=predicted_passage_ids,
                predicted_passage_texts=predicted_passage_texts,
                expected_actions=expected_actions,
                predicted_actions=predicted_actions,
                action_corrects=action_corrects,
                format_valids=format_valids,
            )
            
        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self) -> bool:
        if self.config.trainer.resume_mode == "disable":
            return False

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return False
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path) and self.config.data.custom_train_cls.get('name','') != 'AsycRMDataset':
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            self.total_training_steps += self.global_steps
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
        
        # Determine data offset based on rollout
        if self.config.data.get('start_with_rollout', False):
            rm_dir = self.config.reward_model.async_data_dir
            rollout_data_files = glob.glob(rm_dir + '/rollout_*')
            for file in rollout_data_files:
                step = file.split('_')[-1]
                if step.isdigit():
                    batch = DataProto.load_from_disk(file)
                    uids = set(batch.non_tensor_batch['uid'].tolist())
                    self.wait_reward_step.append(int(step))
                    self.data_offset += len(uids)
                    self.global_steps = max(int(step), self.global_steps)

        return True

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens
        
        Returns:
            global_idx: torch.Tensor, the reorder indices used to reorder the batch
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)
        return global_idx

        
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking
        
        # Initialize vectorized GT logprob switch from Hydra config
        from scrl.llm_agent.vectorized_gt_logprob import init_from_config as init_vectorized_config
        init_vectorized_config(self.config)

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        loaded_checkpoint = self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if (self.val_reward_fn is not None or self._simulated_user_enabled()) and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            val_data_dir = self.config.trainer.get("validation_data_dir", None)
            if val_data_dir:
                json.dump(val_metrics,open(f'{val_data_dir}/metric_step_{self.global_steps}.json','w'))
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        
        # ``global_steps`` is the zero-based index of the update currently being
        # processed. A fresh run therefore starts at 0. Checkpoints are named
        # after their completed zero-based update (e.g. global_step_4 means five
        # updates have completed), so resume from the following update.
        if loaded_checkpoint:
            self.global_steps += 1

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        last_val_metrics = None
        
        simulated_user_enabled = self._simulated_user_enabled()
        if simulated_user_enabled:
            if not self.config.do_search:
                raise ValueError("simulated-user sparse GRPO requires do_search=true")
            if self.config.reward_model.launch_reward_fn_async:
                raise ValueError("simulated-user sparse GRPO does not support async reward workers")
            if str(self.config.algorithm.get("query_group_advantage", "disabled")).lower() != "disabled":
                raise ValueError("simulated-user sparse GRPO does not use query-grouped IG advantages")
            generation_manager = self._make_simulated_user_manager(
                n=self.config.agent_grpo.n,
                is_validation=False,
            )
            print("[SimUser] enabled: legacy retrieval-IG rewards are disabled for this run", flush=True)
        else:
            generation_manager = LLMGenerationManager(
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=self._make_generation_config(n=self.config.agent_grpo.n),
                is_validation=False,
                client=self.client,
            )
        train_reward_type = self.config.reward_model.train_reward_type
        static_convagent_mode = bool(self.config.algorithm.get("static_convagent_mode", False))
        static_chatr1_mode = bool(self.config.algorithm.get("static_chatr1_mode", False))
        if static_convagent_mode and static_chatr1_mode:
            raise ValueError("static ConvAgent and ChatR1 modes are mutually exclusive")
        static_baseline_mode = static_convagent_mode or static_chatr1_mode
        static_baseline_name = "StaticChatR1" if static_chatr1_mode else "StaticConvAgent"
        static_monitor_enabled = static_baseline_mode and bool(
            self.config.trainer.get("static_convagent_monitor_enabled", True)
        )
        if static_monitor_enabled and simulated_user_enabled:
            raise ValueError("static-baseline monitoring cannot be combined with simulated-user rollouts")
        static_monitor_frequency = int(
            self.config.trainer.get("static_convagent_monitor_frequency", 5)
        )
        static_monitor_patience = int(
            self.config.trainer.get("static_convagent_monitor_patience", 3)
        )
        static_monitor_min_delta = float(
            self.config.trainer.get("static_convagent_monitor_min_delta", 0.002)
        )
        static_monitor_stability_window = int(
            self.config.trainer.get("static_convagent_monitor_stability_window", 3)
        )
        static_monitor_stability_tolerance = float(
            self.config.trainer.get("static_convagent_monitor_stability_tolerance", 0.005)
        )
        static_monitor_metric = str(
            self.config.trainer.get("static_convagent_monitor_metric", "")
        ).strip()
        if static_monitor_enabled and static_monitor_frequency < 1:
            raise ValueError("trainer.static_convagent_monitor_frequency must be >= 1")
        static_monitor_history: list[dict[str, float]] = []
        static_stop_reason: Optional[str] = None
        offset = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                
                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_inputs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
#                 if "raw_prompt" in batch.non_tensor_batch:
#                     non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # ``global_steps`` is zero-based before the update.  Treat the
                # current batch as the final one when it completes the target
                # number of optimization steps, not on the following batch.
                is_last_step = self.global_steps + 1 >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    if not self.config.do_search:
                        with _timer('gen', timing_raw):
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                self.async_rollout_manager.wake_up()
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                                self.async_rollout_manager.sleep()

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            with _timer('gen_max', timing_raw):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info['do_sample'] = False
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                                batch = batch.union(gen_baseline_output)
                                reward_baseline_tensor = self.reward_fn(batch)
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                                batch.batch['reward_baselines'] = reward_baseline_tensor

                                del gen_baseline_batch, gen_baseline_output
                    else:
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            assert False, 'REMAX is not supported for search'
                        else:
                            with _timer('gen', timing_raw):
                                generation_manager.timing_raw = timing_raw
                                if simulated_user_enabled:
                                    gen_batch_output, simulated_user_records = generation_manager.run_dialogue_loop(
                                        gen_batch=gen_batch,
                                        reward_models=batch.non_tensor_batch['reward_model'],
                                    )
                                    # This method does not use legacy retrieval-IG rewards.
                                    info_gain_rewards = [[] for _ in range(len(gen_batch_output))]
                                else:
                                    gen_str_list, gen_batch_output, info_gain_rewards, _first_retrieved_passages = generation_manager.run_llm_loop(
                                        gen_batch=gen_batch,
                                        global_steps=self.global_steps,
                                        ground_truths=batch.non_tensor_batch['reward_model']
                                    )
                            for key in gen_batch_output.batch.keys():
                                if key.endswith('_values'):
                                    gen_batch_output.batch[key] = gen_batch_output.batch[key].float()
                                else:
                                    gen_batch_output.batch[key] = gen_batch_output.batch[key].long()
                    if simulated_user_enabled:
                        # ``run_dialogue_loop`` already expands all sibling
                        # trajectories. It now emits one exact prompt/action
                        # row per response event, because policy context may
                        # compact evidence between actions. Do not repeat the
                        # source dialogues here.
                        batch = gen_batch_output
                        if "uid" not in batch.non_tensor_batch or "uidr" not in batch.non_tensor_batch:
                            raise RuntimeError(
                                "simulated-user generation output is missing flattened rollout group metadata"
                            )
                        info_gain_rewards = [[] for _ in range(len(batch))]
                        if "sim_user_row_subtask" in batch.batch:
                            context_groups = len(
                                {
                                    (str(uid), int(subtask))
                                    for uid, subtask in zip(
                                        batch.non_tensor_batch["uid"],
                                        batch.batch["sim_user_row_subtask"].tolist(),
                                    )
                                }
                            )
                        else:
                            rollout_count = int(self.config.agent_grpo.n)
                            if len(batch) % rollout_count:
                                raise RuntimeError(
                                    "simulated-user flattened trajectory rows are not divisible by agent_grpo.n: "
                                    f"rows={len(batch)}, n={rollout_count}"
                                )
                            context_groups = len(batch) // rollout_count
                        if bool(self.config.algorithm.get("simulated_user_exact_context_batch", True)):
                            expected_context_groups = int(self.config.data.train_batch_size)
                            if context_groups != expected_context_groups:
                                raise RuntimeError(
                                    "simulated-user exact packed batch lost or added sub-task contexts: "
                                    f"expected={expected_context_groups}, got={context_groups}"
                                )
                        metrics["sim_user/context_groups"] = float(context_groups)
                        metrics["sim_user/trajectory_rows"] = float(len(batch))
                        metrics["sim_user/dialogues_collected"] = float(
                            len(set(str(uid) for uid in batch.non_tensor_batch["uid"]))
                        )
                        # One source sub-task can now produce a variable
                        # number of exact prompt/action rows (tool actions and
                        # the eventual terminal action are trained
                        # separately). Ray's DP dispatcher requires an equal
                        # number of rows per rank, so pad only to the next
                        # rank divisor and mask copied rows out of all losses.
                        # This does not alter the 128 real sub-task contexts
                        # or any GRPO normalization group.
                        sim_user_real_rows = len(batch)
                        batch, sim_user_dp_pad_size = pad_dataproto_to_divisor(
                            batch, self.actor_rollout_wg.world_size
                        )
                        batch.non_tensor_batch["simulated_user_is_padding"] = np.zeros(
                            len(batch), dtype=bool
                        )
                        if sim_user_dp_pad_size:
                            padding_rows = slice(-sim_user_dp_pad_size, None)
                            batch.non_tensor_batch["simulated_user_is_padding"][padding_rows] = True
                            # ``loss_mask`` is the PPO mask for multi-turn
                            # actions. Zero all sparse-reward metadata as a
                            # defense in depth so these rows are invisible to
                            # return construction as well.
                            for key in (
                                "loss_mask",
                                "sim_user_turn_boundary_mask",
                                "sim_user_action_mask",
                                "sim_user_answer_f1_mask",
                                "sim_user_evidence_utility_mask",
                                "sim_user_search_efficiency_mask",
                                "sim_user_clarity_mask",
                                "sim_user_patience_mask",
                                "sim_user_format_mask",
                                "sim_user_clarify_f1_mask",
                            ):
                                if key in batch.batch:
                                    batch.batch[key][padding_rows].zero_()
                            for key in (
                                "sim_user_subtask_ids",
                                "sim_user_response_depths",
                                "sim_user_event_order",
                                "sim_user_row_subtask",
                            ):
                                if key in batch.batch:
                                    batch.batch[key][padding_rows].fill_(-1)
                            for key in (
                                "sim_user_action_values",
                                "sim_user_answer_f1_values",
                                "sim_user_evidence_utility_values",
                                "sim_user_search_efficiency_values",
                                "sim_user_clarity_values",
                                "sim_user_patience_values",
                                "sim_user_format_values",
                                "sim_user_clarify_f1_values",
                            ):
                                if key in batch.batch:
                                    batch.batch[key][padding_rows].zero_()
                            info_gain_rewards.extend([[] for _ in range(sim_user_dp_pad_size)])
                        metrics["sim_user/dp_padding_rows"] = float(sim_user_dp_pad_size)
                        metrics["sim_user/trajectory_rows"] = float(sim_user_real_rows)
                    else:
                        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.agent_grpo.n, interleave=True)
                        batch.non_tensor_batch["uidr"] = np.tile(np.arange(self.config.actor_rollout_ref.rollout.n), len(batch.batch))
                        batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if simulated_user_enabled and sim_user_dp_pad_size:
                        batch.batch["response_mask"][-sim_user_dp_pad_size:].zero_()
                    if simulated_user_enabled:
                        padding_flags = batch.non_tensor_batch["simulated_user_is_padding"]
                        sim_records = [
                            record
                            for record, is_padding in zip(
                                batch.non_tensor_batch.get("simulated_user_records", []),
                                padding_flags,
                            )
                            if not bool(is_padding)
                        ]
                        sim_events = [
                            event
                            for record in sim_records
                            for event in record.get("subtasks", [])
                        ]
                        if sim_events:
                            metrics["sim_user/tool_calls"] = float(sum(event["predicted_action"] == "tool_call" for event in sim_events))
                            metrics["sim_user/format_failure_rate"] = float(np.mean([
                                not bool(event["format_valid"]) for event in sim_events
                            ]))
                            metrics["sim_user/trailing_content_discard_rate"] = float(np.mean([
                                bool(event.get("trailing_content_discarded", False)) for event in sim_events
                            ]))
                            metrics["sim_user/mean_response_depth"] = float(np.mean([
                                event["response_depth"] for event in sim_events
                            ]))
                            answer_f1_values = [
                                event["answer_f1"]
                                for event in sim_events
                                if event.get("answer_f1") is not None
                            ]
                            if answer_f1_values:
                                metrics["sim_user/answer_f1_reward_mean"] = float(
                                    np.mean(answer_f1_values)
                                )
                                metrics["sim_user/answer_f1_reward_std"] = float(
                                    np.std(answer_f1_values)
                                )
                            evidence_utility_values = [
                                event["evidence_utility"]
                                for event in sim_events
                                if event.get("evidence_utility") is not None
                            ]
                            if evidence_utility_values:
                                metrics["sim_user/evidence_utility_mean"] = float(
                                    np.mean(evidence_utility_values)
                                )
                                metrics["sim_user/evidence_utility_std"] = float(
                                    np.std(evidence_utility_values)
                                )
                            search_efficiency_values = [
                                event["search_efficiency"]
                                for event in sim_events
                                if event.get("search_efficiency") is not None
                            ]
                            if search_efficiency_values:
                                metrics["sim_user/search_efficiency_mean"] = float(
                                    np.mean(search_efficiency_values)
                                )
                                metrics["sim_user/search_efficiency_std"] = float(
                                    np.std(search_efficiency_values)
                                )
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        reorder_idx = self._balance_batch(batch, metrics=metrics)
                        # Synchronously reorder info_gain_rewards to ensure indices match the batch
                        info_gain_rewards = [info_gain_rewards[i] for i in reorder_idx.tolist()]
                    
                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Compute rollout
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    async_data_dir = self.config.reward_model.get("async_data_dir", None)

                    if self.config.reward_model.launch_reward_fn_async:
                        batch.save_to_disk(async_data_dir + f'/rollout_{self.global_steps}')
                    
                    with _timer("reward", timing_raw):
                        if simulated_user_enabled:
                            # This method bypasses legacy retrieval-IG/reward-manager scoring.
                            reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
                            reward_extra_infos_dict = {}
                        else:
                            # compute reward model score
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            # Original ray async RM computation doesn't decouple sampling and update; we change to delayed learning with decoupled sampling and update.
                            if self.config.reward_model.launch_reward_fn_async:
                                self.wait_reward_step.append(self.global_steps)
                                if self.config.data.custom_train_cls.name == 'AsycRMDataset':
                                    train_reward_type = 'empty'
                                if info_gain_rewards is not None:
                                    batch.non_tensor_batch["info_gain_rewards"] = np.array(info_gain_rewards, dtype=object)
                                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer, self.global_steps, train_reward_type)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(
                                    batch, self.reward_fn, train_reward_type, info_gain_rewards=info_gain_rewards
                                )
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                reward_extra_infos_dict={},
                                dump_path=rollout_data_dir,
								rewards=reward_tensor.tolist(),
								info_gain_rewards=info_gain_rewards,
                            )


                    if self.config.reward_model.launch_reward_fn_async:
                        assert async_data_dir is not None
                        ready_step = -1
                        pprint(f"Async waiting batches: {self.wait_reward_step}")
                        for step in sorted(self.wait_reward_step):
                            rm_file = async_data_dir + f"/rm_{step}"
                            if not os.path.exists(rm_file):
                                continue  # reward not ready yet
                            ready_step = step
                            break
                        if ready_step == -1:
                            pprint(f"Not found any ready reward step")
                            progress_bar.update(1)
                            self.global_steps += 1
                            continue
                        else:
                            self.wait_reward_step.remove(ready_step)
                            rm_file = async_data_dir + f"/rm_{ready_step}"
                            rm_used_file = async_data_dir + f"/used_rm_{ready_step}"
                            pprint(f"Found ready reward step: {ready_step}")
                            batch = DataProto.load_from_disk(rm_file) 
                            os.rename(rm_file, rm_used_file)

                    
                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)   
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")   
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if not self.config.reward_model.launch_reward_fn_async:
                            batch.batch["token_level_scores"] = reward_tensor
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
                        info_gain_norm_mode = getattr(self.config.algorithm, 'info_gain_norm_mode', 'joint')  # "joint" or "separate"
                        query_group_advantage = str(self.config.algorithm.get("query_group_advantage", "disabled")).lower()
                        query_group_similarity_threshold = float(
                            self.config.algorithm.get("query_group_similarity_threshold", 0.90)
                        )
                        query_group_debug = bool(self.config.algorithm.get("query_group_debug", True))
                        if query_group_advantage not in {"disabled", "semantic"}:
                            raise ValueError(
                                "algorithm.query_group_advantage must be 'disabled' or 'semantic', "
                                f"got {query_group_advantage!r}"
                            )
                        if query_group_advantage == "semantic" and info_gain_norm_mode != "separate":
                            raise ValueError(
                                "algorithm.query_group_advantage=semantic requires "
                                "algorithm.info_gain_norm_mode=separate"
                            )
                        query_group_metrics = {} if query_group_debug else None
                        simulated_user_metrics = {} if simulated_user_enabled else None
                        simulated_user_component_weights = {
                            name: float(self.config.algorithm.get(f"simulated_user_{name}_weight", 1.0))
                            for name in (
                                "action",
                                "answer_f1",
                                "evidence_utility",
                                "search_efficiency",
                                "clarity",
                                "patience",
                                "format",
                                "clarify_f1",
                            )
                        }
                        # Curriculum Learning: dynamically adjust F1 and IG weights
                        use_curriculum = getattr(self.config.algorithm, 'use_curriculum', False)
                        if use_curriculum:
                            total_steps = self.total_training_steps
                            progress = min(self.global_steps / max(total_steps, 1), 1.0)
                            # Read curriculum config
                            f1_init = getattr(self.config.algorithm, 'curriculum_f1_init', 0.5)
                            f1_final = getattr(self.config.algorithm, 'curriculum_f1_final', 1.0)
                            ig_init = getattr(self.config.algorithm, 'curriculum_ig_init', 1.0)
                            ig_final = getattr(self.config.algorithm, 'curriculum_ig_final', 0.5)
                            # Linear interpolation
                            curriculum_f1_weight = f1_init + (f1_final - f1_init) * progress
                            curriculum_ig_weight = ig_init + (ig_final - ig_init) * progress
                            # Log to metrics
                            metrics["curriculum/f1_weight"] = curriculum_f1_weight
                            metrics["curriculum/ig_weight"] = curriculum_ig_weight
                            metrics["curriculum/progress"] = progress
                        else:
                            curriculum_f1_weight = 1.0
                            curriculum_ig_weight = 1.0

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.agent_grpo.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=(self.config.actor_rollout_ref.rollout.multi_turn.enable or simulated_user_enabled),
                            info_gain_norm_mode=info_gain_norm_mode,
                            curriculum_f1_weight=curriculum_f1_weight,
                            curriculum_ig_weight=curriculum_ig_weight,
                            query_group_advantage=query_group_advantage,
                            query_group_similarity_threshold=query_group_similarity_threshold,
                            query_group_metrics=query_group_metrics,
                            rewrite_bound_rewards=batch.non_tensor_batch.get("rewrite_bound_rewards"),
                            action_rewards=batch.non_tensor_batch.get("action_rewards"),
                            info_gain_weight=float(
                                self.config.algorithm.get(
                                    "static_chatr1_intent_weight",
                                    1.0,
                                )
                                if static_chatr1_mode
                                else self.config.algorithm.get("static_convagent_info_gain_weight", 1.0)
                            ),
                            action_reward_weight=float(
                                self.config.algorithm.get("static_convagent_action_weight", 1.0)
                            ),
                            simulated_user_enabled=simulated_user_enabled,
                            simulated_user_component_weights=simulated_user_component_weights,
                            simulated_user_metrics=simulated_user_metrics,
                        )

                        if query_group_metrics:
                            metrics.update(query_group_metrics)
                        if simulated_user_metrics:
                            metrics.update(simulated_user_metrics)
                        if simulated_user_enabled and rollout_data_dir:
                            os.makedirs(rollout_data_dir, exist_ok=True)
                            padding_flags = batch.non_tensor_batch.get("simulated_user_is_padding")
                            records_for_trace = batch.non_tensor_batch.get("simulated_user_records", [])
                            for row_index, record in enumerate(records_for_trace):
                                if padding_flags is not None and bool(padding_flags[row_index]):
                                    continue
                                for event in record.get("subtasks", []):
                                    boundary_token = event.get("response_boundary_token")
                                    if boundary_token is not None:
                                        event["normalized_advantage"] = float(
                                            batch.batch["advantages"][row_index, int(boundary_token)].item()
                                        )
                            trace_path = os.path.join(
                                rollout_data_dir, f"simulated_user_step_{self.global_steps}.jsonl"
                            )
                            with open(trace_path, "w", encoding="utf-8") as trace_file:
                                for row_index, record in enumerate(records_for_trace):
                                    if padding_flags is not None and bool(padding_flags[row_index]):
                                        continue
                                    trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            print(f"[SimUser] wrote rollout traces to {trace_path}", flush=True)

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = (
                                self.config.actor_rollout_ref.rollout.multi_turn.enable or simulated_user_enabled
                            )
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # The final full validation can take substantially longer
                    # than one update. Emit grouping diagnostics first so a
                    # threshold sweep can be inspected without waiting for it.
                    if is_last_step and query_group_metrics:
                        print("[QueryGroup] final training-update metrics:", flush=True)
                        for metric_name, metric_value in sorted(query_group_metrics.items()):
                            print(f"{metric_name}: {metric_value}", flush=True)

                    # The static ConvAgent-style baseline has no prescribed
                    # final step.  It monitors a conversation-disjoint subset
                    # of the static training data and ends only once that
                    # metric has both plateaued and stabilized.  The configured
                    # total steps remains a safety ceiling, not model selection.
                    static_monitor_due = (
                        static_monitor_enabled
                        and (self.global_steps + 1) % static_monitor_frequency == 0
                    )
                    static_plateau_reached = False
                    if static_monitor_due:
                        print(
                            f"[{static_baseline_name}] monitor validation after completed "
                            f"step {self.global_steps + 1}",
                            flush=True,
                        )
                        with _timer("testing", timing_raw):
                            static_val_metrics = self._validate()
                        metrics.update(static_val_metrics)
                        if not static_monitor_metric:
                            available = sorted(static_val_metrics)
                            raise ValueError(
                                "Set trainer.static_convagent_monitor_metric; "
                                f"available metrics: {available}"
                            )
                        if static_monitor_metric not in static_val_metrics:
                            raise ValueError(
                                "Static ConvAgent monitor metric is missing: "
                                f"{static_monitor_metric}; available={sorted(static_val_metrics)}"
                            )
                        monitor_score = float(static_val_metrics[static_monitor_metric])
                        if not np.isfinite(monitor_score):
                            raise RuntimeError(
                                f"{static_baseline_name} monitor metric is non-finite: {monitor_score}"
                            )
                        static_monitor_history.append(
                            {"completed_step": float(self.global_steps + 1), "score": monitor_score}
                        )
                        monitor_scores = [entry["score"] for entry in static_monitor_history]
                        static_plateau_reached = monitor_plateau_reached(
                            monitor_scores,
                            patience=static_monitor_patience,
                            min_delta=static_monitor_min_delta,
                            stability_window=static_monitor_stability_window,
                            stability_tolerance=static_monitor_stability_tolerance,
                        )
                        metrics["static_convagent/monitor_score"] = monitor_score
                        metrics["static_convagent/monitor_checks"] = float(len(monitor_scores))
                        metrics["static_convagent/monitor_plateau"] = float(static_plateau_reached)
                        last_val_metrics = static_val_metrics
                        if static_plateau_reached:
                            static_stop_reason = "stable_holdout_plateau"
                            print(
                                f"[{static_baseline_name}] stopping on stable holdout plateau: "
                                f"metric={static_monitor_metric}, score={monitor_score:.6f}, "
                                f"checks={len(monitor_scores)}",
                                flush=True,
                            )

                    # Optional quick validation is useful during long online
                    # simulated-user runs.  It intentionally evaluates only a
                    # capped prefix of packed validation batches; the final
                    # validation below remains the reportable full test set.
                    intermediate_validation_freq = int(
                        self.config.algorithm.get(
                            "simulated_user_intermediate_validation_freq", 0
                        )
                    )
                    intermediate_validation_max_batches = int(
                        self.config.algorithm.get(
                            "simulated_user_intermediate_validation_max_batches", 2
                        )
                    )
                    should_run_intermediate_validation = (
                        simulated_user_enabled
                        and not is_last_step
                        and intermediate_validation_freq > 0
                        and (self.global_steps + 1) % intermediate_validation_freq == 0
                    )
                    if should_run_intermediate_validation:
                        if intermediate_validation_max_batches < 1:
                            raise ValueError(
                                "simulated_user_intermediate_validation_max_batches must be >= 1"
                            )
                        print(
                            "[SimUser] intermediate validation after completed "
                            f"step {self.global_steps + 1}: "
                            f"cap={intermediate_validation_max_batches} packed batches",
                            flush=True,
                        )
                        with _timer("testing", timing_raw):
                            intermediate_val_metrics = self._validate(
                                simulated_user_max_batches=intermediate_validation_max_batches
                            )
                        metrics.update(intermediate_val_metrics)

                    # The final validation is always the complete validation
                    # split.  It never inherits the intermediate smoke cap.
                    if (
                        (self.val_reward_fn is not None or simulated_user_enabled)
                        and is_last_step
                        and not static_monitor_due
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    static_final_step = static_baseline_mode and (
                        static_plateau_reached or is_last_step
                    )
                    if static_baseline_mode and is_last_step and static_stop_reason is None:
                        static_stop_reason = "safety_step_ceiling"

                    should_save_checkpoint = (
                        static_final_step
                        if static_baseline_mode
                        else self.config.trainer.save_freq > 0
                        and (is_last_step or (self.global_steps + 1) % self.config.trainer.save_freq == 0)
                    )
                    if should_save_checkpoint:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                        if static_baseline_mode:
                            selection_path = os.path.join(
                                self.config.trainer.default_local_dir,
                                "static_chatr1_selection.json"
                                if static_chatr1_mode
                                else "static_convagent_selection.json",
                            )
                            selection = {
                                "selection_reason": static_stop_reason,
                                "selected_global_step": self.global_steps,
                                "selected_completed_step": self.global_steps + 1,
                                "monitor_metric": static_monitor_metric,
                                "monitor_history": static_monitor_history,
                                "safety_step_ceiling": self.total_training_steps,
                            }
                            with open(selection_path, "w", encoding="utf-8") as selection_file:
                                json.dump(selection, selection_file, indent=2)
                            print(
                                f"[{static_baseline_name}] final checkpoint selected: "
                                f"{self.config.trainer.default_local_dir}/global_step_{self.global_steps}",
                                flush=True,
                            )

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                
                # Persist/log metrics together with checkpoints, plus the final
                # validation metrics.  This is deliberately independent of
                # ``test_freq`` because validation is final-step-only above.
                if is_last_step or static_final_step or should_save_checkpoint:
                    val_data_dir = self.config.trainer.get("validation_data_dir", None)
                    if val_data_dir:
                        with open(f'{val_data_dir}/metric_step_{self.global_steps}.json', 'w') as f:
                            json.dump(metrics, f)
                    logger.log(data=metrics, step=self.global_steps)

                if is_last_step or static_final_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
