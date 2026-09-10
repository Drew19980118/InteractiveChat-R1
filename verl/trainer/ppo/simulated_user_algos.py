"""Sparse, stratified GRPO advantage for simulated-user dialogues."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import torch


def compute_simulated_user_turn_gae_advantage(
    *,
    turn_rewards: torch.Tensor,
    turn_reward_mask: torch.Tensor,
    turn_value_mask: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    rollout_ids: np.ndarray,
    row_subtasks: torch.Tensor,
    event_orders: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Turn-level GAE for flattened simulated-user action rows.

    Each row is one immutable ``observation -> complete policy action`` event.
    The critic value is read only at the first generated token, which represents
    the observation before the action.  Terminal action reward is stored at
    the final generated token, but is reduced to one scalar ``r_t`` here.
    The next value comes from the *next row* in the same
    ``(dialogue, rollout, subtask)`` chain, because that row is the only exact
    serialization of the environment's next observation (tool result or user
    feedback included).

    Actor advantages are broadcast to every generated token of the action;
    critic returns are supervised only at the corresponding first-token state
    via ``sim_user_turn_ppo_value_mask`` in the FSDP critic.
    """
    expected_shape = tuple(response_mask.shape)
    for name, tensor in {
        "turn_rewards": turn_rewards,
        "turn_reward_mask": turn_reward_mask,
        "turn_value_mask": turn_value_mask,
        "values": values,
    }.items():
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"simulated-user Turn-PPO {name} shape {tuple(tensor.shape)} "
                f"does not match response mask {expected_shape}"
            )
    batch_size, _response_length = expected_shape
    if len(index) != batch_size or len(rollout_ids) != batch_size:
        raise ValueError("simulated-user Turn-PPO uid/rollout metadata length mismatch")
    if tuple(row_subtasks.shape) != (batch_size,) or tuple(event_orders.shape) != (batch_size,):
        raise ValueError(
            "simulated-user Turn-PPO row_subtasks and event_orders must contain one value per row"
        )
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("Turn-PPO gamma must be in [0, 1]")
    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError("Turn-PPO lambda must be in [0, 1]")

    advantages = torch.zeros_like(values, dtype=torch.float32)
    returns = torch.zeros_like(values, dtype=torch.float32)
    chains: dict[tuple[str, str, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    for row in range(batch_size):
        valid_positions = response_mask[row].to(torch.bool).nonzero(as_tuple=True)[0]
        boundary_positions = turn_reward_mask[row].to(torch.bool).nonzero(as_tuple=True)[0]
        value_positions = turn_value_mask[row].to(torch.bool).nonzero(as_tuple=True)[0]
        # DP padding rows are intentionally all-zero and must not participate.
        if not len(valid_positions):
            if len(boundary_positions) or len(value_positions):
                raise RuntimeError("Turn-PPO padding row unexpectedly carries reward/value metadata")
            continue
        if len(value_positions) != 1:
            raise RuntimeError("each Turn-PPO action row must have exactly one value-state position")
        if int(value_positions[0].item()) != int(valid_positions[0].item()):
            raise RuntimeError("Turn-PPO value state must be the first generated token of its action")
        if len(boundary_positions) > 1:
            raise RuntimeError("each Turn-PPO action row may have at most one immediate reward")
        subtask = int(row_subtasks[row].item())
        if subtask < 0:
            raise RuntimeError("Turn-PPO action row is missing its source subtask id")
        chains[(str(index[row]), str(rollout_ids[row]), subtask)].append(
            (
                int(event_orders[row].item()),
                row,
                int(value_positions[0].item()),
                int(boundary_positions[0].item()) if len(boundary_positions) else -1,
            )
        )

    action_count = 0
    rewarded_actions = 0
    for key, chain in chains.items():
        chain.sort(key=lambda item: item[0])
        orders = [order for order, _row, _value_position, _reward_position in chain]
        if len(orders) != len(set(orders)):
            raise RuntimeError(f"Turn-PPO duplicate action order in trajectory chain {key}")
        action_count += len(chain)

        next_advantage = 0.0
        next_value = 0.0
        for _order, row, value_position, reward_position in reversed(chain):
            value = float(values[row, value_position].detach().item())
            if not np.isfinite(value):
                raise RuntimeError("Turn-PPO critic produced a non-finite value")
            reward = 0.0
            if reward_position >= 0:
                reward = float(turn_rewards[row, reward_position].item())
                if not np.isfinite(reward):
                    raise RuntimeError("Turn-PPO encountered a non-finite terminal reward")
                rewarded_actions += 1
            delta = reward + float(gamma) * next_value - value
            action_advantage = delta + float(gamma) * float(lam) * next_advantage
            action_return = action_advantage + value

            valid = response_mask[row].to(torch.bool)
            advantages[row, valid] = action_advantage
            returns[row, value_position] = action_return
            next_advantage = action_advantage
            next_value = value

    # Whiten per macro action, not per generated token.  Broadcasting happens
    # only after normalization so long ``<think>`` spans cannot change the
    # relative scale of two action advantages.
    action_advantages = []
    for chain in chains.values():
        for _order, row, value_position, _reward_position in chain:
            action_advantages.append(advantages[row, value_position])
    if action_advantages:
        stacked = torch.stack(action_advantages)
        mean = stacked.mean()
        std = stacked.std(unbiased=False)
        normalized = (stacked - mean) / (std + 1e-8)
        cursor = 0
        for chain in chains.values():
            for _order, row, _value_position, _reward_position in chain:
                valid = response_mask[row].to(torch.bool)
                advantages[row, valid] = normalized[cursor]
                cursor += 1

    metrics = {
        "turn_ppo/actions": float(action_count),
        "turn_ppo/rewarded_terminal_actions": float(rewarded_actions),
        "turn_ppo/trajectory_chains": float(len(chains)),
        "turn_ppo/value_supervision_states": float(turn_value_mask.sum().item()),
    }
    return advantages, returns, metrics


def compute_simulated_user_sparse_grpo_advantage(
    *,
    component_values: dict[str, torch.Tensor],
    component_masks: dict[str, torch.Tensor],
    turn_boundary_mask: torch.Tensor,
    subtask_ids: torch.Tensor,
    response_depths: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    rollout_ids: Optional[np.ndarray] = None,
    event_orders: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    component_weights: Optional[dict[str, float]] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute local sparse return after independent channel normalization.

    A group contains only trajectories which actually reached the same
    ``(dialogue uid, subtask, response depth)``.  No reward is propagated
    across source subtask boundaries; zero-reward tool turns nevertheless get
    the return of the user-visible response that follows them.
    """
    if not component_values:
        raise ValueError("simulated-user GRPO needs at least one reward component")
    bsz, seq_len = response_mask.shape
    expected_shape = (bsz, seq_len)
    device = response_mask.device
    if turn_boundary_mask.shape != expected_shape:
        raise ValueError("simulated-user boundary mask shape does not match response mask")
    if subtask_ids.shape != expected_shape or response_depths.shape != expected_shape:
        raise ValueError("simulated-user subtask/depth tensors do not match response mask")
    if len(index) != bsz:
        raise ValueError(f"simulated-user uid length mismatch: expected {bsz}, got {len(index)}")
    if (rollout_ids is None) != (event_orders is None):
        raise ValueError("simulated-user event-row return needs both rollout_ids and event_orders")
    if rollout_ids is not None and len(rollout_ids) != bsz:
        raise ValueError(f"simulated-user rollout-id length mismatch: expected {bsz}, got {len(rollout_ids)}")
    if event_orders is not None and tuple(event_orders.shape) != (bsz,):
        raise ValueError(
            "simulated-user event_orders must be one integer order per response row; "
            f"expected={(bsz,)}, got={tuple(event_orders.shape)}"
        )

    normalized = torch.zeros(expected_shape, dtype=torch.float32, device=device)
    weights = component_weights or {}
    metrics: dict[str, float] = {
        "sim_user/events": float(turn_boundary_mask.sum().item()),
        "sim_user/normalization_groups": 0.0,
        "sim_user/singleton_groups": 0.0,
        "sim_user/zero_variance_groups": 0.0,
    }

    for component_name, values in component_values.items():
        mask = component_masks.get(component_name)
        if values.shape != expected_shape or mask is None or mask.shape != expected_shape:
            raise ValueError(f"simulated-user component {component_name!r} has incompatible tensors")
        groups: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
        for row, column in mask.to(torch.bool).nonzero(as_tuple=False).tolist():
            if not bool(turn_boundary_mask[row, column]):
                raise RuntimeError(f"component {component_name!r} is not on an assistant turn boundary")
            subtask = int(subtask_ids[row, column].item())
            depth = int(response_depths[row, column].item())
            if subtask < 0 or depth < 0:
                raise RuntimeError(f"component {component_name!r} has missing subtask/depth metadata")
            groups[(str(index[row]), subtask, depth)].append((row, column))

        singletons = 0
        zero_variance = 0
        for records in groups.values():
            if len(records) < 2:
                singletons += 1
                continue
            rows = torch.tensor([row for row, _ in records], dtype=torch.long, device=device)
            columns = torch.tensor([column for _, column in records], dtype=torch.long, device=device)
            group_values = values[rows, columns].to(torch.float32)
            if not torch.isfinite(group_values).all():
                raise RuntimeError(f"component {component_name!r} contains a non-finite reward")
            centered = group_values - group_values.mean()
            variance = (centered.square()).mean()
            if float(variance.item()) <= 1e-12:
                zero_variance += 1
                continue
            if norm_adv_by_std_in_grpo:
                centered = centered / (torch.sqrt(variance + 1e-8) + epsilon)
            normalized[rows, columns] += float(weights.get(component_name, 1.0)) * centered

        metrics["sim_user/normalization_groups"] += float(len(groups))
        metrics["sim_user/singleton_groups"] += float(singletons)
        metrics["sim_user/zero_variance_groups"] += float(zero_variance)
        metrics[f"sim_user/{component_name}_groups"] = float(len(groups))

    advantages = torch.zeros_like(normalized)
    if event_orders is not None:
        # Context is allowed to change between actions: e.g. a new full top-k
        # can evict text from an earlier retrieval.  In that representation
        # every row is one exact action snapshot, and returns must therefore
        # be accumulated across rows belonging to the same rollout/subtask.
        chains: dict[tuple[str, str, int], list[tuple[int, int, int]]] = defaultdict(list)
        for row in range(bsz):
            positions = turn_boundary_mask[row].nonzero(as_tuple=True)[0].tolist()
            # ``ray_trainer`` may add at most world_size-1 zero-loss rows so
            # Ray can split the action rows equally. They must not enter a
            # sparse-GRPO group or trajectory return chain.
            if not positions and not bool(response_mask[row].to(torch.bool).any()):
                continue
            if len(positions) != 1:
                raise RuntimeError(
                    "simulated-user event-row mode requires exactly one action boundary per response row"
                )
            position = positions[0]
            subtask = int(subtask_ids[row, position].item())
            if subtask < 0:
                raise RuntimeError("simulated-user event-row boundary is missing a subtask id")
            key = (str(index[row]), str(rollout_ids[row]), subtask)
            chains[key].append((int(event_orders[row].item()), row, position))

        for key, chain in chains.items():
            chain.sort(key=lambda item: item[0])
            orders = [order for order, _row, _position in chain]
            if len(set(orders)) != len(orders):
                raise RuntimeError(f"simulated-user duplicate action order in trajectory chain {key}")
            next_return = 0.0
            for _order, row, position in reversed(chain):
                value = float(normalized[row, position].item()) + gamma * next_return
                valid = response_mask[row, : position + 1].to(torch.bool)
                if valid.any():
                    view = advantages[row, : position + 1]
                    view[valid] = value
                    advantages[row, : position + 1] = view
                next_return = value
    else:
        for row in range(bsz):
            positions = turn_boundary_mask[row].nonzero(as_tuple=True)[0].tolist()
            if not positions:
                continue
            next_return = 0.0
            current_subtask: Optional[int] = None
            local_returns: list[tuple[int, float]] = []
            for position in reversed(positions):
                subtask = int(subtask_ids[row, position].item())
                if subtask < 0:
                    raise RuntimeError("simulated-user boundary is missing a subtask id")
                if current_subtask is None or subtask != current_subtask:
                    current_subtask = subtask
                    next_return = 0.0
                value = float(normalized[row, position].item()) + gamma * next_return
                local_returns.append((position, value))
                next_return = value
            local_returns.reverse()
            previous_end = 0
            for position, value in local_returns:
                valid = response_mask[row, previous_end : position + 1].to(torch.bool)
                if valid.any():
                    view = advantages[row, previous_end : position + 1]
                    view[valid] = value
                    advantages[row, previous_end : position + 1] = view
                previous_end = position + 1

    metrics["sim_user/nonzero_advantage_tokens"] = float((advantages != 0).sum().item())
    return advantages, advantages.clone(), metrics
