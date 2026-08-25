"""Sparse, stratified GRPO advantage for simulated-user dialogues."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import torch


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
