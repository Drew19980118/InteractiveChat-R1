"""Analytic tests for the alternating recipe's effective-batch objective."""

import importlib.util
from pathlib import Path

import pytest
import torch


_path = Path(__file__).resolve().parents[3] / "verl/recipe/feedback_grpo/losses.py"
_spec = importlib.util.spec_from_file_location("feedback_grpo_losses", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
weighted_grpo_loss = _module.weighted_grpo_loss
trim_decision_padding = _module.trim_decision_padding


def test_microbatch_accumulation_matches_full_weighted_loss_and_gradient():
    initial = torch.tensor([[-2.0, -2.4, -3.0], [-1.2, -1.5, -4.0], [-1.0, -1.0, -1.0]])
    old = initial - torch.tensor([[0.1, 0.0, 0.0], [-0.1, 0.1, 0.0], [0.0, 0.0, 0.0]])
    ref = initial - 0.2
    mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]])
    weights = torch.tensor([0.4, 0.6, 0.0])
    advantages = torch.tensor([1.0, -1.0, 99.0])
    full = initial.clone().requires_grad_()
    loss, _ = weighted_grpo_loss(full, old, advantages, mask, weights, ref_log_probs=ref)
    loss.backward()
    split = initial.clone().requires_grad_()
    total = 0.0
    for row in range(3):
        piece, _ = weighted_grpo_loss(
            split[row:row + 1], old[row:row + 1], advantages[row:row + 1],
            mask[row:row + 1], weights[row:row + 1], ref_log_probs=ref[row:row + 1],
        )
        piece.backward()
        total += piece.detach()
    torch.testing.assert_close(total, loss.detach())
    torch.testing.assert_close(split.grad, full.grad)
    assert torch.count_nonzero(split.grad[2]) == 0
    assert split.grad[1, 2] == 0


def test_two_rounds_have_equal_mass_despite_different_group_counts():
    # Two first responses versus two groups of two second responses.
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    weights = torch.tensor([0.25, 0.25, 0.125, 0.125, 0.125, 0.125])
    logp = torch.full((6, 2), -2.0, requires_grad=True)
    loss, _ = weighted_grpo_loss(logp, logp.detach(), advantages, torch.ones_like(logp), weights, kl_coef=0)
    loss.backward()
    torch.testing.assert_close(logp.grad[:2].abs().sum(), logp.grad[2:].abs().sum())


def test_segment_token_weighting_equals_one_episode_token_mean():
    logp = torch.tensor([[-2.0, -2.2, -2.4, -2.6]], requires_grad=True)
    old = logp.detach() - 0.1
    whole, _ = weighted_grpo_loss(logp, old, torch.tensor([0.7]), torch.ones_like(logp), torch.ones(1), kl_coef=0)
    whole.backward()
    whole_grad = logp.grad.clone()
    split = logp.detach().clone().requires_grad_()
    for start, end in [(0, 1), (1, 4)]:
        piece, _ = weighted_grpo_loss(
            split[:, start:end], old[:, start:end], torch.tensor([0.7]),
            torch.ones_like(split[:, start:end]), torch.tensor([(end - start) / 4]), kl_coef=0,
        )
        piece.backward()
    torch.testing.assert_close(split.grad, whole_grad)


def test_targets_detached_and_zero_advantage_has_no_policy_gradient():
    logp = torch.full((2, 3), -2.0, requires_grad=True)
    old = logp.detach().clone().requires_grad_()
    ref = (logp.detach() - 0.2).requires_grad_()
    advantages = torch.zeros(2, requires_grad=True)
    weights = torch.tensor([0.5, 0.5], requires_grad=True)
    loss, metrics = weighted_grpo_loss(logp, old, advantages, torch.ones_like(logp), weights, ref_log_probs=ref)
    loss.backward()
    assert metrics["actor/pg_loss"] == 0
    assert logp.grad.abs().sum() > 0  # KL still regularizes a zero-variance group.
    assert old.grad is ref.grad is advantages.grad is weights.grad is None


def test_clipping_uses_advantage_sign():
    old = torch.full((2, 1), -2.0)
    logp = (old + torch.tensor([[2.0], [-2.0]])).requires_grad_()
    loss, metrics = weighted_grpo_loss(
        logp, old, torch.tensor([1.0, -1.0]), torch.ones_like(logp), torch.tensor([0.5, 0.5]), kl_coef=0,
    )
    torch.testing.assert_close(loss, torch.tensor(-0.2))  # (-1.2 + 0.8) / 2.
    assert metrics["actor/pg_clipfrac"] == 1
    loss.backward()
    assert torch.count_nonzero(logp.grad) == 0


def test_masked_and_zero_weight_padding_can_contain_nan():
    logp = torch.tensor([[-2.0, float("nan")], [float("nan"), float("nan")]], requires_grad=True)
    old = logp.detach().clone()
    loss, _ = weighted_grpo_loss(
        logp, old, torch.tensor([1.0, float("nan")]),
        torch.tensor([[1, 0], [1, 1]]), torch.tensor([1.0, 0.0]), kl_coef=0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    torch.testing.assert_close(logp.grad, torch.tensor([[-1.0, 0.0], [0.0, 0.0]]))


def test_global_weighting_survives_simulated_dp_average():
    x = torch.full((4, 2), -2.0, requires_grad=True)
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
    advantage = torch.tensor([1.0, -1.0, 2.0, -2.0])
    full, _ = weighted_grpo_loss(x, x.detach(), advantage, torch.ones_like(x), weights, kl_coef=0)
    full.backward()
    expected = x.grad.clone()
    grads = []
    for rank in range(2):
        local = x.detach().clone().requires_grad_()
        selection = slice(2 * rank, 2 * rank + 2)
        loss, _ = weighted_grpo_loss(
            local[selection], x.detach()[selection], advantage[selection],
            torch.ones_like(local[selection]), weights[selection], kl_coef=0,
        )
        (2 * loss).backward()
        grads.append(local.grad)
    torch.testing.assert_close(torch.stack(grads).mean(0), expected)


def test_positive_weight_empty_sequence_is_rejected():
    x = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="at least one generated token"):
        weighted_grpo_loss(x, x, torch.ones(1), torch.zeros_like(x), torch.ones(1), kl_coef=0)


def test_trim_keeps_prompt_response_boundary_and_prediction_targets_aligned():
    batch = {
        "input_ids": torch.tensor([[0, 0, 11, 12, 31, 32, 0, 0]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]]),
        "position_ids": torch.tensor([[0, 0, 0, 1, 2, 3, 0, 0]]),
        "responses": torch.tensor([[31, 32, 0, 0]]),
        "response_mask": torch.tensor([[1, 1, 0, 0]]),
        "old_log_probs": torch.tensor([[-1.0, -2.0, 0, 0]]),
        "ref_log_prob": torch.tensor([[-2.0, -3.0, 0, 0]]),
        "advantages": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "loss_weights": torch.tensor([0.25]),
    }
    trimmed = trim_decision_padding(batch)
    assert trimmed["input_ids"].tolist() == [[11, 12, 31, 32]]
    assert trimmed["responses"].tolist() == [[31, 32]]
    assert trimmed["position_ids"].tolist() == [[0, 1, 2, 3]]
    assert trimmed["old_log_probs"].tolist() == [[-1.0, -2.0]]
    assert trimmed["advantages"].tolist() == [[0.5, 0.5]]
    assert trimmed["loss_weights"].tolist() == [0.25]
