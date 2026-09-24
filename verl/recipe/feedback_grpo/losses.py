"""Weighted GRPO objectives for the alternating user/system recipe.

This module deliberately imports only torch so that the exact weighting and
stop-gradient rules can be tested without a distributed GPU runtime.
"""

from typing import Dict, Mapping, Optional, Tuple

import torch


def trim_decision_padding(batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Trim common left-prompt/right-response padding without shifting labels.

    Call after sequence-parallel gathering (normally on one sequence), never
    before the collective: peers must present identically shaped tensors to
    all-gather. Response tensor columns still begin at the original response
    boundary; full-sequence and response-only fields are sliced separately.
    """
    result = dict(batch)
    response_width = batch["responses"].shape[-1]
    input_width = batch["input_ids"].shape[-1]
    prompt_width = input_width - response_width
    if prompt_width < 1:
        raise ValueError("A generated decision needs at least one prompt token.")
    attention = batch["attention_mask"].bool()
    prompt_columns = attention[:, :prompt_width].any(dim=0).nonzero(as_tuple=True)[0]
    response_columns = attention[:, prompt_width:].any(dim=0).nonzero(as_tuple=True)[0]
    if prompt_columns.numel() == 0 or response_columns.numel() == 0:
        raise ValueError("Dispatch padding must duplicate a real prompt/response row with weight zero.")
    left = int(prompt_columns[0])
    keep_response = int(response_columns[-1]) + 1
    for name in ("input_ids", "attention_mask", "position_ids"):
        result[name] = batch[name][..., left:prompt_width + keep_response]
    for name in ("responses", "response_mask", "old_log_probs", "ref_log_prob", "entropys"):
        if name in batch:
            result[name] = batch[name][..., :keep_response]
    if "advantages" in batch and batch["advantages"].ndim == 2 and batch["advantages"].shape[-1] == response_width:
        result["advantages"] = batch["advantages"][..., :keep_response]
    return result


def weighted_grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_weights: torch.Tensor,
    *,
    ref_log_probs: Optional[torch.Tensor] = None,
    entropy: Optional[torch.Tensor] = None,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    kl_coef: float = 0.001,
    entropy_coef: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Sum globally weighted, token-mean sequence losses without renormalizing.

    ``loss_weights`` are coefficients over the COMPLETE effective training
    batch, not the current microbatch. They must therefore never be divided by
    the number of local rows/microbatches or their local weight sum. The caller
    compensates FSDP gradient averaging separately. A multi-decision episode
    uses segment weights proportional to its generated token counts, which
    makes splitting its trajectory into decision rows loss-invariant.

    Old-policy/reference probabilities, rewards/advantages, masks and weights
    are targets. Only current-policy log probabilities (and optional entropy)
    receive gradients. Zero-weight dispatch padding is ignored completely.
    """
    if log_probs.ndim != 2 or old_log_probs.shape != log_probs.shape:
        raise ValueError("Current and old log probabilities must both have shape [B, T].")
    if response_mask.shape != log_probs.shape:
        raise ValueError("response_mask must have shape [B, T].")
    if loss_weights.numel() != log_probs.shape[0]:
        raise ValueError("Exactly one loss weight is required per sequence row.")
    if not 0 <= clip_ratio_low < 1 or clip_ratio_high < 0:
        raise ValueError("Invalid GRPO clipping interval.")
    if kl_coef < 0 or entropy_coef < 0:
        raise ValueError("KL and entropy coefficients must be nonnegative.")

    weights = loss_weights.detach().to(device=log_probs.device, dtype=log_probs.dtype).reshape(-1)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("loss_weights must be finite and nonnegative.")
    mask = response_mask.detach().to(device=log_probs.device).bool()
    active = mask & weights.gt(0).unsqueeze(-1)
    if ((mask.sum(-1) == 0) & weights.gt(0)).any():
        raise ValueError("Positive-weight rows must contain at least one generated token.")
    token_counts = active.sum(-1).clamp_min(1).to(log_probs.dtype)

    def target(value: torch.Tensor, name: str) -> torch.Tensor:
        value = value.detach().to(device=log_probs.device, dtype=log_probs.dtype)
        if value.ndim == 1 and value.shape[0] == log_probs.shape[0]:
            value = value.unsqueeze(-1)
        try:
            value = value.expand_as(log_probs)
        except RuntimeError as error:
            raise ValueError(f"{name} must broadcast to [B, T].") from error
        if not torch.isfinite(value[active]).all():
            raise ValueError(f"{name} is not finite on generated tokens.")
        return torch.where(active, value, 0.0)

    if not torch.isfinite(log_probs[active]).all():
        raise ValueError("Current-policy log probabilities are not finite on generated tokens.")
    current = torch.where(active, log_probs, 0.0)
    old = target(old_log_probs, "old_log_probs")
    advantage = target(advantages, "advantages")

    def weighted_mean(values: torch.Tensor) -> torch.Tensor:
        # torch.where also excludes NaNs on dispatch-padding/masked tokens.
        means = torch.where(active, values, 0.0).sum(-1) / token_counts
        return (means * weights).sum()

    log_ratio = current - old
    # The outer surrogate/KL clipping already saturates these extreme tails;
    # bounding the exponent prevents inf/NaN intermediates during backward.
    ratio = log_ratio.clamp(-20.0, 20.0).exp()
    clipped = ratio.clamp(1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    unclipped_loss = -advantage * ratio
    clipped_loss = -advantage * clipped
    pg_loss = weighted_mean(torch.maximum(unclipped_loss, clipped_loss))
    clipfrac = weighted_mean((clipped_loss > unclipped_loss).to(log_probs.dtype))

    kl_loss = current.sum() * 0.0
    if kl_coef:
        if ref_log_probs is None:
            raise ValueError("ref_log_probs is required when kl_coef is nonzero.")
        ref = target(ref_log_probs, "ref_log_probs")
        delta = ref - current
        # Same low_var_kl (k3) estimator and cap used by the existing actor.
        kld = (delta.clamp(-20.0, 20.0).exp() - delta - 1.0).clamp(0.0, 10.0)
        kl_loss = weighted_mean(kld)

    entropy_mean = current.sum() * 0.0
    if entropy is not None:
        if entropy.shape != log_probs.shape:
            raise ValueError("entropy must have shape [B, T].")
        if not torch.isfinite(entropy[active]).all():
            raise ValueError("Entropy is not finite on generated tokens.")
        entropy_mean = weighted_mean(entropy if entropy_coef else entropy.detach())
    elif entropy_coef:
        raise ValueError("entropy is required when entropy_coef is nonzero.")

    loss = pg_loss + kl_coef * kl_loss - entropy_coef * entropy_mean
    metrics = {
        "actor/pg_loss": pg_loss.detach(),
        "actor/pg_clipfrac": clipfrac.detach(),
        "actor/ppo_kl": weighted_mean(-log_ratio).detach(),
        "actor/kl_loss": kl_loss.detach(),
        "actor/entropy_loss": entropy_mean.detach(),
        "actor/entropy": entropy_mean.detach(),
        "actor/loss": loss.detach(),
        "actor/loss_weight_sum": weights.sum(),
    }
    return loss, metrics
