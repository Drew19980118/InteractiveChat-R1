"""Opt-in FSDP workers for alternating feedback GRPO.

The ordinary PPO/ConvAgent/TurnPPO workers are unchanged. Each policy lives in
its own Ray worker group, model, optimizer and reference model. One RPC updates
only that policy and takes exactly one optimizer step over its effective batch.
"""

import math
from typing import Dict, Iterable, Mapping, Optional

import torch

from verl import DataProto
from verl.recipe.feedback_grpo.losses import trim_decision_padding, weighted_grpo_loss
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class FeedbackGRPOActor(DataParallelPPOActor):
    def update_feedback_policy(
        self,
        data: DataProto,
        data_parallel_size: int,
        micro_batches: Optional[Iterable[Mapping[str, torch.Tensor]]] = None,
    ) -> Dict[str, float]:
        """Accumulate fixed-snapshot decision rows before a single update.

        Use one sequence per microbatch. All ranks receive the same row count
        after dispatch padding/SP gathering, so all ranks issue an identical
        number of FSDP forwards/backwards even with unequal sequence lengths.
        Padding rows still execute their zero-weight forward/backward.
        """
        if int(self.config.get("ppo_epochs", 1)) != 1:
            raise ValueError("Feedback GRPO currently requires ppo_epochs=1.")
        if data.batch is None or data.batch.batch_size[0] == 0:
            raise ValueError("Do not dispatch an empty feedback update.")
        required = (
            "responses", "input_ids", "attention_mask", "position_ids",
            "old_log_probs", "advantages", "response_mask", "loss_weights",
        )
        missing = set(required) - set(data.batch.keys())
        if missing:
            raise ValueError(f"Missing feedback update tensors: {sorted(missing)}")
        kl_coef = float(self.config.kl_loss_coef) if self.config.use_kl_loss else 0.0
        if kl_coef and self.config.kl_loss_type != "low_var_kl":
            raise ValueError("Feedback GRPO uses the existing low_var_kl regularizer.")
        if kl_coef and "ref_log_prob" not in data.batch:
            raise ValueError("Missing fixed initial-reference log probabilities.")
        temperature = float(data.meta_info["temperature"])
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Update temperature must match a positive sampling temperature.")

        self.actor_module.train()
        self.actor_optimizer.zero_grad(set_to_none=True)
        entropy_coef = float(self.config.entropy_coeff)
        compute_entropy = entropy_coef != 0 or bool(self.config.get("feedback_log_entropy", True))
        clip_ratio = float(self.config.clip_ratio)
        low = self.config.get("clip_ratio_low", None)
        high = self.config.get("clip_ratio_high", None)
        totals = {}
        # No PPO minibatch/epoch loop: generated candidates in each effective
        # GRPO group share their sampling snapshot and are balanced by the
        # explicit global row coefficients. The singleton-first System phase
        # contributes only second-round candidates.
        if micro_batches is None:
            micro_batches = data.batch.split(1)
        for micro_batch in micro_batches:
            micro_batch = trim_decision_padding(micro_batch)
            entropy, log_probs = self._forward_micro_batch(
                micro_batch=micro_batch,
                temperature=temperature,
                calculate_entropy=compute_entropy,
            )
            loss, metrics = weighted_grpo_loss(
                log_probs=log_probs,
                old_log_probs=micro_batch["old_log_probs"],
                advantages=micro_batch["advantages"],
                response_mask=micro_batch["response_mask"],
                loss_weights=micro_batch["loss_weights"],
                ref_log_probs=micro_batch.get("ref_log_prob", None),
                entropy=entropy,
                clip_ratio_low=clip_ratio if low is None else float(low),
                clip_ratio_high=clip_ratio if high is None else float(high),
                kl_coef=kl_coef,
                entropy_coef=entropy_coef,
            )
            # FSDP averages over all ranks; SP peers jointly differentiate a
            # duplicated input batch (Gather backward supplies SP scaling).
            # The remaining averaging is exactly world_size / SP_size.
            (loss * data_parallel_size).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, torch.zeros_like(value)) + value

        grad_norm = self._optimizer_step()
        self.actor_optimizer.zero_grad(set_to_none=True)
        names = sorted(totals)
        metric_vector = torch.stack([totals[key] for key in names])
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(metric_vector)
            # SP peers see identical rows after all-gather; remove duplicates.
            metric_vector /= self.ulysses_sequence_parallel_size
        result = dict(zip(names, metric_vector.detach().cpu().tolist()))
        result["actor/grad_norm"] = float(grad_norm.detach().cpu())
        result["actor/optimizer_step_applied"] = float(torch.isfinite(grad_norm))
        result["actor/kl_coef"] = kl_coef
        return result


class FeedbackGRPOWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            self.actor = FeedbackGRPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_feedback_policy(self, data: DataProto):
        assert self._is_actor
        # The effective batch can still contain multiple generated-token
        # segments per source. Keep it on CPU and stage only a few rank-local
        # decision rows. In the singleton-first System phase, only the n
        # same-prompt second-round candidates contribute loss rows.
        data = data.to("cpu")
        device = torch.cuda.current_device()
        local_summary = torch.tensor(
            [float(data.batch["loss_weights"].sum()), float(len(data))],
            dtype=torch.float64, device=device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_summary)
        if not math.isclose(float(local_summary[0]), 1.0, rel_tol=1e-4, abs_tol=1e-5):
            raise ValueError("Complete feedback-update row weights must sum to one globally.")
        if int(local_summary[1]) != len(data) * self.world_size:
            raise ValueError("All ranks must receive equal row counts (pad with zero-weight rows).")
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(self.actor_optimizer, device_id=torch.cuda.current_device())
        try:
            with self.ulysses_sharding_manager:
                block_size = max(1, int(self.config.actor.get("feedback_update_rows_per_rank", 1)))

                def gpu_micro_batches():
                    # Slicing is along rows only until after SP gathering, so
                    # every rank's collective input shape agrees at each block.
                    for start in range(0, len(data), block_size):
                        block = data[start:start + block_size].to(device)
                        block = self.ulysses_sharding_manager.preprocess_data(block)
                        yield from block.batch.split(1)

                metrics = self.actor.update_feedback_policy(
                    data,
                    data_parallel_size=self.world_size // self.ulysses_sequence_parallel_size,
                    micro_batches=gpu_micro_batches(),
                )
            # _optimizer_step deliberately skips nonfinite gradients. Do not
            # advance the scheduler if the model was not actually updated.
            if metrics["actor/optimizer_step_applied"]:
                self.actor_lr_scheduler.step()
            metrics["actor/lr"] = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["perf/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            metrics["perf/max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
            return DataProto(meta_info={"metrics": metrics})
        finally:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
