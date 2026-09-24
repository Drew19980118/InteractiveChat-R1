"""Two independent, single-GPU Ray/FSDP policy groups for feedback GRPO.

The system and learned-user policies are each 3B and each owns one whole
visible GPU.  A one-rank FSDP group is intentional: fractional Ray GPU
allocation can map two ranks of an NCCL group to the same physical GPU, while
the two policies must remain independent anyway.  This topology eliminates
cross-GPU NCCL and lets both trainable policies coexist safely.  No external
simulator service is used.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import uuid

from .engine import Generation, Observation, TrainItem


def worker_config(args, role):
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[3]
    cfg = OmegaConf.load(root / "verl/trainer/config/ppo_trainer.yaml")
    changes = {
        "data.max_prompt_length": args.max_prompt_length,
        "data.max_response_length": args.max_response_length if role == "system" else args.feedback_length,
        # veRL's FSDP loader indexes this value (``src[-1]``) before calling
        # Path-aware filesystem utilities, so argparse ``Path`` instances are
        # not accepted here.  Keep command-line path handling in ``main`` but
        # cross the veRL config boundary as a plain string.
        "actor_rollout_ref.model.path": str(getattr(args, f"{role}_model")),
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 64,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": args.max_model_len,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.use_torch_compile": False,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.clip_ratio": 0.2,
        "actor_rollout_ref.actor.entropy_coeff": args.entropy_coef,
        "actor_rollout_ref.actor.use_kl_loss": args.kl_coef > 0,
        "actor_rollout_ref.actor.kl_loss_coef": args.kl_coef,
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.optim.lr": args.learning_rate,
        "actor_rollout_ref.actor.optim.total_training_steps": args.max_system_updates * max(1, math.ceil(args.user_updates_per_phase / args.system_updates_per_phase)),
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0,
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": args.sequence_parallel,
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": False,
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": args.max_model_len,
        "actor_rollout_ref.ref.ulysses_sequence_parallel_size": args.sequence_parallel,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": False,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": args.max_model_len,
        "actor_rollout_ref.rollout.n": 1,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        # Each independent policy owns one full GPU and is a one-rank FSDP
        # group. TP must stay one: using all visible GPUs would make a policy
        # claim the other policy's card a second time.
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": args.rollout_memory,
        "actor_rollout_ref.rollout.max_model_len": args.max_model_len,
        "actor_rollout_ref.rollout.max_num_batched_tokens": args.max_model_len,
        "actor_rollout_ref.rollout.max_num_seqs": args.rollout_batch_size,
        "actor_rollout_ref.rollout.enable_chunked_prefill": False,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.free_cache_engine": True,
    }
    for key, value in changes.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    OmegaConf.resolve(cfg)
    return cfg.actor_rollout_ref


class RayBackend:
    def __init__(self, args):
        import ray
        import torch
        from transformers import AutoTokenizer
        from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
        from .workers import FeedbackGRPOWorker
        self.args, self.groups = args, {}
        self.policy_world_size = 1
        if ray.is_initialized():
            raise RuntimeError("Run this recipe as a standalone process with its own Ray cluster")
        if not torch.cuda.is_available() or torch.cuda.device_count() != args.n_gpus:
            raise RuntimeError("Visible CUDA devices must match --n-gpus exactly")
        # address=local ignores another job's RAY_ADDRESS; never stop its cluster.
        ray.init(address="local", num_gpus=args.n_gpus, include_dashboard=False,
                 runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "false", "NCCL_DEBUG": "WARN"}})
        self.tokenizers = {role: AutoTokenizer.from_pretrained(getattr(args, f"{role}_model"))
                           for role in ("system", "user")}
        run_id = "feedback_" + uuid.uuid4().hex[:10]
        for role in ("system", "user"):
            print(f"[FeedbackGRPO] initializing independent {role} policy", flush=True)
            # A distinct full-GPU placement group for each role forces Ray to
            # allocate different physical GPUs. Do not use max_colocate_count
            # here: its fractional GPU scheduling is invalid for two NCCL
            # ranks, as it can place both ranks on one GPU.
            pool = RayResourcePool([self.policy_world_size], use_gpu=True,
                                   max_colocate_count=1, name_prefix=f"{run_id}_{role}")
            group = RayWorkerGroup(resource_pool=pool,
                ray_cls_with_init=RayClassWithInitArgs(ray.remote(FeedbackGRPOWorker),
                                                       config=worker_config(args, role), role="actor_rollout_ref"),
                name_prefix=f"{run_id}_{role}")
            self.groups[role] = group
            group.init_model()
        from tools_server.util import MessageClient
        self.tools = MessageClient()

    def _prompt_tokens(self, role, messages, initial):
        tokenizer = self.tokenizers[role]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if role == "system":
            prompt += "<think>"  # Native ConvAgent's fixed reasoning prefix, not a sampled token.
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_limit = self.args.max_response_length if role == "system" else self.args.feedback_length
        limit = self.args.max_model_len - response_limit - 32
        if initial:
            limit = min(limit, self.args.max_prompt_length)
        return ids[-limit:]

    @staticmethod
    def _pad_rows(rows, pad_token):
        import torch
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), pad_token, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, -len(row):] = torch.tensor(row, dtype=torch.long)
            mask[i, -len(row):] = 1
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        return ids, mask, positions

    def generate(self, role, messages, *, greedy, initial=False):
        from verl import DataProto
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        tokenizer = self.tokenizers[role]
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        result = []
        for start in range(0, len(messages), self.args.rollout_batch_size):
            chunk = messages[start:start + self.args.rollout_batch_size]
            rows = [self._prompt_tokens(role, m, initial) for m in chunk]
            ids, mask, positions = self._pad_rows(rows, pad)
            data = DataProto.from_dict(tensors={"input_ids": ids, "attention_mask": mask, "position_ids": positions},
                                       meta_info={"do_sample": not greedy})
            data, padding = pad_dataproto_to_divisor(data, self.policy_world_size)
            output = unpad_dataproto(self.groups[role].generate_sequences(data), padding)
            response_width = output.batch["responses"].shape[-1]
            for i, prompt_ids in enumerate(rows):
                response_ids = output.batch["responses"][i][output.batch["attention_mask"][i, -response_width:].bool()].tolist()
                if not response_ids:
                    raise RuntimeError("Model returned zero tokens, not even EOS")
                text = tokenizer.decode(response_ids, skip_special_tokens=True)
                result.append(Generation(prompt_ids, response_ids, ("<think>" if role == "system" else "") + text))
        return result

    def search(self, queries):
        tasks = [{"idx": i, "question": example.query, "think": "",
                  "tool_call": {"name": "web_search", "arguments": {"query": [query]}},
                  "total_number": len(queries)} for i, (example, query) in enumerate(queries)]
        results = self.tools.submit_tasks(tasks)
        if len(results) != len(tasks):
            raise RuntimeError("Retriever returned mismatched task count")
        observations = []
        # Preserve request order even if a handler implementation returns in completion order.
        indexed = {int(result["idx"]): result for result in results}
        for i in range(len(tasks)):
            content = indexed[i].get("content")
            if isinstance(content, str) and content.startswith("Error:"):
                raise RuntimeError(f"Retriever failure (not a policy reward): {content}")
            try:
                payload = json.loads(content) if isinstance(content, str) else content
            except (TypeError, json.JSONDecodeError) as error:
                raise RuntimeError("Invalid retriever observation") from error
            if not isinstance(payload, list):
                raise RuntimeError("Retriever must return a list of query results")
            passages = []
            for query_result in payload:
                if query_result.get("error"):
                    raise RuntimeError(f"Retriever query failed: {query_result['error']}")
                for page in query_result.get("web_page_info_list", []):
                    passages.append({"passage_id": str(page.get("passage_id") or ""),
                                     "passage_text": str(page.get("quick_summary", page.get("passage_text", "")) or "")})
            observations.append(Observation(json.dumps(payload, ensure_ascii=False), passages[:3]))
        return observations

    def _training_data(self, role, items):
        import torch
        from verl import DataProto
        tokenizer = self.tokenizers[role]
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        prompts, pmask, _ = self._pad_rows([item.generation.prompt_ids for item in items], pad)
        width = max(len(item.generation.response_ids) for item in items)
        responses = torch.full((len(items), width), pad, dtype=torch.long)
        rmask = torch.zeros_like(responses)
        for i, item in enumerate(items):
            tokens = item.generation.response_ids
            responses[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            rmask[i, :len(tokens)] = 1
        attention = torch.cat((pmask, rmask), dim=-1)
        return DataProto.from_dict(tensors={
            "prompts": prompts, "responses": responses, "input_ids": torch.cat((prompts, responses), -1),
            "attention_mask": attention, "position_ids": (attention.cumsum(-1) - 1).clamp_min(0),
            "response_mask": rmask, "advantages": torch.tensor([i.advantage for i in items], dtype=torch.float32),
            "loss_weights": torch.tensor([i.weight for i in items], dtype=torch.float32)},
            meta_info={"temperature": 1.0})

    def update(self, role: str, items: list[TrainItem]):
        import torch
        from verl.protocol import pad_dataproto_to_divisor
        from .engine import validate_train_items
        validate_train_items(items)
        if not items:
            raise ValueError("Cannot update from an empty batch")
        data = self._training_data(role, items)
        data, padding = pad_dataproto_to_divisor(data, self.policy_world_size)
        if padding:
            data.batch["loss_weights"][-padding:] = 0
        group = self.groups[role]
        old_parts, ref_parts = [], []
        chunk_size = max(
            self.policy_world_size,
            (self.args.logprob_batch_size // self.policy_world_size) * self.policy_world_size,
        )
        for start in range(0, len(data), chunk_size):
            chunk = data[start:start + chunk_size]
            old_parts.append(group.compute_log_prob(chunk).batch["old_log_probs"].detach().cpu())
            if self.args.kl_coef:
                ref_parts.append(group.compute_ref_log_prob(chunk).batch["ref_log_prob"].detach().cpu())
        data.batch["old_log_probs"] = torch.cat(old_parts)
        if ref_parts:
            data.batch["ref_log_prob"] = torch.cat(ref_parts)
        output = group.update_feedback_policy(data)
        raw = output.meta_info["metrics"]
        # Dispatch concatenates identical all-reduced worker metrics into lists.
        metrics = {key: sum(value) / len(value) if isinstance(value, list) else float(value)
                   for key, value in raw.items()}
        if metrics["actor/optimizer_step_applied"] != 1:
            raise FloatingPointError(f"{role} update skipped due to nonfinite gradients; aborting instead of counting a step")
        return {f"{role}/{key}": value for key, value in metrics.items()}

    def save_policy(self, role, path, step):
        self.groups[role].save_checkpoint(local_path=str(path), hdfs_path=None, global_step=step, max_ckpt_to_keep=None)

    def load_pair(self, path):
        for role in ("system", "user"):
            self.groups[role].load_checkpoint(local_path=str(path / role), del_local_after_load=False)

    def export_pair(self, root):
        for role in ("system", "user"):
            self.groups[role].export_actor_hf(export_path=str(root / role), max_shard_size="2GB")

    def close(self):
        import ray
        ray.shutdown()
