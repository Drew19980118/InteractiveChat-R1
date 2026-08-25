# =============================================================================
# Based on the Search-R1 example from the Search-R1 project.
#
# Original Authors: Jin Bowen, Zeng Hansi, Yue Zhenrui, Wang Dong, Zamani Hamed, Han Jiawei
#
# License: Apache 2.0
# Project URL: https://github.com/PeterGriffinJin/Search-R1
# =============================================================================

import torch
import copy
import re
import os
import json
from typing import Any, List, Dict, Tuple, Optional

# Verification flag for vectorized GT LogProb computation
# Set IGPO_VERIFY_VECTORIZED=true to enable comparison between vectorized and original mode
VERIFY_VECTORIZED = os.environ.get('IGPO_VERIFY_VECTORIZED', '').lower() in ('true', '1', 'yes')
if VERIFY_VECTORIZED:
    print("[IGPO] Verification mode enabled: will compare vectorized vs original results")

import math

from dataclasses import dataclass
from tensordict import TensorDict
from scrl.llm_agent.tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.reward_score.ground_truth import select_answer_ground_truth
import numpy as np
import traceback
import torch.nn.functional as F
from tools_server.util import MessageClient
from tools_server.initialize_prompts import SYSTEM_PROMPT


@dataclass
class GenerationConfig:
    max_turns: int
    num_gpus: int
    data_writing_path: str = None
    model_name: str = None
    n: int = 1
    project_name: str = None
    experiment_name: str = None
    search_engine: str = "online_search"
    nnodes: int = 1
    oss_access_key_id: str = ''
    oss_access_key_secret: str = ''
    oss_endpoint: str = ''
    system_prompt: Optional[str] = SYSTEM_PROMPT
    codeact_env_disabled: bool = True
    # info_gain_type: "prob_diff" (probability difference) or "log_prob_diff" (log probability difference)
    info_gain_type: str = "prob_diff"
    # ``semantic`` records E5 query vectors for Query Grouped-IGPO. Keeping
    # this opt-in avoids any network call or metadata change for legacy IGPO.
    query_group_advantage: str = "disabled"
    # Enable the extra first-turn rewrite-vs-original retrieval reward.
    rewrite_bound_advantage: bool = False

    rewrite_bound_topk: int = 3
    # The handler enforces this cap; the generator uses it to state the same
    # contract in the system prompt.
    max_search_queries: int = 3
    # Allow <think>...</think><nonanswer></nonanswer> as a legal terminal action.
    allow_nonanswer: bool = False
    # Keep the GT-logprob implementation choice in the rollout config rather
    # than relying on a process-global environment fallback.
    use_vectorized_gt_logprob: bool = False
    # The rollout engine's context window.  Multi-turn tool results are part of
    # the prompt, so this must bound both generation and GT log-prob scoring.
    max_model_len: Optional[int] = None
    max_response_length: int = 512
    context_safety_margin: int = 32
    

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        client = None,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.system_prompt = config.system_prompt or ""
        if config.allow_nonanswer:
            # The base prompt says there are two forms. This appended instruction
            # deliberately overrides that wording only for action-labelled data.
            self.system_prompt += (
                "\n\nFor this task, there is one additional legal terminal form:\n"
                "<think>YOUR THINKING PROCESS</think>\n"
                "<nonanswer></nonanswer>\n"
                "Use it only when the question should receive no answer. The "
                "nonanswer tag must contain no text."
            )
        if config.rewrite_bound_advantage:
            self.system_prompt += (
                "\n\nOn your first web_search tool call, rewrite the final user "
                "question into a standalone retrieval query. Resolve pronouns, "
                "ellipsis, and references using the prior conversation. Do not "
                "add unsupported facts. The first web_search must contain exactly "
                "one non-empty query string: that standalone rewrite."
            )
        if config.max_search_queries == 1:
            self.system_prompt += (
                "\n\nFor every web_search tool call in this experiment, "
                "arguments.query must be a JSON list containing exactly one "
                "non-empty search query. Do not submit multiple alternative queries."
            )
        self.codeact_env_disabled = config.codeact_env_disabled

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id
        ))

        self.client = client

    def _model_context_window(self) -> Optional[int]:
        """Return a valid rollout context window, or ``None`` when unbounded."""
        try:
            max_model_len = int(self.config.max_model_len)
        except (TypeError, ValueError):
            return None
        return max_model_len if max_model_len > 0 else None

    def _prompt_token_budget(self, pseudo_response_tokens: int = 0) -> Optional[int]:
        """Reserve room for generation / pseudo GT tokens inside the context window.

        A search trajectory is repeatedly serialized into ``messages_list``.  A
        long retrieved passage therefore affects both the next rollout prompt
        and the pseudo response used to compute information gain.  Reserving
        the larger of the normal rollout response and pseudo GT response keeps
        both operations inside the same model context window.
        """
        max_model_len = self._model_context_window()
        if max_model_len is None:
            return None

        try:
            max_response_length = max(0, int(self.config.max_response_length))
        except (TypeError, ValueError):
            max_response_length = 0
        try:
            safety_margin = max(0, int(self.config.context_safety_margin))
        except (TypeError, ValueError):
            safety_margin = 0

        reserved_tokens = max(max_response_length, pseudo_response_tokens) + safety_margin
        if reserved_tokens >= max_model_len:
            raise ValueError(
                "The configured context window is too small for one response: "
                f"max_model_len={max_model_len}, reserved_tokens={reserved_tokens}."
            )
        return max_model_len - reserved_tokens

    def _tokenize_with_left_truncation(
        self,
        texts: List[str],
        *,
        max_length: Optional[int],
        padding: bool = True,
    ):
        """Tokenize while retaining the newest tool context when a cap is set."""
        if max_length is None:
            return self.tokenizer(texts, return_tensors="pt", padding=padding)

        old_truncation_side = getattr(self.tokenizer, "truncation_side", "right")
        try:
            self.tokenizer.truncation_side = "left"
            return self.tokenizer(
                texts,
                return_tensors="pt",
                padding=padding,
                truncation=True,
                max_length=max_length,
            )
        finally:
            self.tokenizer.truncation_side = old_truncation_side

    def _update_right_side(self, original_right_side: Dict, 
                           cur_responses: torch.Tensor,
                           next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side of rollings."""
        if next_obs_ids is not None:
            responses = self.tensor_fn.concatenate_with_padding(
                [original_right_side['responses'], cur_responses, next_obs_ids],
                pad_to_left=False
            )
        else:
            responses = self.tensor_fn.concatenate_with_padding(
                [original_right_side['responses'], cur_responses],
                pad_to_left=False
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        
        return {'responses': responses[:, :effective_len]}

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, next_obs_ids: torch.Tensor) -> DataProto:
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        return DataProto.from_dict({
                'input_ids': new_input_ids[:, -effective_len:],
                'position_ids': new_position_ids[:, -effective_len:],
                'attention_mask': new_attention_mask[:, -effective_len:]
            })

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']
        return next_obs_ids
        
    def postprocess_predictions(self, rollings_active: DataProto, gen_output: DataProto) -> Tuple[List[int], List[bool]]:
        """Postprocess predictions to remove padding and convert to list of strings."""
        """return: list of query contents including history"""

        pass
        return [{"prompt":""} for _ in range(rollings_active.batch['input_ids'].shape[0])]


    def execute_predictions(
        self, tool_call_list, total_number
    ) :
        query_contents = [{"idx": tool_call[0], "question": tool_call[1], "think": tool_call[2],
                           "tool_call": tool_call[3], "total_number":total_number} for tool_call in tool_call_list]

        query_contents = self.client.submit_tasks(query_contents)
        return query_contents

    @staticmethod
    def _canonicalize_search_action(tool_call: Any) -> Optional[str]:
        """Return one stable text representation for a web-search action.

        One IG reward is associated with one tool action, while the action may
        contain up to three retrieval queries. Preserve their order and encode
        the whole compound action as a single query-group item.
        """
        if not isinstance(tool_call, dict) or tool_call.get("name") != "web_search":
            return None
        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            return None
        query_value = arguments.get("query", [])
        if isinstance(query_value, str):
            query_value = [query_value]
        if not isinstance(query_value, list):
            return None

        normalized_queries = []
        for query in query_value[:3]:
            if not isinstance(query, str):
                continue
            normalized = re.sub(r"\s+", " ", query).strip()
            if normalized:
                normalized_queries.append(normalized)
        return "\n<query_sep>\n".join(normalized_queries) or None

    @staticmethod
    def _is_single_web_search_query(tool_call: Any) -> bool:
        """Return whether a web-search action contains exactly one query."""
        if not isinstance(tool_call, dict) or tool_call.get("name") != "web_search":
            return False
        arguments = tool_call.get("arguments")
        if not isinstance(arguments, dict):
            return False
        query = arguments.get("query")
        if isinstance(query, str):
            return bool(query.strip())
        return (
            isinstance(query, list)
            and len(query) == 1
            and isinstance(query[0], str)
            and bool(query[0].strip())
        )
    @staticmethod
    def _escape_tool_observation(content: Any) -> str:
        """Prevent external tool text from injecting ChatML control tokens.

        The same helper is used by both the web-search and code-act observation
        branches so every tool result is serialized under one escaping policy.
        """
        return (
            str(content)
            .replace("<|im_start|>", "<|im_start_escaped|>")
            .replace("<|im_end|>", "<|im_end_escaped|>")
        )

    @staticmethod
    def _count_serialized_assistant_turns(response_text: str) -> int:
        """Mirror ``info_gain.py``'s assistant-turn parser exactly.

        The final response tensor may have been left-truncated to fit the
        model context. Counting its decoded representation lets rollout
        metadata retain only IG rewards whose assistant boundaries still
        exist in that tensor.
        """
        separator = "\n<|im_start|>assistant\n"
        positions = []
        search_start = 0
        while True:
            position = response_text.find(separator, search_start)
            if position == -1:
                break
            # 只有前一个消息以 <|im_end|> 正常结束时，才是真实 assistant role。
            prefix = response_text[:position].rstrip()
            if position == 0 or prefix.endswith("<|im_end|>"):
                positions.append(position)
            search_start = position + 1
        if not positions:
            return 1
        return len(positions) + (1 if positions[0] > 0 else 0)

    @staticmethod
    def _extract_top_retrieved_passages(tool_result: Any, limit: int = 3) -> list[dict[str, str]]:
        """Extract the first ``limit`` passages from one web-search result.

        ``Handler._handle_web_search`` returns a list of query results, each
        containing ``web_page_info_list``.  A model may submit more than one
        query in its first tool call, so preserve handler order and flatten the
        result lists.  This records the passages actually supplied to the
        agent, rather than issuing a second retrieval only for evaluation.
        """
        if isinstance(tool_result, str):
            try:
                tool_result = json.loads(tool_result)
            except json.JSONDecodeError:
                return []

        if not isinstance(tool_result, list):
            return []

        passages: list[dict[str, str]] = []
        for query_result in tool_result:
            if not isinstance(query_result, dict):
                continue
            pages = query_result.get("web_page_info_list", [])
            if not isinstance(pages, list):
                continue
            for page in pages:
                if not isinstance(page, dict):
                    continue
                passage_id = page.get("passage_id", "")
                passage_text = page.get("quick_summary", page.get("passage_text", ""))
                passages.append(
                    {
                        "passage_id": "" if passage_id is None else str(passage_id),
                        "passage_text": "" if passage_text is None else str(passage_text),
                    }
                )
                if len(passages) == limit:
                    return passages
        return passages

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus * self.config.nnodes
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        if remainder == 0:
            output = self.actor_rollout_wg.generate_sequences(active_batch)
            return output
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
        padded_active_batch = DataProto.from_dict(padded_batch)

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output



    def parse_question(self, input_ids: torch.Tensor) -> str:
        """Parse question to get the query content."""
        query_contents = self.tokenizer.batch_decode(input_ids)
        query_contents = [re.sub(r'^(<\|endoftext\|>)+', '', content) for content in query_contents]
        user_marker = "<|im_start|>user\n"
        end_marker = "<|im_end|>"
        parsed_queries = []
        for content in query_contents:
            if user_marker in content:
                # Normal, untruncated ChatML prompt.
                content = content.rsplit(user_marker, 1)[1]
            else:
                # Left truncation can remove the leading ChatML user marker.
                # The retained prefix is still user content; stop at the
                # following assistant message when it is present.
                content = content.split("<|im_start|>assistant\n", 1)[0]
            parsed_queries.append(content.split(end_marker, 1)[0])
        query_contents = parsed_queries
        return query_contents

    @staticmethod
    def _extract_last_user_question(user_content: str) -> str:
        """Return the final question from a ConvAgent dialogue wrapper."""
        marker = "Current user question:"
        if marker in user_content:
            return user_content.rsplit(marker, 1)[1].strip()
        return str(user_content).strip()
    def parse_response(self, input_ids: torch.Tensor, think: bool = False) -> List[Tuple[bool, str, str]]:
        """Parse the legal agent actions for one generated turn.

        ``<tool_call>`` stays active. ``<answer>`` and, when enabled, an empty
        ``<nonanswer>`` terminate the trajectory. Any mixed, malformed, or
        unsupported action is an invalid terminal response.
        """
        response_contents = self.tokenizer.batch_decode(input_ids)
        results = []
        for i, content in enumerate(response_contents):
            if think:
                content = "<think>" + content
            has_think = "<think>" in content and "</think>" in content
            has_answer = "<answer>" in content or "</answer>" in content
            has_nonanswer = "<nonanswer>" in content or "</nonanswer>" in content
            has_tool_call = "<tool_call>" in content or "</tool_call>" in content

            # A response may contain exactly one action form. In particular,
            # answer/nonanswer cannot be combined in one terminal turn.
            if not has_think or sum((has_answer, has_nonanswer, has_tool_call)) != 1:
                results.append((True, "", ""))
                continue

            think_content = content.split("<think>", 1)[1].split("</think>", 1)[0]
            if has_answer:
                if "<answer>" not in content or "</answer>" not in content:
                    results.append((True, "", ""))
                    continue
                answer = content.split("<answer>", 1)[1].split("</answer>", 1)[0]
                results.append((True, think_content, answer))
            elif has_nonanswer:
                if (
                    not self.config.allow_nonanswer
                    or "<nonanswer>" not in content
                    or "</nonanswer>" not in content
                ):
                    results.append((True, "", ""))
                    continue
                nonanswer_content = content.split("<nonanswer>", 1)[1].split("</nonanswer>", 1)[0]
                if nonanswer_content.strip():
                    results.append((True, "", ""))
                else:
                    results.append((True, think_content, ""))
            elif has_tool_call and self.codeact_env_disabled:
                if "<tool_call>" not in content or "</tool_call>" not in content:
                    results.append((True, "", ""))
                    continue
                tool_call_text = content.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
                try:
                    tool_call = json.loads(tool_call_text)
                    assert "name" in tool_call, "no valid function name in tool call"
                    assert "arguments" in tool_call, "no valid arguments in tool call"
                    assert tool_call["name"] not in [""], "invalid tool name"

                    results.append((False, think_content, tool_call))
                except Exception as exc:
                    if i < 10:
                        print(f"model tool call format error: {exc}")
                        print(content.replace("<|endoftext|>", ""))
                    results.append((True, "", ""))
            else:
                results.append((True, "", ""))
        return results
    def pseudo_generate_sequences(self, prompts, response):
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = idx.size(0)
        eos_token_id = self.tokenizer.eos_token_id
        non_tensor_batch = prompts.non_tensor_batch
        response = pad_2d_list_to_length(response, self.tokenizer.pad_token_id).to(idx.device)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        last_valid_pos_ids = (attention_mask.sum(dim=1, keepdim=True).long() - 1).clamp(min=0)
        response_position_ids = last_valid_pos_ids + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def _build_rollings_from_messages(
        self,
        messages_batch: List[List[Dict[str, Any]]],
        prompt_token_budget: Optional[int],
    ) -> DataProto:
        """Serialize a message subset exactly as one normal rollout turn."""
        serialized = self.tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=False,
        )
        if isinstance(serialized, str):
            serialized = [serialized]
        serialized = [text + "<think>" for text in serialized]
        tokenized = self._tokenize_with_left_truncation(
            serialized,
            max_length=prompt_token_budget,
        )
        pad_mask = tokenized["input_ids"] != self.tokenizer.pad_token_id
        sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)
        input_ids = tokenized["input_ids"].gather(1, sorted_indices)
        attention_mask = tokenized["attention_mask"].gather(1, sorted_indices)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)
        return DataProto.from_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )

    @staticmethod
    def _replace_first_search_with_original_query(
        messages: List[Dict[str, Any]],
        original_query: str,
        escaped_original_observation: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Counterfactually replace the first web-search action and its result.

        The generated first-turn thought and all other context are retained.
        Only the query argument and retrieval observation are changed, so the
        resulting GT log-prob is a paired score for the same rollout.
        """
        copied = copy.deepcopy(messages)
        for message_index, message in enumerate(copied):
            if message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call_index, wrapped_call in enumerate(tool_calls):
                if not isinstance(wrapped_call, dict):
                    continue
                function = wrapped_call.get("function")
                if not isinstance(function, dict) or function.get("name") != "web_search":
                    continue
                arguments = function.get("arguments")
                if not isinstance(arguments, dict):
                    return None
                function = copy.deepcopy(function)
                function["arguments"] = copy.deepcopy(arguments)
                function["arguments"]["query"] = [original_query]
                tool_calls = copy.deepcopy(tool_calls)
                tool_calls[tool_call_index] = copy.deepcopy(wrapped_call)
                tool_calls[tool_call_index]["function"] = function
                message["tool_calls"] = tool_calls
                for next_message in copied[message_index + 1:]:
                    if next_message.get("role") != "tool":
                        continue
                    if next_message.get("name") == "web_search":
                        next_message["content"] = escaped_original_observation
                        return copied
                return None
        return None

    def _compute_counterfactual_gt_values(
        self,
        messages_list: List[List[Dict[str, Any]]],
        original_observations: Dict[int, Tuple[str, str]],
        pseudo_resps_with_gt: List[List[int]],
        gt_idx: List[List[int]],
        prompt_token_budget: Optional[int],
    ) -> Dict[int, float]:
        """Return paired original-query GT values for active first-search rollouts."""
        sample_indices: List[int] = []
        counterfactual_messages: List[List[Dict[str, Any]]] = []
        for sample_idx in sorted(original_observations):
            original_query, escaped_observation = original_observations[sample_idx]
            replaced = self._replace_first_search_with_original_query(
                messages_list[sample_idx], original_query, escaped_observation
            )
            if replaced is None:
                raise RuntimeError(
                    "Rewrite-bound counterfactual is missing its first web-search "
                    f"turn for sample={sample_idx}"
                )
            sample_indices.append(sample_idx)
            counterfactual_messages.append(replaced)

        if not sample_indices:
            return {}
        counterfactual_rollings = self._build_rollings_from_messages(
            counterfactual_messages,
            prompt_token_budget,
        )
        pseudo_output = self.pseudo_generate_sequences(
            counterfactual_rollings,
            [pseudo_resps_with_gt[index] for index in sample_indices],
        )
        # The FSDP worker-group splitter requires equal-sized shards.  First
        # search rollout counts are data dependent (e.g. 449 on four GPUs),
        # so pad only this counterfactual scoring batch and immediately discard
        # the duplicated tail.  No real trajectory or reward is duplicated.
        pseudo_output_padded, pad_size = pad_dataproto_to_divisor(
            pseudo_output,
            self.actor_rollout_wg.world_size,
        )
        if pad_size:
            print(
                "[RewriteBound] padding counterfactual GT-logprob batch "
                f"from {len(pseudo_output)} to {len(pseudo_output_padded)} "
                f"for world_size={self.actor_rollout_wg.world_size}",
                flush=True,
            )
        log_prob_output = self.actor_rollout_wg.compute_log_prob(pseudo_output_padded)
        log_prob_output = unpad_dataproto(log_prob_output, pad_size=pad_size)
        values: Dict[int, float] = {}
        for local_index, sample_idx in enumerate(sample_indices):
            start, end = gt_idx[sample_idx]
            if start >= end:
                continue
            log_probs = log_prob_output.batch["old_log_probs"][local_index, start:end]
            mean_log_prob = log_probs.mean().item()
            if not math.isfinite(mean_log_prob):
                continue
            if self.config.info_gain_type == "log_prob_diff":
                values[sample_idx] = mean_log_prob
            else:
                values[sample_idx] = torch.exp(torch.tensor(mean_log_prob)).item()
        return values

    def run_llm_loop(self, gen_batch: DataProto, global_steps: int, ground_truths: list) -> Tuple[Dict, Dict, list, list]:
        """Run main LLM generation loop.

        The fourth return value holds the first retrieval's top-three passages
        for every generated sample.  It is consumed by validation export only;
        an empty list means that sample never issued a valid web-search call
        (for example, an immediate answer).
        """


        # Standalone validation scripts do not necessarily inherit the legacy
        # launcher environment. A single-node run is rank zero by definition.
        node_rank = int(os.environ.get("PET_NODE_RANK", "0"))
        print(f"node {node_rank} gains {len(gen_batch.batch['input_ids'])} * {self.config.n} datas!", flush=True)
        query_contents = self.parse_question(gen_batch.batch['input_ids'])
        original_last_user_queries = [
            self._extract_last_user_question(query_content)
            for query_content in query_contents
        ]
        
        messages_list = []
        agent_grpo_idx = []
        for gt in ground_truths:
            # ConvAgent stores a list of {action, response, passage_id, ...}
            # candidates.  IGPO uses the first answer candidate; samples with
            # no answer candidate deliberately have an empty target and receive
            # no outcome or information-gain reward.
            gt['ground_truth'] = select_answer_ground_truth(gt.get('ground_truth', ''))
            if "<|answer_split|>" in gt['ground_truth']:
                gt['ground_truth'] = gt['ground_truth'].split("<|answer_split|>")[0]
            _gt = gt['ground_truth'].strip()
            if _gt.startswith('['):
                # Some normal QReCC answers begin with '[' (for example a
                # bracketed citation), but are not JSON.  Only convert the
                # legacy label-list format when it is valid JSON and every
                # member actually has a ``label`` field.
                try:
                    parsed_gt = json.loads(_gt)
                except json.JSONDecodeError:
                    parsed_gt = None
                if isinstance(parsed_gt, list) and all(
                    isinstance(item, dict) and 'label' in item for item in parsed_gt
                ):
                    label = 'true'
                    for item in parsed_gt:
                        if str(item['label']).lower() == 'false':
                            label = 'false'
                            break
                    gt['ground_truth'] = label

        # Preprocess ground_truths to align with batch_size * n
        ground_truths_rolling = []
        for gt in ground_truths:
            for _ in range(self.config.n):
                ground_truths_rolling.append(gt)

        # Use offset_mapping to precisely calculate ground truth token range
        # Avoid index offset caused by subword tokenization boundary effects
        PREFIX = "\nNow there's enough information to answer\n</think>\n<answer>\n"
        SUFFIX = "\n</answer><|im_end|>"
        
        pseudo_resps_with_gt = []
        gt_idx = []
        
        for ground_truth in ground_truths_rolling:
            gt_text = ground_truth['ground_truth']
            full_text = f"{PREFIX}{gt_text}{SUFFIX}"
            
            # Use offset_mapping to get precise character-token mapping
            encoding = self.tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
            token_ids = encoding['input_ids'].tolist()[0]
            offset_mapping = encoding['offset_mapping'].tolist()[0]  # [(char_start, char_end), ...]
            
            pseudo_resps_with_gt.append(token_ids)

            # An empty selected ground truth must not create a pseudo-answer
            # supervision signal. Keep the structural prefix/suffix for
            # batching, but make its GT token range empty so all information-
            # gain computation skips this sample.
            if not gt_text.strip():
                gt_idx.append([0, 0])
                continue
            
            if len(token_ids) == 0:
                print(f"❗❗❗ EMPTY token_ids for ground_truth: '{gt_text}'")
                gt_idx.append([0, 0])
                continue
            
            # Calculate ground truth position in original string
            gt_char_start = len(PREFIX)
            gt_char_end = len(PREFIX) + len(gt_text)
            
            # Find precise token indices through offset_mapping
            gt_token_start = None
            gt_token_end = None
            
            for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                # Find the first token covering gt_char_start
                if gt_token_start is None and char_end > gt_char_start:
                    gt_token_start = token_idx
                # Find the last token covering gt content (char_start < gt_char_end)
                if char_start < gt_char_end and char_end > 0:
                    gt_token_end = token_idx + 1
            
            # Boundary check
            if gt_token_start is None:
                gt_token_start = len(token_ids)
            if gt_token_end is None:
                gt_token_end = len(token_ids)
            
            gt_idx.append([gt_token_start, gt_token_end])

        # This budget is applied every turn before both rollout generation and
        # pseudo GT log-prob scoring.  ``pseudo_generate_sequences`` pads GT
        # answers to the longest answer in this batch, so reserve that exact
        # maximum rather than only the typical answer length.
        max_pseudo_response_tokens = max((len(token_ids) for token_ids in pseudo_resps_with_gt), default=0)
        prompt_token_budget = self._prompt_token_budget(max_pseudo_response_tokens)
        if prompt_token_budget is not None:
            print(
                "[IGPO] Multi-turn context cap: "
                f"prompt<={prompt_token_budget}, pseudo_response<={max_pseudo_response_tokens}, "
                f"model_context={self._model_context_window()}"
            )
        

        for idx, query_content in enumerate(query_contents):
            for _ in range(self.config.n):
                if self.system_prompt:
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": query_content}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": query_content}
                    ]
                messages_list.append(messages)
                agent_grpo_idx.append(idx)
        activate_list = [i for i in range(len(messages_list))]
        message_string_list = ["" for _ in range(len(messages_list))]

        # Ensure output directory exists (only create when project_name and experiment_name are valid)
        output_dir = None
        if self.config.project_name and self.config.experiment_name:
            output_dir = f"./outputs/{self.config.project_name}/{self.config.experiment_name}/rollout"
            if not os.path.exists(output_dir):
                print(f"Directory not exist, create at {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
        
        # Create information gain reward: list[list]
        gt_log_probs_per_turn = [[] for _ in range(len(messages_list))]
        gt_entropys_per_turn = [[] for _ in range(len(messages_list))]
        info_gain_rewards = [[] for _ in range(len(messages_list))]
        # Same positional layout as info_gain_rewards. ``None`` means this IG
        # boundary has no first-turn rewrite counterpart.
        rewrite_bound_rewards = [[] for _ in range(len(messages_list))]
        # Parallel to ``info_gain_rewards``. A false entry is an intentionally
        # zero-valued structural IG slot (empty GT, missing baseline, or
        # non-finite score). It must keep its turn boundary, but must not
        # participate in query-semantic normalization.
        info_gain_query_eligible = [[] for _ in range(len(messages_list))]
        first_retrieved_passages = [[] for _ in range(len(messages_list))]
        # One entry per completed tool action. ``None`` deliberately marks an
        # invalid/non-search action so positional alignment is never silently
        # shifted when its later IG reward is produced.
        tool_action_queries = [[] for _ in range(len(messages_list))]
        # Filled after first-turn tool execution. Each entry carries the one
        # original last-user-query retrieval observation shared by its uid.
        first_turn_original_observations: Dict[int, Tuple[str, str]] = {}
        gt_values = {}  # Store previous turn's value (probability or log probability, depending on info_gain_type)

        def append_ig_slot(sample_idx: int, value: float, eligible: bool, rewrite_value: Optional[float] = None):
            """Keep IG, query, and Rewrite-Bound metadata index-aligned."""
            info_gain_rewards[sample_idx].append(value)
            info_gain_query_eligible[sample_idx].append(eligible)
            rewrite_bound_rewards[sample_idx].append(rewrite_value)

        # Vectorized switch detection
        use_vectorized_gt_logprob = bool(self.config.use_vectorized_gt_logprob)
        print(
            "[IGPO] Vectorized GT LogProb: "
            f"{'ENABLED' if use_vectorized_gt_logprob else 'DISABLED'} (GenerationConfig)"
        )
        if self.config.rewrite_bound_advantage and use_vectorized_gt_logprob:
            raise ValueError(
                "Rewrite-Bound IGPO currently requires "
                "algorithm.use_vectorized_gt_logprob=false so its paired "
                "counterfactual value is computed in the same immediate path."
            )
        # ========== Vectorized computation: data collection structure ==========
        # When vectorization is enabled, delay GT log probs computation for batch processing after loop
        vectorized_data_collector = None
        # For verification: also compute original mode results when VERIFY_VECTORIZED is enabled
        original_mode_gt_values = {} if (use_vectorized_gt_logprob and VERIFY_VECTORIZED) else None
        original_mode_info_gains = [[] for _ in range(len(messages_list))] if (use_vectorized_gt_logprob and VERIFY_VECTORIZED) else None
        
        if use_vectorized_gt_logprob:
            vectorized_data_collector = {
                'pseudo_outputs_per_turn': [],  # List of pseudo_gen_output for each turn
                'activate_lists_per_turn': [],  # activate_list for each turn
                'gt_idx': gt_idx,  # GT token range
                'num_samples': len(messages_list),
            }
            print(f"[IGPO] Vectorized GT LogProb: Collecting data for batch computation...")
            if VERIFY_VECTORIZED:
                print(f"[IGPO] Verification mode: will also compute original mode for comparison")

        for step in range(self.config.max_turns):
            print(f"node {node_rank} step {step} start!")
            activate_messages_list = [messages_list[i] for i in activate_list]

            if activate_list == []:
                break
            try:
                rollings_active = self.tokenizer.apply_chat_template(activate_messages_list, add_generation_prompt=True, tokenize=False)
            except Exception as e:
                print(f"Error in tokenizer.apply_chat_template: {e}")
                # Fallback strategy: process each message individually
                rollings_active = []
                for msg in activate_messages_list:
                    try:
                        result = self.tokenizer.apply_chat_template([msg], add_generation_prompt=True, tokenize=False)
                        rollings_active.extend(result)
                    except Exception as inner_e:
                        print(f"Failed to process message: {inner_e}")
                        raise  # Cannot recover, raise exception
    
            think = True
            
            if think:
                rollings_active = [rolling + "<think>" for rolling in rollings_active]
            else:
                rollings_active = [rolling for rolling in rollings_active]
            
            # Tool responses can be arbitrarily long.  Cap the serialized
            # conversation *before* it reaches vLLM or compute_log_prob;
            # left truncation preserves the most recent retrieval evidence.
            rollings_active = self._tokenize_with_left_truncation(
                rollings_active,
                max_length=prompt_token_budget,
            )
                
            pad_mask = rollings_active['input_ids'] != self.tokenizer.pad_token_id
            sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)
            rollings_active['input_ids'] = rollings_active['input_ids'].gather(1, sorted_indices)
            rollings_active['attention_mask'] = rollings_active['attention_mask'].gather(1, sorted_indices)
            
            attention_mask = rollings_active['attention_mask']
            rollings_active['position_ids'] = self.tensor_fn.create_position_ids(attention_mask)

            print(f"node {node_rank}, turn {step} rollings_active is {len(rollings_active['input_ids'])} datas")
            rollings_active = DataProto.from_dict({
                'input_ids': rollings_active['input_ids'],
                'attention_mask': rollings_active['attention_mask'],
                'position_ids': rollings_active['position_ids'],
            })

            if self.is_validation:
                rollings_active.meta_info["do_sample"] = False
            
            if step == 0:
                info_gain_rollings_active = copy.deepcopy(rollings_active)
            else:
                if info_gain_rollings_active.batch['input_ids'].shape[1] < rollings_active.batch['input_ids'].shape[1]:
                    info_gain_rollings_active.batch['input_ids'] = F.pad(
                        info_gain_rollings_active.batch['input_ids'], 
                        pad=(0, rollings_active.batch['input_ids'].shape[1] - info_gain_rollings_active.batch['input_ids'].shape[1]),
                        mode='constant',
                        value=self.tokenizer.pad_token_id,
                       )
					
                    info_gain_rollings_active.batch['attention_mask'] = F.pad(
                        info_gain_rollings_active.batch['attention_mask'], 
                        pad=(0, rollings_active.batch['attention_mask'].shape[1] - info_gain_rollings_active.batch['attention_mask'].shape[1]),
                        mode='constant',
                        value=0,  # attention_mask uses 0 for padding positions
                        )

                    info_gain_rollings_active.batch['position_ids'] = F.pad(
                        info_gain_rollings_active.batch['position_ids'], 
                        pad=(0, rollings_active.batch['position_ids'].shape[1] - info_gain_rollings_active.batch['position_ids'].shape[1]),
                        mode='constant',
                        value=0,  # position_ids uses 0 for padding
                        )
                
                    for i in range(len(activate_list)):
                        info_gain_rollings_active.batch['input_ids'][activate_list[i], :] = rollings_active.batch['input_ids'][i]
                        info_gain_rollings_active.batch['attention_mask'][activate_list[i], :] = rollings_active.batch['attention_mask'][i]
                        info_gain_rollings_active.batch['position_ids'][activate_list[i], :] = rollings_active.batch['position_ids'][i]
                else:
                    src_len = rollings_active.batch['input_ids'].shape[1]
                    ig_len = info_gain_rollings_active.batch['input_ids'].shape[1]
                    for i in range(len(activate_list)):
                        idx = activate_list[i]
                        info_gain_rollings_active.batch['input_ids'][idx, :src_len] = rollings_active.batch['input_ids'][i]
                        info_gain_rollings_active.batch['attention_mask'][idx, :src_len] = rollings_active.batch['attention_mask'][i]
                        info_gain_rollings_active.batch['position_ids'][idx, :src_len] = rollings_active.batch['position_ids'][i]
                        if src_len < ig_len:
                            info_gain_rollings_active.batch['input_ids'][idx, src_len:] = self.tokenizer.pad_token_id
                            info_gain_rollings_active.batch['attention_mask'][idx, src_len:] = 0
                            info_gain_rollings_active.batch['position_ids'][idx, src_len:] = 0
            
            pseudo_gen_output = self.pseudo_generate_sequences(info_gain_rollings_active, pseudo_resps_with_gt)
            
            # ========== GT LogProb computation (vectorized or immediate) ==========
            if use_vectorized_gt_logprob and vectorized_data_collector is not None:
                # Vectorized mode: collect data, delay computation
                # Save current turn's pseudo_gen_output (need clone to avoid modification by subsequent operations)
                pseudo_output_clone = DataProto.from_dict({
                    'prompts': pseudo_gen_output.batch['prompts'].clone(),
                    'responses': pseudo_gen_output.batch['responses'].clone(),
                    'input_ids': pseudo_gen_output.batch['input_ids'].clone(),
                    'attention_mask': pseudo_gen_output.batch['attention_mask'].clone(),
                    'position_ids': pseudo_gen_output.batch['position_ids'].clone(),
                })
                vectorized_data_collector['pseudo_outputs_per_turn'].append(pseudo_output_clone)
                vectorized_data_collector['activate_lists_per_turn'].append(list(activate_list))
                
                # ========== Verification: also compute original mode for comparison ==========
                if VERIFY_VECTORIZED and original_mode_gt_values is not None:
                    # Compute log_probs using original mode (immediate computation)
                    verify_log_probs = self.actor_rollout_wg.compute_log_prob(pseudo_gen_output)
                    info_gain_type = self.config.info_gain_type
                    
                    if step == 0:
                        for i in activate_list:
                            if gt_idx[i][0] >= gt_idx[i][1]:
                                continue
                            log_probs = verify_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]]
                            mean_log_prob = log_probs.mean().item()
                            if math.isnan(mean_log_prob) or math.isinf(mean_log_prob):
                                continue
                            if info_gain_type == "log_prob_diff":
                                original_mode_gt_values[i] = mean_log_prob
                            else:
                                original_mode_gt_values[i] = torch.exp(torch.tensor(mean_log_prob)).item()
                    else:
                        for i in activate_list:
                            if gt_idx[i][0] >= gt_idx[i][1]:
                                original_mode_info_gains[i].append(0.0)
                                continue
                            if i not in original_mode_gt_values:
                                original_mode_info_gains[i].append(0.0)
                                continue
                            log_probs = verify_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]]
                            mean_log_prob = log_probs.mean().item()
                            if not math.isfinite(mean_log_prob):
                                original_mode_info_gains[i].append(0.0)
                                continue
                            if info_gain_type == "log_prob_diff":
                                cur_value = mean_log_prob
                                info_gain = cur_value - original_mode_gt_values[i]
                            else:
                                cur_value = torch.exp(torch.tensor(mean_log_prob)).item()
                                info_gain = cur_value - original_mode_gt_values[i]
                            if math.isnan(info_gain) or math.isinf(info_gain):
                                original_mode_info_gains[i].append(0.0)
                                continue
                            original_mode_info_gains[i].append(info_gain)
                            original_mode_gt_values[i] = cur_value
            else:
                # Original mode: immediate computation
                pseudo_gen_output_log_probs = self.actor_rollout_wg.compute_log_prob(pseudo_gen_output)
                
                # ========== Compute info_gain_reward based on info_gain_type ==========
                # "prob_diff": Use probability difference exp(mean(log P_t)) - exp(mean(log P_{t-1}))
                # "log_prob_diff": Use log probability difference mean(log P_t) - mean(log P_{t-1})
                
                info_gain_type = self.config.info_gain_type  # "prob_diff" or "log_prob_diff"
                counterfactual_gt_values = {}
                if step == 1 and self.config.rewrite_bound_advantage:
                    counterfactual_gt_values = self._compute_counterfactual_gt_values(
                        messages_list=messages_list,
                        original_observations=first_turn_original_observations,
                        pseudo_resps_with_gt=pseudo_resps_with_gt,
                        gt_idx=gt_idx,
                        prompt_token_budget=prompt_token_budget,
                    )
                
                if step == 0:
                    for i in activate_list:
                        # Check if gt_idx range is valid
                        if i >= len(gt_idx) or gt_idx[i][0] >= gt_idx[i][1]:
                            continue
                        log_probs = pseudo_gen_output_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]]
                        mean_log_prob = log_probs.mean().item()
                        
                        # Skip if mean_log_prob is nan or inf
                        if math.isnan(mean_log_prob) or math.isinf(mean_log_prob):
                            continue
                        
                        if info_gain_type == "log_prob_diff":
                            gt_values[i] = mean_log_prob
                        else:  # "prob_diff" (default)
                            gt_values[i] = torch.exp(torch.tensor(mean_log_prob)).item()
                        
                        gt_log_probs_per_turn[i].append(log_probs.tolist())
                        gt_entropys_per_turn[i].append(pseudo_gen_output_log_probs.batch['entropys'][i, gt_idx[i][0]:gt_idx[i][1]].tolist())
                else:
                    for i in activate_list:
                        # Check if gt_idx range is valid
                        if gt_idx[i][0] >= gt_idx[i][1]:
                            # Keep one IG slot per completed tool action even when
                            # this sample has no usable answer supervision.
                            append_ig_slot(i, 0.0, False)
                            continue
                        # A missing baseline is a zero IG signal, not a missing
                        # turn: preserving the slot keeps reward/query/token alignment.
                        if i not in gt_values:
                            append_ig_slot(i, 0.0, False)
                            continue
                        log_probs = pseudo_gen_output_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]]
                        mean_log_prob = log_probs.mean().item()
                        
                        # Check for nan in mean_log_prob
                        if not math.isfinite(mean_log_prob):
                            append_ig_slot(i, 0.0, False)
                            continue
                        
                        prev_value = gt_values[i]  # Save previous value for verification
                        
                        if info_gain_type == "log_prob_diff":
                            # Use log probability difference
                            cur_value = mean_log_prob
                            info_gain = cur_value - gt_values[i]
                        else:  # "prob_diff" (default)
                            # Use probability difference
                            cur_value = torch.exp(torch.tensor(mean_log_prob)).item()
                            info_gain = cur_value - gt_values[i]
                        
                        # Check for nan/inf in info_gain
                        if math.isnan(info_gain) or math.isinf(info_gain):
                            append_ig_slot(i, 0.0, False)
                            continue
                        
                        rewrite_value = None
                        if step == 1 and i in counterfactual_gt_values:
                            # The pre-retrieval baseline cancels: this is
                            # IG(rewrite) - IG(original) for the same rollout.
                            rewrite_value = cur_value - counterfactual_gt_values[i]
                            if not math.isfinite(rewrite_value):
                                rewrite_value = None
                        append_ig_slot(i, info_gain, True, rewrite_value)
                        gt_values[i] = cur_value
                        
                        gt_log_probs_per_turn[i].append(log_probs.tolist())
                        gt_entropys_per_turn[i].append(pseudo_gen_output_log_probs.batch['entropys'][i, gt_idx[i][0]:gt_idx[i][1]].tolist())       

            gen_output = self._generate_with_gpu_padding(rollings_active)
            
            meta_info = gen_output.meta_info
            print(f"node {node_rank}, turn {step} gen_output {len(gen_output.batch['responses'])} datas")

            results = self.parse_response(gen_output.batch['responses'], think=think)
            assert len(results) == len(activate_list)  # After each round update, result count equals active query count
            activate_list_copy = []
            tool_call_list = []
            for i in range(len(results)):
                if results[i][0]:
                    message_string_list[activate_list[i]] = self.tokenizer.decode(rollings_active.batch['input_ids'][i], skip_special_tokens=False).replace("<|endoftext|>", "") + self.tokenizer.decode(gen_output.batch['responses'][i], skip_special_tokens=False).replace("<|endoftext|>", "")
                else:
                    # Rewrite-Bound gives a first tool-call reward only to one
                    # standalone rewrite query. A multi-query / wrong-tool first
                    # action is terminal-invalid rather than being silently
                    # compared against the original-query bound.
                    if (
                        step == 0
                        and self.config.rewrite_bound_advantage
                        and not self._is_single_web_search_query(results[i][2])
                    ):
                        message_string_list[activate_list[i]] = (
                            self.tokenizer.decode(rollings_active.batch['input_ids'][i], skip_special_tokens=False)
                            .replace("<|endoftext|>", "")
                            + self.tokenizer.decode(gen_output.batch['responses'][i], skip_special_tokens=False)
                            .replace("<|endoftext|>", "")
                        )
                    else:
                        activate_list_copy.append(activate_list[i])
                        tool_call_list.append((activate_list[i], messages_list[activate_list[i]][1]["content"], results[i][1], results[i][2]))
                    
            tool_call_list = self.execute_predictions(tool_call_list,len(messages_list))
            print(f"node {node_rank}, turn {step} tool_call_list {len(tool_call_list)} datas")
            if step == 0 and self.config.rewrite_bound_advantage:
                if self.client is None or not hasattr(self.client, "submit_tasks"):
                    raise RuntimeError("Rewrite-Bound IGPO requires an active MessageClient")
                original_tasks_by_uid: Dict[int, Dict[str, Any]] = {}
                first_search_samples: List[int] = []
                for tool_result in tool_call_list:
                    sample_idx = tool_result.get("idx") if isinstance(tool_result, dict) else None
                    tool_call = tool_result.get("tool_call", {}) if isinstance(tool_result, dict) else {}
                    if (
                        not isinstance(sample_idx, int)
                        or sample_idx < 0
                        or sample_idx >= len(agent_grpo_idx)
                        or not isinstance(tool_call, dict)
                        or tool_call.get("name") != "web_search"
                    ):
                        continue
                    uid_index = agent_grpo_idx[sample_idx]
                    original_query = original_last_user_queries[uid_index]
                    if not original_query:
                        continue
                    first_search_samples.append(sample_idx)
                    original_tasks_by_uid.setdefault(
                        uid_index,
                        {
                            "idx": uid_index,
                            "question": original_query,
                            "think": "",
                            "tool_call": {
                                "name": "web_search",
                                "arguments": {
                                    "query": [original_query],
                                    "topk": int(self.config.rewrite_bound_topk),
                                },
                            },
                            "total_number": len(query_contents),
                        },
                    )
                if original_tasks_by_uid:
                    original_results = self.client.submit_tasks(list(original_tasks_by_uid.values()))
                    observations_by_uid = {
                        item.get("idx"): self._escape_tool_observation(item.get("content", ""))
                        for item in original_results
                        if isinstance(item, dict) and isinstance(item.get("idx"), int)
                    }
                    for sample_idx in first_search_samples:
                        uid_index = agent_grpo_idx[sample_idx]
                        if uid_index not in observations_by_uid:
                            raise RuntimeError(
                                "Rewrite-Bound original-query retrieval returned no result "
                                f"for uid={uid_index}"
                            )
                        first_turn_original_observations[sample_idx] = (
                            original_last_user_queries[uid_index],
                            observations_by_uid[uid_index],
                        )
                    print(
                        "[RewriteBound] original-query retrievals="
                        f"{len(original_tasks_by_uid)}, first-search-rollouts={len(first_search_samples)}",
                        flush=True,
                    )
            for tool_result in tool_call_list:
                sample_idx = tool_result.get("idx") if isinstance(tool_result, dict) else None
                tool_call = tool_result.get("tool_call", {}) if isinstance(tool_result, dict) else {}
                if not isinstance(sample_idx, int) or sample_idx < 0 or sample_idx >= len(tool_action_queries):
                    continue

                # The same action chronology is later used by info_gain.py:
                # IG item k belongs to tool action k, not to the next turn.
                tool_action_queries[sample_idx].append(self._canonicalize_search_action(tool_call))

                if (
                    first_retrieved_passages[sample_idx]
                    or not isinstance(tool_call, dict)
                    or tool_call.get("name") != "web_search"
                ):
                    continue
                first_retrieved_passages[sample_idx] = self._extract_top_retrieved_passages(
                    tool_result.get("content")
                )
            for i in range(len(tool_call_list)):
                escaped_observation = self._escape_tool_observation(
                    tool_call_list[i].get("content", "")
                )
                if not self.codeact_env_disabled:  # code act enabled
                    messages_list[tool_call_list[i]['idx']].append(
                        {
                            "role": "assistant", 
                            "content": "<think>" + tool_call_list[i]['think'] + "</think>"+"\n<code>" + str(tool_call_list[i]['tool_call']['arguments']['code']) + "</code>", 
                        }
                    )
                    try:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "user", 
                                "content": "<code_response>" + escaped_observation + "</code_response>",
                            }
                        )
                    except:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "user", 
                                "content": "<code_response>" + 'Return format error, code execution failed' + "</code_response>",
                            }
                        )
                else:
                    messages_list[tool_call_list[i]['idx']].append(
                        {
                            "role": "assistant", 
                            "content": "<think>" + tool_call_list[i]['think'] + "</think>", 
                            "tool_calls": [
                                            {
                                                "type": "function", 
                                                "function": tool_call_list[i]['tool_call']
                                            }
                                        ]
                        }
                    )
                    try:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "tool", 
                                "name": tool_call_list[i]['tool_call']['name'],
                                "content": escaped_observation,
                            }
                        )
                    except:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "tool", 
                                "name": '',
                                "content": 'Return format error, tool call failed'
                            }
                        )
            print(f"Turn {step} ended, node {node_rank} originally had {len(activate_list)} queries, now has {len(activate_list_copy)} queries")
            activate_list = activate_list_copy
           
        
        # ========== Vectorized GT LogProb batch computation (Prealigned Version) ==========
        # This section uses the prealigned vectorization strategy for better mathematical rigor.
        # It is completely independent and does not affect original mode computation.
        if use_vectorized_gt_logprob and vectorized_data_collector is not None:
            num_turns_collected = len(vectorized_data_collector['pseudo_outputs_per_turn'])
            _USE_LEGACY_VECTORIZED = False  # Legacy implementation is disabled
            if num_turns_collected > 0:
                # Use prealigned vectorized module (lazy import)
                from scrl.llm_agent.prealigned_vectorized import compute_vectorized_gt_logprob
                
                info_gain_type = self.config.info_gain_type
                gt_idx = vectorized_data_collector['gt_idx']
                
                # Call prealigned vectorized computation
                vectorized_result = compute_vectorized_gt_logprob(
                    pseudo_outputs_per_turn=vectorized_data_collector['pseudo_outputs_per_turn'],
                    activate_lists_per_turn=vectorized_data_collector['activate_lists_per_turn'],
                    gt_idx=gt_idx,
                    actor_rollout_wg=self.actor_rollout_wg,
                    tokenizer=self.tokenizer,
                    info_gain_type=info_gain_type,
                    enable_strict_validation=VERIFY_VECTORIZED,
                )
                
                # Update results from vectorized computation
                gt_values = vectorized_result['gt_values']
                info_gain_rewards = vectorized_result['info_gain_rewards']
                info_gain_query_eligible = vectorized_result['info_gain_query_eligible']
                gt_log_probs_per_turn = vectorized_result['gt_log_probs_per_turn']
                gt_entropys_per_turn = vectorized_result['gt_entropys_per_turn']
                
                # Store vectorized mean_log_probs for verification (if enabled)
                if VERIFY_VECTORIZED:
                    vectorized_mode_log_probs_per_turn = vectorized_result.get('vectorized_mean_log_probs', None)
                
            if num_turns_collected > 0 and _USE_LEGACY_VECTORIZED:
                # ===== LEGACY: Left-padding implementation (disabled) =====
                print(f"[IGPO] Vectorized GT LogProb: Processing {num_turns_collected} turns in batch...")
                
                # Batch compute GT log probs for all turns
                info_gain_type = self.config.info_gain_type
                all_log_probs_results = []
                
                # Strategy: Reconstruct input_ids so ALL samples have response of length max_response_len
                # 
                # Problem: Different turns have different response lengths. compute_log_prob uses:
                #   response_length = responses.size(1)  # = max_response_len after padding
                #   logits = output[:, -response_length-1:-1]  # slices last response_length logits
                # 
                # If we only left-pad input_ids without restructuring:
                #   Turn 0 (resp_len=200): [PAD...][prompt][resp_200] → slice [-2001:-1] gets prompt! ✗
                #   Turn 3 (resp_len=800): [PAD...][prompt][resp_800] → slice [-2001:-1] gets prompt! ✗
                # 
                # Solution: Reconstruct input_ids to have uniform response length:
                #   1. Extract prompt = input_ids[:, :-original_resp_len]
                #   2. Extract response = input_ids[:, -original_resp_len:]
                #   3. Pad response to max_response_len (right padding)
                #   4. Rebuild input_ids = [prompt][response_padded]
                #   5. Then left-pad the whole input_ids to max_seq_len
                # 
                # After restructuring:
                #   Turn 0: [PAD...][prompt][resp_200][PAD_to_2000] → slice [-2001:-1] gets exactly response ✓
                #   Turn 3: [PAD...][prompt][resp_800][PAD_to_2000] → slice [-2001:-1] gets exactly response ✓
                
                gt_idx = vectorized_data_collector['gt_idx']
                
                # Step 1: Collect all turns' data
                # Note: pseudo_generate_sequences pads responses to the max GT length in the batch.
                # Since pseudo_resps_with_gt is fixed across all turns, response width is uniform.
                all_input_ids = []
                all_attention_mask = []
                all_position_ids = []
                all_responses = []
                turn_batch_sizes = []
                turn_response_lengths = []
                
                for turn_idx, pseudo_output in enumerate(vectorized_data_collector['pseudo_outputs_per_turn']):
                    batch_size = pseudo_output.batch['input_ids'].shape[0]
                    turn_batch_sizes.append(batch_size)
                    all_input_ids.append(pseudo_output.batch['input_ids'])
                    all_attention_mask.append(pseudo_output.batch['attention_mask'])
                    all_position_ids.append(pseudo_output.batch['position_ids'])
                    all_responses.append(pseudo_output.batch['responses'])
                    turn_response_lengths.append(pseudo_output.batch['responses'].shape[1])
                
                max_response_len = max(turn_response_lengths)
                max_seq_len = max(t.shape[1] for t in all_input_ids)
                
                # Step 2: Apply LEFT padding to input_ids, attention_mask, and position_ids
                # CRITICAL: Use original position_ids and apply left padding correctly
                # position_ids for PAD tokens should be 0 (will be ignored due to attention_mask=0)
                padded_input_ids = []
                padded_attention_mask = []
                padded_position_ids = []
                padded_responses = []
                
                for i in range(len(all_input_ids)):
                    curr_seq_len = all_input_ids[i].shape[1]
                    curr_resp_len = turn_response_lengths[i]
                    left_pad_len = max_seq_len - curr_seq_len
                    
                    if left_pad_len > 0:
                        # Left padding for input_ids
                        padded_input_ids.append(F.pad(all_input_ids[i], (left_pad_len, 0), value=self.tokenizer.pad_token_id))
                        # Left padding for attention_mask (PAD = 0)
                        padded_attention_mask.append(F.pad(all_attention_mask[i], (left_pad_len, 0), value=0))
                        # Left padding for position_ids (PAD = 0, will be ignored)
                        padded_position_ids.append(F.pad(all_position_ids[i], (left_pad_len, 0), value=0))
                    else:
                        padded_input_ids.append(all_input_ids[i])
                        padded_attention_mask.append(all_attention_mask[i])
                        padded_position_ids.append(all_position_ids[i])
                    
                    # Responses: right padding to max_response_len (if needed)
                    resp_pad_len = max_response_len - curr_resp_len
                    if resp_pad_len > 0:
                        padded_responses.append(F.pad(all_responses[i], (0, resp_pad_len), value=self.tokenizer.pad_token_id))
                    else:
                        padded_responses.append(all_responses[i])
                
                # Step 3: Merge into one large batch
                merged_input_ids = torch.cat(padded_input_ids, dim=0)
                merged_attention_mask = torch.cat(padded_attention_mask, dim=0)
                merged_position_ids = torch.cat(padded_position_ids, dim=0)
                merged_responses = torch.cat(padded_responses, dim=0)
                
                # Compute prompts (everything except the last max_response_len tokens)
                prompt_len = max_seq_len - max_response_len
                merged_prompts = merged_input_ids[:, :prompt_len]
                
                merged_batch = DataProto.from_dict({
                    'prompts': merged_prompts,
                    'responses': merged_responses,
                    'input_ids': merged_input_ids,
                    'attention_mask': merged_attention_mask,
                    'position_ids': merged_position_ids,
                })
                
                total_samples = merged_input_ids.shape[0]
                print(f"[IGPO] Vectorized: Merged {num_turns_collected} turns, batch={total_samples}, seq_len={max_seq_len}")
                
                # Call compute_log_prob ONCE for all turns
                merged_log_probs = self.actor_rollout_wg.compute_log_prob(merged_batch)
                merged_old_log_probs = merged_log_probs.batch['old_log_probs']
                merged_entropys = merged_log_probs.batch['entropys']
                
                print(f"[IGPO] Vectorized: compute_log_prob completed, merged_old_log_probs.shape = {merged_old_log_probs.shape}")
                
                # Extract each turn's results and compute info_gain_rewards
                # turn_boundaries marks the start index of each turn in the merged batch
                turn_boundaries = [0]
                for bs in turn_batch_sizes:
                    turn_boundaries.append(turn_boundaries[-1] + bs)
                
                for turn_idx in range(num_turns_collected):
                    start_idx = turn_boundaries[turn_idx]
                    end_idx = turn_boundaries[turn_idx + 1]
                    activate_list_for_turn = vectorized_data_collector['activate_lists_per_turn'][turn_idx]
                    
                    # Extract current turn's log_probs from merged results
                    turn_old_log_probs = merged_old_log_probs[start_idx:end_idx]
                    turn_entropys = merged_entropys[start_idx:end_idx]
                    
                    if turn_idx == 0:
                        # First turn: initialize gt_values
                        for local_idx, global_idx in enumerate(activate_list_for_turn):
                            if gt_idx[global_idx][0] >= gt_idx[global_idx][1]:
                                continue
                            log_probs = turn_old_log_probs[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]]
                            mean_log_prob = log_probs.mean().item()
                            
                            if math.isnan(mean_log_prob) or math.isinf(mean_log_prob):
                                continue
                            
                            if info_gain_type == "log_prob_diff":
                                gt_values[global_idx] = mean_log_prob
                            else:
                                gt_values[global_idx] = torch.exp(torch.tensor(mean_log_prob)).item()
                            
                            gt_log_probs_per_turn[global_idx].append(log_probs.tolist())
                            gt_entropys_per_turn[global_idx].append(turn_entropys[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]].tolist())
                    else:
                        # Subsequent turns: compute info_gain
                        for local_idx, global_idx in enumerate(activate_list_for_turn):
                            if gt_idx[global_idx][0] >= gt_idx[global_idx][1]:
                                continue
                            if global_idx not in gt_values:
                                continue
                            log_probs = turn_old_log_probs[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]]
                            mean_log_prob = log_probs.mean().item()
                            
                            if math.isnan(mean_log_prob):
                                continue
                            
                            if info_gain_type == "log_prob_diff":
                                cur_value = mean_log_prob
                                info_gain = cur_value - gt_values[global_idx]
                            else:
                                cur_value = torch.exp(torch.tensor(mean_log_prob)).item()
                                info_gain = cur_value - gt_values[global_idx]
                            
                            if math.isnan(info_gain) or math.isinf(info_gain):
                                continue
                            
                            info_gain_rewards[global_idx].append(info_gain)
                            gt_values[global_idx] = cur_value
                            
                            gt_log_probs_per_turn[global_idx].append(log_probs.tolist())
                            gt_entropys_per_turn[global_idx].append(turn_entropys[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]].tolist())
                
                # Statistics and print results
                total_info_gains = sum(len(r) for r in info_gain_rewards)
                print(f"[IGPO] Vectorized GT LogProb COMPLETED: "
                      f"{num_turns_collected} turns merged, "
                      f"{total_samples} total samples, "
                      f"1 compute_log_prob call, "
                      f"{total_info_gains} info_gain values computed")
                
                # ========== Verification: compare vectorized vs original mode ==========
                if VERIFY_VECTORIZED and original_mode_info_gains is not None:
                    print(f"\n[VERIFY] ========== Comparing Vectorized vs Original Mode ==========")
                    mismatch_count = 0
                    match_count = 0
                    total_compared = 0
                    max_abs_diff = 0.0
                    
                    for sample_idx in range(len(messages_list)):
                        vec_gains = info_gain_rewards[sample_idx]
                        orig_gains = original_mode_info_gains[sample_idx]
                        
                        if len(vec_gains) != len(orig_gains):
                            print(f"[VERIFY] Sample {sample_idx}: LENGTH MISMATCH! vectorized={len(vec_gains)}, original={len(orig_gains)}")
                            mismatch_count += 1
                            continue
                        
                        for turn_idx, (v, o) in enumerate(zip(vec_gains, orig_gains)):
                            total_compared += 1
                            abs_diff = abs(v - o)
                            max_abs_diff = max(max_abs_diff, abs_diff)
                            
                            # Check if values are close (tolerance for floating point)
                            if abs_diff > 1e-5:
                                # Only print first 10 mismatches to avoid log spam
                                if mismatch_count < 10:
                                    print(f"[VERIFY] Sample {sample_idx}, Turn {turn_idx}: MISMATCH! "
                                          f"vectorized={v:.8f}, original={o:.8f}, diff={abs_diff:.8e}")
                                mismatch_count += 1
                            else:
                                match_count += 1
                    
                    print(f"[VERIFY] ========== Summary ==========")
                    print(f"[VERIFY] Total values compared: {total_compared}")
                    print(f"[VERIFY] Matches (diff < 1e-5): {match_count}")
                    print(f"[VERIFY] Mismatches: {mismatch_count}")
                    print(f"[VERIFY] Max absolute difference: {max_abs_diff:.8e}")
                    if mismatch_count == 0 and total_compared > 0:
                        print(f"[VERIFY] ✓ PASSED: Vectorized mode is numerically equivalent to original mode!")
                    elif total_compared == 0:
                        print(f"[VERIFY] ⚠ WARNING: No values to compare (no info_gains computed)")
                    else:
                        print(f"[VERIFY] ✗ FAILED: Found {mismatch_count} mismatches!")
                    print(f"[VERIFY] ====================================\n")
            else:
                print(f"[IGPO] Vectorized GT LogProb: No turns collected (all samples may have finished early)")
        
        # Save gt_log_probs to local output directory (uncomment for debugging if needed)
        # gt_log_probs_path = os.path.join(output_dir, f"gt_log_probs_{global_steps}.json")
        # with open(gt_log_probs_path, 'w') as f:
        #     json.dump({"gt_log_probs_per_turn": gt_log_probs_per_turn, "gt_entropys_per_turn": gt_entropys_per_turn}, f)

        if activate_list != []:
            for i in activate_list:
                # Use add_generation_prompt=False to avoid adding an extra separator at the end,
                # which would cause turn mismatch in info_gain.py
                message_string_list[i] = self.tokenizer.apply_chat_template(messages_list[i], add_generation_prompt=False, tokenize=False)
        
        response_str_list = []
        initial_prompt_list = []
        for i, messages in enumerate(messages_list):
            initial_prompt = self.tokenizer.apply_chat_template(messages[0:2], add_generation_prompt=True, tokenize=False)
            initial_prompt_list.append(initial_prompt)
            response_str_list.append(message_string_list[i][len(initial_prompt):])
        
        final_prompt_budget = self._prompt_token_budget()
        prompts_tokenizered = self._tokenize_with_left_truncation(
            initial_prompt_list,
            max_length=final_prompt_budget,
        )

        prompts_repeated = prompts_tokenizered['input_ids']
        pad_mask = prompts_repeated != self.tokenizer.pad_token_id
        sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)
        

        prompts_repeated = prompts_repeated.gather(1, sorted_indices)
        prompts_attention_mask = prompts_tokenizered['attention_mask'].gather(1, sorted_indices)

        # The final trajectory batch is later used for actor updates.  Its
        # response portion contains all prior assistant/tool turns and must be
        # bounded as well, otherwise it can reintroduce the same overlong
        # sequence after a successful rollout.
        max_model_len = self._model_context_window()
        response_token_budget = None
        if max_model_len is not None:
            response_token_budget = max_model_len - prompts_repeated.shape[1]
            if response_token_budget <= 0:
                raise ValueError(
                    "Initial prompt already fills the rollout context window: "
                    f"prompt_tokens={prompts_repeated.shape[1]}, max_model_len={max_model_len}."
                )
        _tokenized_responses = self._tokenize_with_left_truncation(
            response_str_list,
            max_length=response_token_budget,
        )
        responses = _tokenized_responses['input_ids']
        responses_attention_mask = _tokenized_responses['attention_mask']

        # The response is token-left-truncated to fit the model context. When
        # this removes old assistant/tool turns, trim the same oldest IG
        # reward/query pairs before token-level reward parsing and semantic
        # grouping. This is an explicit structural alignment step, not a
        # fallback that keeps mismatched metadata.
        truncated_ig_pairs = 0
        truncated_ig_samples = 0
        prefix_padded_ig_slots = 0
        prefix_padded_ig_samples = 0
        for sample_idx in range(len(info_gain_rewards)):
            valid_length = int(responses_attention_mask[sample_idx].sum().item())
            retained_response = self.tokenizer.decode(
                responses[sample_idx][:valid_length], skip_special_tokens=False
            )
            expected_ig_count = max(0, self._count_serialized_assistant_turns(retained_response) - 1)
            actual_ig_count = len(info_gain_rewards[sample_idx])
            if len(info_gain_query_eligible[sample_idx]) != actual_ig_count:
                raise RuntimeError(
                    "IG query-eligibility alignment failure before response truncation: "
                    f"sample={sample_idx}, eligibility={len(info_gain_query_eligible[sample_idx])}, "
                    f"computed_ig_rewards={actual_ig_count}"
                )
            if len(rewrite_bound_rewards[sample_idx]) != actual_ig_count:
                raise RuntimeError(
                    "Rewrite-Bound / IG alignment failure before response truncation: "
                    f"sample={sample_idx}, rewrite_slots={len(rewrite_bound_rewards[sample_idx])}, "
                    f"computed_ig_rewards={actual_ig_count}"
                )
                # A left-truncated suffix can begin inside a tool result. The
                # text parser then sees that fragment before the next assistant
                # marker and reports one or more synthetic assistant turns.
                # Preserve the real reward/query chronology by prefix-padding
                # only those synthetic slots with a zero reward and no query.
                full_response_length = len(self.tokenizer(response_str_list[sample_idx])["input_ids"])
                if full_response_length <= valid_length:
                    raise RuntimeError(
                        "IG reward/response alignment failure without response truncation: "
                        f"sample={sample_idx}, serialized_ig_boundaries={expected_ig_count}, "
                        f"computed_ig_rewards={actual_ig_count}"
                    )
                paired_queries = tool_action_queries[sample_idx][:actual_ig_count]
                paired_eligibility = info_gain_query_eligible[sample_idx][:actual_ig_count]
                paired_rewrite_rewards = rewrite_bound_rewards[sample_idx][:actual_ig_count]
                if len(paired_queries) != actual_ig_count:
                    raise RuntimeError(
                        "IG query/action alignment failure before response truncation: "
                        f"sample={sample_idx}, tool_actions={len(tool_action_queries[sample_idx])}, "
                        f"computed_ig_rewards={actual_ig_count}"
                    )
                synthetic_slots = expected_ig_count - actual_ig_count
                info_gain_rewards[sample_idx] = [0.0] * synthetic_slots + info_gain_rewards[sample_idx]
                info_gain_query_eligible[sample_idx] = [False] * synthetic_slots + paired_eligibility
                rewrite_bound_rewards[sample_idx] = [None] * synthetic_slots + paired_rewrite_rewards
                tool_action_queries[sample_idx] = [None] * synthetic_slots + paired_queries
                prefix_padded_ig_slots += synthetic_slots
                prefix_padded_ig_samples += 1
                continue
            if expected_ig_count == actual_ig_count:
                continue

            paired_queries = tool_action_queries[sample_idx][:actual_ig_count]
            paired_eligibility = info_gain_query_eligible[sample_idx][:actual_ig_count]
            paired_rewrite_rewards = rewrite_bound_rewards[sample_idx][:actual_ig_count]
            if len(paired_queries) != actual_ig_count:
                raise RuntimeError(
                    "IG query/action alignment failure before response truncation: "
                    f"sample={sample_idx}, tool_actions={len(tool_action_queries[sample_idx])}, "
                    f"computed_ig_rewards={actual_ig_count}"
                )
            truncated_ig_pairs += actual_ig_count - expected_ig_count
            truncated_ig_samples += 1
            if expected_ig_count:
                info_gain_rewards[sample_idx] = info_gain_rewards[sample_idx][-expected_ig_count:]
                info_gain_query_eligible[sample_idx] = paired_eligibility[-expected_ig_count:]
                rewrite_bound_rewards[sample_idx] = paired_rewrite_rewards[-expected_ig_count:]
                tool_action_queries[sample_idx] = paired_queries[-expected_ig_count:]
            else:
                info_gain_rewards[sample_idx] = []
                info_gain_query_eligible[sample_idx] = []
                rewrite_bound_rewards[sample_idx] = []
                tool_action_queries[sample_idx] = []
        if truncated_ig_pairs:
            print(
                "[IGPO] Response truncation removed "
                f"{truncated_ig_pairs} IG reward/action pairs across {truncated_ig_samples} trajectories"
            )
        if prefix_padded_ig_slots:
            print(
                "[IGPO] Response truncation inserted "
                f"{prefix_padded_ig_slots} zero-IG prefix slots across "
                f"{prefix_padded_ig_samples} trajectories"
            )

        attention_mask = torch.cat((prompts_attention_mask, responses_attention_mask), dim=-1)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)
        
        message_tensor = DataProto.from_dict({
            'prompts': prompts_repeated,
            'responses': responses,
            'input_ids': torch.cat((prompts_repeated, responses), dim=-1),
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        })
        message_tensor.meta_info.update(meta_info)
        message_tensor.non_tensor_batch['agent_grpo_idx'] = np.array(agent_grpo_idx, dtype=object)

        if self.config.rewrite_bound_advantage:
            for sample_idx, rewards in enumerate(info_gain_rewards):
                if len(rewrite_bound_rewards[sample_idx]) != len(rewards):
                    raise RuntimeError(
                        "Rewrite-Bound final metadata alignment failure: "
                        f"sample={sample_idx}, rewrite_slots={len(rewrite_bound_rewards[sample_idx])}, "
                        f"ig_rewards={len(rewards)}"
                    )
            rewrite_reward_array = np.empty(len(rewrite_bound_rewards), dtype=object)
            rewrite_reward_array[:] = rewrite_bound_rewards
            message_tensor.non_tensor_batch['rewrite_bound_rewards'] = rewrite_reward_array
            print(
                "[RewriteBound] valid first-turn rewards="
                f"{sum(sum(value is not None for value in rewards) for rewards in rewrite_bound_rewards)}",
                flush=True,
            )

        if self.config.query_group_advantage == "semantic":
            # IG reward k is assigned to the end of assistant turn k. A final
            # tool call with no following observation has no IG reward and is
            # intentionally trimmed here.
            aligned_query_texts = []
            aligned_query_eligibility = []
            for sample_idx, rewards in enumerate(info_gain_rewards):
                action_queries = tool_action_queries[sample_idx]
                action_eligibility = info_gain_query_eligible[sample_idx]
                if len(action_queries) < len(rewards):
                    raise RuntimeError(
                        "Query Grouped-IGPO alignment failure: "
                        f"sample={sample_idx}, tool_actions={len(action_queries)}, ig_rewards={len(rewards)}"
                    )
                if len(action_eligibility) != len(rewards):
                    raise RuntimeError(
                        "Query Grouped-IGPO eligibility alignment failure: "
                        f"sample={sample_idx}, eligibility={len(action_eligibility)}, ig_rewards={len(rewards)}"
                    )
                paired_queries = []
                for query, eligible in zip(action_queries[:len(rewards)], action_eligibility):
                    if not isinstance(eligible, (bool, np.bool_)):
                        raise RuntimeError(
                            "Query Grouped-IGPO eligibility metadata is malformed: "
                            f"sample={sample_idx}, value={eligible!r}"
                        )
                    # A structurally-present zero IG has no valid target-side
                    # comparison. Its query is deliberately a placeholder so
                    # it cannot shift the statistics of a real semantic group.
                    paired_queries.append(query if eligible else None)
                aligned_query_texts.append(paired_queries)
                aligned_query_eligibility.append(list(action_eligibility))

            unique_queries = []
            seen_queries = set()
            for sample_queries in aligned_query_texts:
                for query in sample_queries:
                    if query is not None and query not in seen_queries:
                        seen_queries.add(query)
                        unique_queries.append(query)

            query_to_embedding = {}
            if unique_queries:
                if self.client is None or not hasattr(self.client, "embed_queries"):
                    raise RuntimeError("Semantic Query Grouped-IGPO requires MessageClient.embed_queries")
                raw_embeddings = self.client.embed_queries(unique_queries)
                embedding_matrix = np.asarray(raw_embeddings, dtype=np.float32)
                if embedding_matrix.ndim != 2 or embedding_matrix.shape[0] != len(unique_queries):
                    raise RuntimeError(
                        "Local retriever /embed returned invalid matrix: "
                        f"shape={embedding_matrix.shape}, queries={len(unique_queries)}"
                    )
                if not np.isfinite(embedding_matrix).all():
                    raise RuntimeError("Local retriever /embed returned non-finite values")
                norms = np.linalg.norm(embedding_matrix, axis=1)
                if np.any(norms <= 0) or not np.allclose(norms, 1.0, atol=1e-3, rtol=1e-3):
                    raise RuntimeError("Local retriever /embed must return L2-normalized embeddings")
                query_to_embedding = {
                    query: embedding_matrix[index].tolist()
                    for index, query in enumerate(unique_queries)
                }

            aligned_query_embeddings = [
                [None if query is None else query_to_embedding[query] for query in sample_queries]
                for sample_queries in aligned_query_texts
            ]
            query_text_array = np.empty(len(aligned_query_texts), dtype=object)
            query_text_array[:] = aligned_query_texts
            query_embedding_array = np.empty(len(aligned_query_embeddings), dtype=object)
            query_embedding_array[:] = aligned_query_embeddings
            query_eligibility_array = np.empty(len(aligned_query_eligibility), dtype=object)
            query_eligibility_array[:] = aligned_query_eligibility
            message_tensor.non_tensor_batch['ig_query_texts'] = query_text_array
            message_tensor.non_tensor_batch['ig_query_embeddings'] = query_embedding_array
            message_tensor.non_tensor_batch['ig_query_eligible'] = query_eligibility_array
            print(
                "[IGPO] Query grouping metadata: "
                f"ig_actions={sum(len(items) for items in aligned_query_texts)}, "
                f"eligible_ig_actions={sum(sum(items) for items in aligned_query_eligibility)}, "
                f"valid_queries={len(unique_queries)}"
            )

        print("generation completed")

        print(f"node {node_rank} message_string_list {len(message_string_list)}")

        return message_string_list, message_tensor, info_gain_rewards, first_retrieved_passages
