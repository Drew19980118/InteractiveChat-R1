# Copyright 2025 Individual Contributor: Mert Unsal
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

from verl import DataProto
# from verl.workers.reward_manager import register
import asyncio

from openai import AsyncOpenAI
from typing import Dict, List, Any
import re
from functools import reduce
from operator import or_


def filter_think(text):
    import re
    think = ''
    if '<think>' in text and '</think>' in text:
        if re.match(r'<think>([\s\S]*)</think>', text, re.S) is not None:
            think = re.match(r'<think>([\s\S]*)</think>', text, re.S).group(0)
        ans = re.sub(r'<think>[\s\S]*</think>', '', text, re.S)
        return think, ans
    else:
        return '', text


# @register("group_async_batch")
class GroupBatchRewardManager:
    """
    A batch reward manager that computes rewards for a batch of data.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards.
        reward_fn_key (str): The key to use for the reward function.
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        self.reward_meta = ['data_source', 'solution_str', 'ground_truth', 'extra_info',]
        self.model = reward_kwargs.get("model", 'auto')
        self.aclient = AsyncOpenAI(base_url=self.reward_kwargs.get('url', 'http://localhost:8000/v1'), api_key='congyu', timeout=3600, max_retries=5)
        print(f'init manager ok {self.reward_kwargs} {self.model }')

    async def call_llm(self, messages, semaphore):
        async with semaphore:
            try:
                response = await self.aclient.chat.completions.create(messages=messages, model=self.model, **self.reward_kwargs.get('chat_kwargs', {}))
                content = response.choices[0].message.content
                if content is None:
                    content = ''
                return content
            except Exception:
                import traceback
                print(traceback.format_exc())
                return ''

    def apply_group_length_penlty(self, grouped_data):
        raise NotImplementedError

    def reduce_grouped_data_pairwise(self, grouped_data: Dict[str, List], operator):
        # group wise reduce
        # Can compute many things here, e.g. lambda x,y: is_equal(x['solution_str'], x['ground_truth']) + is_equal(y['solution_str'], y['ground_truth'])
        #  lambda x,y: is_equal(x['solution_str'], x['ground_truth'])
        return reduce(operator, [param for uidr, param in grouped_data.items()])

    def reduce_grouped_data_listrwise(self, grouped_data: Dict[str, List], operator):
        # group wise reduce
        # Can compute many things here, e.g. lambda x,y: is_equal(x['solution_str'], x['ground_truth']) + is_equal(y['solution_str'], y['ground_truth'])
        #  lambda x,y: is_equal(x['solution_str'], x['ground_truth'])
        return operator(filter(lambda x: x['result'], [param for uidr, param in grouped_data.items()]))

    def build_message(self, question, answer, ground_truth):  # sample wise reward
        messages = [
            {
                "role": 'user',
                "content": f"""Please check if the following pred answer is objectively reasonable
<question> {question} </question>
<ground truth answers> {ground_truth} </ground truth answers>
<pred answer> {answer} </pred answer>

Output using <judge>yes/no</judge>"""
            }
        ]
        return messages

    def rm_judge_score(self, rm_response) -> float:  # sample wise reward
        think, rm_response = filter_think(rm_response)
        result = re.findall(r'<judge>(.*?)</judge>', rm_response)
        # Post-processing logic
        if len(result) > 0 and result[-1] == 'yes':
            return {"score": 0.2, "judge_valid": 1}
        elif len(result) > 0 and result[-1] == 'no':
            return {"score": -0.2, "judge_valid": 1}
        else:
            return {"score": -0.2, "judge_valid": 0}

    def build_grouped_messages(self, grouped_data: Dict[str, List]):  # group wise reward
        answers = []
        for uidr, param in grouped_data.items():
            question = param['question_str']
            ground_truth = param['ground_truth']
            think, solution_str = filter_think(param['solution_str'])
            answers.append(f"<answers_{uidr}>{solution_str}</answers_{uidr}>")
        gp_answer = '\n'.join(answers)
        messages = [
            {"role": "system", "content": """Based on the given <question> and <answer_$id> for different $ids, score each answer using format <judge_$id>0~10</judge_$id>"""},
            {"role": "user", "content": f"<question>{question}</question>\n\n<grount_truth>{ground_truth}</ground_truth>\n\n{gp_answer}"}
        ]
        return messages

    def rm_judge_gp_score(self, rm_response) -> Dict[int, float]:  # group wise reward
        think, rm_response = filter_think(rm_response)
        matches = re.findall(r'<judge_(\d+)>(.*?)</judge_\d+>', rm_response, flags=re.DOTALL)
        result_dict = {int(judge_id): float(score) for judge_id, score in matches}
        return result_dict

    async def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        semaphore = asyncio.Semaphore(self.reward_kwargs.get("concurrency", 10))

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        tasks = []
        tasks_grouped = []
        scores = []
        gp_scores_dict = {}
        gp_rm_scores = []
        rm_scores = []
        rule_scores = []
        total_scores = []
        params = []
        already_printed = {}
        uids = []
        grouped_messages = defaultdict(dict)
        for i in range(len(data)):
            dataitem = data[i]
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
            extra_info = dataitem.non_tensor_batch.get("extra_info", None)
            data_source = dataitem.non_tensor_batch[self.reward_fn_key]
            uid = dataitem.non_tensor_batch['uid']
            uidr = dataitem.non_tensor_batch['uidr']
            question_str = prompt_str.split("user\n")[-1].split("assistant\n")[0]
            ground_truth = dataitem.non_tensor_batch["reward_model"].get("ground_truth", None)
            answer_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            messages = self.build_message(question=question_str, answer=answer_str, ground_truth=ground_truth)
            # Create async task
            tasks.append(self.call_llm(messages, semaphore))
            param = dict(data_source=data_source,
                         solution_str=answer_str,
                         ground_truth=ground_truth,
                         extra_info=extra_info,
                         uidr=uidr,
                         uid=uid,
                         question_str=question_str)
            grouped_messages[uid][uidr] = param
            params.append(param)

        for uid, grouped_data in sorted(grouped_messages.items(), key=lambda _: _[0]):
            _messages = self.build_grouped_messages(grouped_data)
            tasks_grouped.append(self.call_llm(_messages, semaphore))
            uids.append(uid)

        backtrace = defaultdict(dict)
        # update rule_scores
        for i in range(len(data)):
            param = params[i]
            compute_param = {k: v for k, v in param.items() if k in self.reward_meta}
            rule_score = self.compute_score(**compute_param)
            if isinstance(rule_score, float) or isinstance(rule_score, int):
                rule_score = {'score': rule_score}
            rule_scores.append(rule_score)
            backtrace[i]['rule_score'] = rule_score
        total_scores.append(rule_scores)

        # update rm_score
        # gather executes all tasks concurrently, maintaining order
        rm_responses = await asyncio.gather(*tasks)
        if rm_responses:
            for i in range(len(data)):
                param = params[i]
                rm_response = rm_responses[i]
                rm_score = self.rm_judge_score(rm_response)
                rm_scores.append(rm_score)
                backtrace[i]['rm_response'] = rm_response
        else:
            rm_scores = [{'score': 0} for i in range(len(data))]
        total_scores.append(rm_scores)

        # update gp_rm_score
        # gather executes all tasks concurrently, maintaining order
        rm_responses_gp = await asyncio.gather(*tasks_grouped)
        if rm_responses_gp:
            # uid -> map
            uid2gprm = {}
            for i in range(len(uids)):
                for uidr, score_uidr in self.rm_judge_gp_score(rm_responses_gp[i]).items():
                    gp_scores_dict[(uids[i], uidr)] = score_uidr
                    uid2gprm[uids[i]] = rm_responses_gp[i]
            # uid -> data_i
            for i in range(len(data)):
                param = params[i]
                rule_gp = gp_scores_dict.get((param['uid'], int(param['uidr'])), float('nan'))
                if rule_gp == rule_gp:
                    rule_gp = {'score': rule_gp, "gp_judge": rule_gp, "gp_judge_valid": 1}
                else:
                    rule_gp = {'score': 0, "gp_judge": rule_gp, "gp_judge_valid": 0}
                gp_rm_scores.append(rule_gp)
                backtrace[i]['rm_response_gp'] = uid2gprm[param['uid']]
        else:
            gp_rm_scores = [{'score': 0} for i in range(len(data))]
        total_scores.append(gp_rm_scores)

        # merge scores
        total_scores_for_accmu = list(zip(*total_scores))
        for i in range(len(data)):
            param = params[i]
            score = reduce(or_, total_scores_for_accmu[i])
            score['score'] = sum([x['score'] for x in total_scores_for_accmu[i]])

            scores.append(score)

            data_source = param['data_source']
            if already_printed.get(data_source, 0) < self.num_examine:
                prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=False)

                print("[judge prompt]", prompt_str)
                print("[response]", param['answer_str'])
                print("[judge_result]", backtrace[i].get('rm_response'))
                print("[judge_gp_result]", backtrace[i].get('rm_response_gp'))
                print("[ground_truth]", param['ground_truth'])
                print("[score]", scores[i])

        return scores

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        print(data.non_tensor_batch['uid'])
        print(data.non_tensor_batch['uidr'])
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        # scores = await self.verify(data)

        scores = asyncio.run(self.verify(data))
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
