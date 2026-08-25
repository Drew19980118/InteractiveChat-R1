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
# from . import gsm8k, math, prime_math, prime_code
import json

def _default_compute_score(data_source, prompt_str, solution_str, ground_truth, extra_info=None, val_type='f1', info_gain_reward=None, batch_size=1, tokenizer=None, is_validation=False):
    if type(data_source) != str:
        reslist = []
        if val_type == 'llm':
            # llm only supports batch mode
            from . import llm_judge
            reslist = llm_judge.compute_score_batch(prompt_str, solution_str, ground_truth, data_source, batch_size)
        else:
            for data_source_e,solution_str_e,ground_truth_e in zip(data_source, solution_str, ground_truth):
                if data_source_e in ['nq', "2wiki", "Bamboogle", "hotpotqa", "musique", "tq", "popqa","browse_comp", "browse_comp_zh", "xbench_deepsearch","hotpot","zhihu", "inscit", "coral", "qrecc", "topiocqa"]:
                    from . import info_gain
                    res = info_gain.compute_score(solution_str_e, ground_truth_e, data_source, val_type=val_type, info_gain_reward=info_gain_reward, tokenizer=tokenizer, is_validation=is_validation)
                    reslist.append(res)
                elif data_source_e in ['future']:
                    from . import stock_judge
                    res = stock_judge.compute_score(solution_str_e, ground_truth_e, data_source_e, val_type=val_type)
                    reslist.append(res)
                elif data_source_e in ['Factbench', 'politifact', 'liar2', 'elobench', 'Chinese_Rumor_Dataset', 'fever', 'lair', 'MDFEND-Weibo21', 'twitter_factchecking_test', 'factchecker_history','elobench','health_fact']:
                    from . import fact_test
                    res = fact_test.compute_score(solution_str_e, ground_truth_e, data_source_e, val_type=val_type)
                    reslist.append(res)
                else:
                    raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
        return reslist 
    else:
        if data_source == "openai/gsm8k":
            from . import gsm8k

            res = gsm8k.compute_score(solution_str, ground_truth)
        elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
            from . import math

            res = math.compute_score(solution_str, ground_truth)
            # [Optional] Math-Verify Integration
            # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
            # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
            # To use it, override the `compute_score` function with the following implementation:

            # from . import math_verify
            # res = math_verify.compute_score(solution_str, ground_truth)
        elif data_source == "math_dapo" or data_source.startswith("aime"):
            from . import math_dapo

            res = math_dapo.compute_score(solution_str, ground_truth)
        elif data_source in [
            "numina_aops_forum",
            "numina_synthetic_math",
            "numina_amc_aime",
            "numina_synthetic_amc",
            "numina_cn_k12",
            "numina_olympiads",
        ]:
            from . import prime_math

            res = prime_math.compute_score(solution_str, ground_truth)
        elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
            from . import prime_code

            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
        elif data_source in ["hiyouga/geometry3k"]:
            from . import geo3k

            res = geo3k.compute_score(solution_str, ground_truth)
        elif data_source in ['elobench', 'Factbench', 'nq', "2wiki", "Bamboogle", "hotpotqa", "musique", "tq", "popqa", "browse_comp", "browse_comp_zh", "xbench_deepsearch","hotpot","zhihu", "inscit", "coral", "qrecc", "topiocqa"]:
            from . import info_gain
            res = info_gain.compute_score(solution_str, ground_truth, data_source, val_type=val_type, info_gain_reward=info_gain_reward, tokenizer=tokenizer, is_validation=is_validation)
        elif data_source in ['politifact', 'liar2', 'elobench', 'Chinese_Rumor_Dataset', 'fever', 'lair', 'MDFEND-Weibo21', 'twitter_factchecking_test', 'factchecker_history', 'health_fact']:
            from . import fact_test
            res = fact_test.compute_score(solution_str, ground_truth, data_source, val_type=val_type, info_gain_reward=info_gain_reward)
        else:
            raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

        # if isinstance(res, dict):
        #     return res
        # elif isinstance(res, (int, float, bool)):
        #     return float(res)
        # else:
        #     return float(res[0])
        return res


