# InteractiveChat-R1

This README documents the current **Feedback-GRPO InsCiT 3B** experiment only.

## Method

Feedback-GRPO alternates two independent Qwen2.5-3B-Instruct policies:

- **System policy**: produces the current query's action-formatted response.
- **User policy**: reads only the System terminal content plus dialogue context and produces one feedback message. It is a learned policy, not a value critic and not an external vLLM simulator.

Both policies have their own parameters and optimizers. Training uses GRPO without
a learned value head.

| Component | Current setting |
| --- | --- |
| Dataset | static ConvAgent InsCiT |
| Model initialization | two independent Qwen2.5-3B-Instruct policies |
| System source cost | 1 first response + 1 User feedback + 8 revised responses |
| System GRPO loss | revised System responses only; first-response loss weight = 0 |
| User phase | 1 shared first System response, 8 User feedbacks, 1 revised response per feedback |
| System reward | maximum-reference answer F1 + 0.5 × action score |
| Retrieval / extra format reward | disabled |
| Rollout group size | 8 |
| Train / monitor source batch | 128 / 256 |
| User / System alternation | 5 User updates, then 5 System updates |
| Holdout | 10% dialogue-disjoint source split, seed 42 |
| Validation | every 5 completed System updates |
| Early stopping | 3 selected-track validations without strict improvement |

A valid permissible terminal action has action score +1; wrong or malformed
actions have -0.5. For answer actions, F1 is the maximum token-set F1 over
all released answer references. Clarify/nonanswer actions receive no answer-text
F1. Retrieval reward is not used.

In the System phase, a malformed first response is retried at most twice. A
source with no legal first terminal action is skipped. Malformed revised
responses are filtered; an effective System GRPO group requires at least two
valid revised candidates.

## One-time environment setup

Run these steps once in the server-side InteractiveChat-R1 project:

~~~bash
MAX_JOBS=8 bash scripts/install_h100_eval_env.sh interactivechat-r1
conda activate interactivechat-r1
python -m pip install -r requirements-eval-metrics.txt
python -m pip install -r requirements-retriever.txt
# Transformers in this environment requires huggingface_hub < 1.0.
python -m pip install -U wandb "huggingface_hub>=0.26,<1.0"
~~~

Authenticate W&B and Hugging Face once per server. The Python Hugging Face login
works regardless of whether the installed hf CLI uses the old or new auth
subcommands. Do not put an access token in shell history, a script, or Git.

~~~bash
wandb login

python - <<'PY'
from huggingface_hub import HfApi, login
login()  # prompts securely if no token is cached
print("Hugging Face account:", HfApi().whoami()["name"])
PY
~~~

## Download only the 3B model and InsCiT assets

All model paths passed to the launcher must be local model directories containing
config.json.

~~~bash
mkdir -p models data/static_convagent_raw collection
hf download Qwen/Qwen2.5-3B-Instruct \
  --local-dir models/Qwen2.5-3B-Instruct

hf download DrewZhang/conv --repo-type dataset \
  --include "ConvAgent/**" \
  --local-dir data/static_convagent_raw
~~~

Download the dataset-matched InsCiT passage shards and build the row-aligned
corpus/index. Both retrieval files must come from InsCiT. Reserve roughly 350 GB
of free disk space for the downloaded shards, corpus, and merged index.

~~~bash
hf download DrewZhang/inscit-passages-index --repo-type dataset \
  --local-dir collection/inscit --max-workers 4

python scripts/build_inscit_corpus.py \
  --collection-dir collection/inscit

cat collection/inscit/e5_Flat.index.part_* \
  > collection/inscit/e5_Flat.index

test -f data/static_convagent_raw/ConvAgent/inscit/inscit_train.parquet
test -f data/static_convagent_raw/ConvAgent/inscit/inscit_test.parquet
test -f collection/inscit/inscit_index.jsonl
test -f collection/inscit/e5_Flat.index
~~~

## GPU-FAISS setup for the InsCiT retriever

GPU-FAISS must run on GPUs separate from Feedback-GRPO training. The documented
workflow places the GPU-resident InsCiT index on GPUs 2 and 3 and the two 3B
training policies on GPUs 0 and 1.

~~~bash
conda activate interactivechat-r1
bash scripts/install_gpu_faiss_cuda121.sh

CUDA_VISIBLE_DEVICES=2,3 python - <<'PY'
import faiss
print("Faiss:", getattr(faiss, "__version__", "unknown"))
print("GPU API:", hasattr(faiss, "GpuMultipleClonerOptions"))
print("Visible GPUs:", faiss.get_num_gpus())
assert hasattr(faiss, "GpuMultipleClonerOptions")
assert faiss.get_num_gpus() == 2
PY
~~~

The verification must print GPU API: True and Visible GPUs: 2. If the installer
proposes removing PyTorch, vLLM, FlashAttention, or the Conda environment,
cancel that transaction and resolve the package conflict first.

## Start the InsCiT GPU-FAISS retriever on GPUs 2 and 3

Start this before Feedback-GRPO training. The large FAISS index and E5 query
encoder stay on GPUs 2 and 3, separate from the two System/User policies.

~~~bash
mkdir -p logs
nohup env \
  CUDA_VISIBLE_DEVICES=2,3 \
  RETRIEVER_FAISS_GPU=true \
  RETRIEVER_INDEX_PATH=$PWD/collection/inscit/e5_Flat.index \
  RETRIEVER_CORPUS_PATH=$PWD/collection/inscit/inscit_index.jsonl \
  RETRIEVER_MODEL_PATH=intfloat/e5-base-v2 \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  bash scripts/run_local_retriever_server.sh \
  > logs/inscit_feedback_grpo_retriever.log 2>&1 &

tail -f logs/inscit_feedback_grpo_retriever.log
~~~

Wait for Uvicorn running on http://127.0.0.1:8002. The training launcher also
performs a /retrieve readiness probe and exits before training if the retriever
cannot serve requests.

## Train Feedback-GRPO on GPUs 0 and 1

No external user-simulator server is required: the learned User policy runs
inside the Feedback-GRPO process. This command runs the feedback-refinement
setting, so it selects and early-stops using the post-feedback second System
response. It must be reported as a two-round interaction result, not as a
directly comparable one-response baseline score.

~~~bash
nohup env \
  CUDA_VISIBLE_DEVICES=0,1 \
  N_GPUS=2 \
  ULYSSES_SEQUENCE_PARALLEL_SIZE=1 \
  SYSTEM_MODEL_PATH=$PWD/models/Qwen2.5-3B-Instruct \
  USER_MODEL_PATH=$PWD/models/Qwen2.5-3B-Instruct \
  SELECTION_SETTING=feedback-refinement \
  EXPERIMENT_NAME=feedback_grpo_inscit_qwen25_3b_feedback_second_only_cuda01_v1 \
  HOLDOUT_FRACTION=0.10 \
  SPLIT_SEED=42 \
  ROLLOUT_N=8 \
  TRAIN_BATCH_SIZE=128 \
  VAL_BATCH_SIZE=256 \
  USER_UPDATES_PER_PHASE=5 \
  SYSTEM_UPDATES_PER_PHASE=5 \
  VALIDATE_EVERY=5 \
  EARLY_STOP_PATIENCE=3 \
  MAX_EMPTY_SYSTEM_BATCHES=20 \
  ROLLOUT_BATCH_SIZE=8 \
  LOGPROB_BATCH_SIZE=2 \
  ROLLOUT_GPU_MEMORY_UTILIZATION=0.10 \
  BERT_SCORE_DEVICE=cpu \
  BERT_SCORE_BATCH_SIZE=8 \
  WANDB_ENABLED=true \
  WANDB_PROJECT=interactivechat-r1 \
  WANDB_RUN_GROUP=inscit_feedback_grpo_3b_feedback_second_only_cuda01 \
  HF_UPLOAD=true \
  HF_UPLOAD_NUM_WORKERS=4 \
  HF_SYSTEM_REPO_ID=DrewZhang/interactivechat-r1-feedback-grpo-inscit-qwen25-3b-feedback-second-only-cuda01-system \
  HF_USER_REPO_ID=DrewZhang/interactivechat-r1-feedback-grpo-inscit-qwen25-3b-feedback-second-only-cuda01-user \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  bash scripts/run_feedback_grpo_inscit_3b_train_eval_export_upload.sh \
  > logs/feedback_grpo_inscit_3b_feedback_second_only_cuda01_v1.log 2>&1 &
~~~

The launcher validates both response views every five System updates, saves
only the best selected System/User checkpoint pair, evaluates that pair on the
official InsCiT test set, exports both policies, and uploads both repositories
when HF_UPLOAD=true.

For a direct-response experiment, use a fresh EXPERIMENT_NAME, W&B group, and
two new HF repository IDs, and set SELECTION_SETTING=direct-response. That
track selects by the first response. Do not resume an older pre-second-only
experiment into this recipe.

## Results and artifacts

~~~text
outputs/feedback_grpo/<experiment>/final_checkpoint.txt
exports/feedback_grpo/<experiment>/system/
exports/feedback_grpo/<experiment>/user/

eval_log/feedback_grpo/<experiment>/monitor/system_step_*/
  first_response_metrics/metrics_summary.json
  feedback_round_two_metrics/metrics_summary.json

eval_log/feedback_grpo/<experiment>/official_test/
  first_response_metrics/metrics_summary.json
  feedback_round_two_metrics/metrics_summary.json
~~~

W&B contains separate user/*, system/*, val/direct_response/*, and
val/feedback_refinement/* series. For this launch, the selected metric is
val/feedback_refinement/composite.

For the reward equations, GRPO grouping/filtering rules, checkpoint semantics,
and all diagnostics, see [docs/feedback_grpo.md](docs/feedback_grpo.md).

