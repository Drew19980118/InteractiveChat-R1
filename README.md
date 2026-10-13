# InteractiveChat-R1: InSCIT static vs GPT-4.1 simulated-dynamic context

This repository is configured for one experiment only: test whether ConvAgent
trained on **static ground-truth dialogue histories** behaves differently from
ConvAgent trained on **frozen GPT-4.1 simulated-dynamic histories**.

The experiment is not a human online-dialogue study. Its correct name is
**GPT-4.1 simulated dynamic context**.

## What is held fixed

The two datasets contain exactly the same current query, direct-answer
references, evidence labels, expected action, `pair_id`, and conversation-level
train/monitor assignment. The sole differing field is the public history.

- `static`: each preceding Assistant turn is the first non-empty InSCIT
  `directAnswer` reference.
- `dynamic`: each preceding Assistant turn is produced by frozen
  `gpt-4.1-2025-04-14` after one local retrieval. A GPT-4.1 User Judge can
  issue up to two natural corrective feedback messages before the final answer
  is accepted into the public history.

For each dynamic historical turn the builder executes:

```text
public history + current query
  -> GPT-4.1 function call: standalone search query
  -> local InSCIT E5/FAISS retriever: top-3 passages
  -> GPT-4.1: <answer>...</answer>, conditioned on those passages
  -> GPT-4.1 User Judge: directness-only decision / optional feedback
```

The Judge does not see ground-truth answers and checks only whether the answer
addresses the question, not factual correctness. A separate GPT-4.1 dependency
check can see prior ground truth solely to decide whether a dynamic history is
sufficient for the next query. Static and dynamic rows are removed together
when that gate fails.

The RL reward in both runs is the same full ConvAgent reward:

\[
R = R_{\mathrm{F1}} + 0.5(R_{\mathrm{retrieval}} + R_{\mathrm{action}}).
\]

No BERTScore, IGT, sentence-BERTScore, or format reward is used for training.

## GPU layout

| GPUs | Process |
|---|---|
| `0,1` | InSCIT E5 encoder + GPU-FAISS retriever on port `8002` |
| `3,4` | Two-GPU Qwen2.5-3B ConvAgent training and final evaluation |

Keep the retriever running for data construction, RL training, monitor
validation, and 2x2 evaluation.

## 1. One-time environment setup

The reproducible setup is Linux x86_64, Python 3.10, PyTorch 2.4.0+cu121,
vLLM 0.6.3, and FlashAttention 2.5.8. Do **not** run a broad
`pip install -r requirements.txt`: it can replace the pinned PyTorch/vLLM
runtime.

The installer builds FlashAttention with Ninja and multiple compile workers.
`MAX_JOBS=8` is the default; raise it only if the host has ample CPU and RAM.
The installer refuses to compile FlashAttention with a CUDA toolkit whose
version differs from `torch.version.cuda`, preventing the CUDA 13 vs cu121
mismatch.

```bash
cd /workspace/zwd/InteractiveChat-R1

MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 \
  bash scripts/install_h100_eval_env.sh interactivechat-r1

conda activate interactivechat-r1

# GPT-4.1 client, SwanLab, BERTScore, and parquet/request dependencies.
python -m pip install --upgrade-strategy only-if-needed \
  -r requirements-context-ablation.txt
python -m pip check

python - <<'PY'
import torch
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("flash-attn: OK")
PY
```

If a pre-existing environment is missing only FlashAttention, do not use a
plain `pip install flash-attn` command. First expose a CUDA toolkit matching
the installed Torch CUDA version, then rebuild dependency-free with multiple
workers:

```bash
conda activate interactivechat-r1

export CUDA_HOME=/usr/local/cuda-12.1   # must match: python -c 'import torch; print(torch.version.cuda)'
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 \
  python -m pip install --force-reinstall --no-deps --no-build-isolation \
  --no-cache-dir --verbose 'flash-attn==2.5.8'

python -c 'from flash_attn.bert_padding import index_first_axis; print("flash-attn: OK")'
```

`--verbose` prints compilation commands; in another terminal, use `htop` or
`ps -ef | grep -E "ninja|cc1plus"` to observe parallel compilation. Do not
use a CUDA 13 system compiler to build an extension for Torch `+cu121`.

Authenticate SwanLab once in an interactive terminal:

```bash
swanlab login
```

There is no Hugging Face login or model upload in this experiment.

## 2. Required local assets

Before starting, the following paths must exist:

```bash
test -d models/Qwen2.5-3B-Instruct
test -f data/inscit_train.json
test -f data/inscit_test.json
test -f collection/inscit/e5_Flat.index
test -f collection/inscit/inscit_index.jsonl
```

Install GPU FAISS once in the locked Conda environment if it is not already
available:

```bash
conda activate interactivechat-r1
FAISS_VISIBLE_GPUS=0,1 FAISS_EXPECTED_GPUS=2 \
  bash scripts/install_gpu_faiss_cuda121.sh
```

## 3. Start the InSCIT retriever on GPUs 0 and 1

```bash
cd /workspace/zwd/InteractiveChat-R1
mkdir -p logs

nohup env \
  CUDA_VISIBLE_DEVICES=0,1 \
  RETRIEVER_FAISS_GPU=true \
  RETRIEVER_INDEX_PATH=$PWD/collection/inscit/e5_Flat.index \
  RETRIEVER_CORPUS_PATH=$PWD/collection/inscit/inscit_index.jsonl \
  RETRIEVER_MODEL_PATH=intfloat/e5-base-v2 \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  bash scripts/run_local_retriever_server.sh \
  > logs/inscit_context_ablation_retriever_cuda01.log 2>&1 &

tail -f logs/inscit_context_ablation_retriever_cuda01.log
```

Wait for Uvicorn on `http://127.0.0.1:8002`, then verify it:

```bash
curl -sS -X POST http://127.0.0.1:8002/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["InSCIT readiness"],"topk":3,"return_scores":true}'
```

## 4. Configure GPT-4.1 securely and construct paired data

Use a newly rotated API key belonging to the server/account that has permitted
OpenAI API access. Never add the key to README, a shell script, Git, command
history, or a `nohup env OPENAI_API_KEY=...` command.

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
echo
```

Run a two-conversation smoke build first. Successful requests are cached;
`RESUME=true` continues interrupted work without repeating completed API calls.

```bash
cd /workspace/zwd/InteractiveChat-R1

nohup env \
  OUTPUT_DIR=$PWD/data/inscit_context_ablation/gpt41_smoke \
  BUILDER_PROVIDER=openai \
  BUILDER_MODEL=gpt-4.1-2025-04-14 \
  LIMIT_CONVERSATIONS=2 \
  WORKERS=2 \
  MAX_TRANSIENT_RETRIES=0 \
  RESUME=true \
  RETRIEVER_URL=http://127.0.0.1:8002/retrieve \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  bash scripts/run_build_inscit_context_ablation.sh \
  > logs/inscit_context_ablation_gpt41_smoke.log 2>&1 &

tail -f logs/inscit_context_ablation_gpt41_smoke.log
```

After checking the smoke `audit/` files, build the complete frozen dataset:

```bash
nohup env \
  OUTPUT_DIR=$PWD/data/inscit_context_ablation/gpt41_v1 \
  BUILDER_PROVIDER=openai \
  BUILDER_MODEL=gpt-4.1-2025-04-14 \
  WORKERS=4 \
  MAX_TRANSIENT_RETRIES=0 \
  RESUME=true \
  RETRIEVER_URL=http://127.0.0.1:8002/retrieve \
  LOG_FILE=$PWD/logs/inscit_dynamic_builder_gpt41.log \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  bash scripts/run_build_inscit_context_ablation.sh \
  > logs/inscit_context_ablation_gpt41_full.log 2>&1 &
```

`MAX_TRANSIENT_RETRIES=0` means retry forever for transient API/retriever
connection failures, timeouts, 408/429, and 5xx responses, with capped
exponential backoff and jitter. HTTP 400/401/403, model/schema failures, and
repeated model-format failures stop rather than silently filtering data.

Outputs:

```text
data/inscit_context_ablation/gpt41_v1/
  static/inscit_{train,test}.parquet
  dynamic/inscit_{train,test}.parquet
  audit/{train,test}/*.json
  paired_index.jsonl
  dropped_turns.jsonl
  retry_summary.json
  manifest.json
```

## 5. Train paired static and dynamic Qwen2.5-3B ConvAgent models on GPUs 3 and 4

The suite always trains **static first**, then starts a fresh Qwen2.5-3B model
for dynamic training. The two runs do not share parameters. A single paired,
conversation-disjoint 90/10 train/monitor split is created with seed 42.

Each run validates every five RL updates, selects by
`(F1 + BERTScore-F1 + NDCG@3) / 3`, and early-stops after three consecutive
validations without a strict improvement. Action accuracy is logged but is not
part of checkpoint selection.

```bash
nohup env \
  CUDA_VISIBLE_DEVICES=3,4 \
  N_GPUS=2 \
  ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
  MODEL_PATH=$PWD/models/Qwen2.5-3B-Instruct \
  DATA_ROOT=$PWD/data/inscit_context_ablation/gpt41_v1 \
  EXPERIMENT_PREFIX=convagent_inscit_gpt41_simulated \
  SWANLAB_PROJ_NAME=interactivechat-r1 \
  SWANLAB_RUN_GROUP=inscit_context_ablation_gpt41_seed42 \
  SWANLAB_MODE=online \
  SELECTION_BERTSCORE_DEVICE=cpu \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  bash scripts/run_inscit_context_ablation_convagent_3b_train_suite.sh \
  > logs/inscit_context_ablation_gpt41_train_cuda34.log 2>&1 &

tail -f logs/inscit_context_ablation_gpt41_train_cuda34.log
```

SwanLab shows independent runs named:

```text
convagent_inscit_gpt41_simulated_static_context_seed42
convagent_inscit_gpt41_simulated_dynamic_context_seed42
```

It records RL reward/loss/KL/entropy/learning-rate metrics and validation F1,
BERTScore F1, NDCG@3, action accuracy, and selection composite. No Hugging
Face upload is performed; the selected checkpoints remain under
`outputs/paper_static_convagent/`.

## 6. Run the final paired 2x2 test on GPUs 3 and 4

Run this after both training runs have stopped. It evaluates the two selected
checkpoints on both static and dynamic test histories: SS, SD, DS, and DD.

```bash
nohup env \
  CUDA_VISIBLE_DEVICES=3,4 \
  N_GPUS=2 \
  ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
  MODEL_PATH=$PWD/models/Qwen2.5-3B-Instruct \
  DATA_ROOT=$PWD/data/inscit_context_ablation/gpt41_v1 \
  STATIC_EXPERIMENT=convagent_inscit_gpt41_simulated_static_context_seed42 \
  DYNAMIC_EXPERIMENT=convagent_inscit_gpt41_simulated_dynamic_context_seed42 \
  SUMMARY_DIR=$PWD/eval_results/inscit_context_ablation_gpt41_seed42 \
  BERT_SCORE_DEVICE=cpu \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  bash scripts/run_inscit_context_ablation_2x2_eval.sh \
  > logs/inscit_context_ablation_gpt41_2x2_eval_cuda34.log 2>&1 &
```

The final paired report is:

```text
eval_results/inscit_context_ablation_gpt41_seed42/context_ablation_2x2.md
```

It reports F1, BERTScore F1, NDCG@3, action accuracy, terminal/format success,
sample count, history-bearing turns, dependency-required and independent
subsets, and pair-level bootstrap confidence intervals.
