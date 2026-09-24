# InteractiveChat-R1

This repository provides the current protocol-matched comparison among **ConvAgent**, **ChatR1**, and online **TurnPPO**. Older static-baseline entrypoints are retired; use only the latest scripts below.

## Current protocol

| Setting | Value |
|---|---:|
| Policy initialization | Qwen2.5-3B-Instruct or Qwen2.5-7B-Instruct |
| Static baseline rollout count | n=8 |
| TurnPPO rollout count | n=1 |
| Feedback-GRPO System rollout count | 1 first response + 8 revised responses per source |
| Feedback-GRPO System loss | revised-response GRPO only; first-response loss weight = 0 |
| Training / monitor batch | 128 / 256 |
| Prompt / response / model context | 4096 / 500 / 8192 tokens |
| PPO mini-batch / micro-batch | 64 / 1 per GPU |
| Holdout split | 10% dialogue-disjoint, first source user question, seed 42 |
| Monitor cadence | every 5 completed updates |
| Selection score | (clip(F1)+clip(BERTScore-F1)+clip(NDCG@3))/3 |
| Early stopping | 3 successive monitor checks without strict improvement |
| Persisted checkpoint | the one selected best checkpoint only |

Every method emits one answer. If a source sample has several acceptable answer references, answer training reward, monitor F1/BERTScore, and final F1/BERTScore use the **maximum** score over them. TopiOCQA NDCG@3 compares normalized passage text; other benchmarks compare passage IDs.

- **ConvAgent** uses GRPO and its action space: answer, clarify, nonanswer, and search. Its paper reward is maximum answer F1 plus direct-evidence reward. Clarify/nonanswer do not receive answer-text F1. Its final summary contains F1, BERTScore-F1, NDCG@3, and action accuracy.
- **ChatR1** uses PPO/GAE (gamma=lambda=1) with a separate actor and critic, both initialized from the matching 3B or 7B Qwen model. It retains one answer-only source row with all answer references and reports F1, BERTScore-F1, and NDCG@3.
- **TurnPPO** uses online PPO/GAE (gamma=.99, lambda=.95) and distinct 3B actor/critic models. A frozen Qwen-32B simulator gives public feedback only as later-turn context, never satisfaction or patience reward. Its terminal reward combines action/format terms with max answer F1 (or clarify F1 when gold is clarify), then performs dynamic and static InsCiT test evaluation.
- **Feedback-GRPO** alternates two independent 3B policies: a System policy that produces action-formatted responses and a learned User policy that generates one textual feedback message. The System phase samples one legal first response, one frozen-User feedback, and an eight-candidate revised-response GRPO group; only the revised responses update the System policy. The User phase uses the improvement from first to second System reward to update feedback tokens. It has independent direct-response and feedback-refinement selection tracks, and exports/uploads both selected policies.

## Setup

~~~bash
MAX_JOBS=8 bash scripts/install_h100_eval_env.sh interactivechat-r1
conda activate interactivechat-r1
python -m pip install -r requirements-eval-metrics.txt
python -m pip install -r requirements-retriever.txt
python -m pip install -U wandb huggingface_hub
wandb login
python - <<'PY'
from huggingface_hub import HfApi, login
login()  # securely prompts for a write-enabled token when one is not cached
print("Hugging Face account:", HfApi().whoami()["name"])
PY
~~~

Expected local paths:

~~~text
models/Qwen2.5-3B-Instruct/
models/Qwen2.5-7B-Instruct/
models/Qwen2.5-32B-Instruct/
data/static_convagent_raw/ConvAgent/
data/static_chatr1_raw/ChatR1/
data/raw/inscit_train.json
data/raw/inscit_test.json
data/sim_user_inscit_train.parquet
data/sim_user_inscit_test.parquet
collection/{inscit,qrecc}/
~~~

## Download model, passage, and dynamic-data assets

### Qwen policy and simulator models

All paths supplied to the launchers must be **local Hugging Face model directories** (they must contain `config.json`). Download the public models once per server:

~~~bash
mkdir -p models
hf download Qwen/Qwen2.5-3B-Instruct \
  --local-dir models/Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct
hf download Qwen/Qwen2.5-32B-Instruct \
  --local-dir models/Qwen2.5-32B-Instruct
~~~

The 3B and 7B models initialize the corresponding policy (and, where used,
the separately initialized critic). The 32B model is required only for the
frozen TurnPPO user simulator. Before a private-model download or an upload,
authenticate with `hf login`; verify the active account with `hf whoami`.

### Static ConvAgent and ChatR1 Parquets

Download the released static data before running either baseline:

~~~bash
hf download DrewZhang/conv --repo-type dataset \
  --include "ConvAgent/**" --local-dir data/static_convagent_raw
hf download DrewZhang/conv --repo-type dataset \
  --include "ChatR1/**" --local-dir data/static_chatr1_raw
~~~

### Passage corpora and E5 FAISS indexes

The local retriever requires a **dataset-matched pair**: one merged
`e5_Flat.index` and one row-aligned JSONL corpus. Do not mix an InsCiT index
with QReCC data (or the reverse). Keep the source shards after merging: they
make interrupted downloads resumable and allow the merged index to be rebuilt.

InsCiT uses `DrewZhang/inscit-passages-index` (14 FAISS shards and 5 Parquet
shards). Reserve roughly 350 GB free for the shards, JSONL corpus, and merged
index:

~~~bash
hf download DrewZhang/inscit-passages-index --repo-type dataset \
  --local-dir collection/inscit --max-workers 4

python scripts/build_inscit_corpus.py \
  --collection-dir collection/inscit

cat collection/inscit/e5_Flat.index.part_* \
  > collection/inscit/e5_Flat.index

test -f collection/inscit/inscit_index.jsonl
test -f collection/inscit/e5_Flat.index
~~~

QReCC uses `DrewZhang/qrecc-passages-index`. Its downloader is resumable and
intentionally uses a separate Conda environment so it cannot modify the
training environment. It enforces a 500-GB free-space check:

~~~bash
QRECC_DOWNLOAD_ENV=hf-download HF_MAX_WORKERS=8 \
  bash scripts/download_qrecc_index.sh

python scripts/build_qrecc_corpus.py \
  --collection-dir collection/qrecc

cat collection/qrecc/e5_Flat.index.part_* \
  > collection/qrecc/e5_Flat.index

test -f collection/qrecc/qrecc_index.jsonl
test -f collection/qrecc/e5_Flat.index
~~~

#### GPU FAISS installation and verification

The QReCC GPU index needs a **GPU-enabled** FAISS Python binding. A plain
`faiss-cpu` wheel can load and serve the index on host memory, but it does not
provide `GpuMultipleClonerOptions`; setting `RETRIEVER_FAISS_GPU=true` with
that wheel will fail. The older Conda-forge CUDA 11.8 FAISS 1.8 build can also
expose CPU-only bindings despite its package name. Use the following tested
installation in the activated Python-3.10 `interactivechat-r1` environment.

Run the supplied one-time installer. It removes only existing FAISS packages
and pip wheels, then installs the known-good PyTorch-channel CUDA 12.1 FAISS
binding. If Conda's displayed package plan proposes removing PyTorch, vLLM,
FlashAttention, or the environment itself, cancel and resolve that conflict
instead.

~~~bash
conda activate interactivechat-r1
bash scripts/install_gpu_faiss_cuda121.sh
~~~

Verify the actual Python binding before starting the retriever. This must print
`GPU API: True` and `Visible GPUs: 2`.

~~~bash
CUDA_VISIBLE_DEVICES=0,1 python - <<'PY'
import faiss

print("Faiss:", getattr(faiss, "__version__", "unknown"))
print("GPU API:", hasattr(faiss, "GpuMultipleClonerOptions"))
print("Visible GPUs:", faiss.get_num_gpus())
assert hasattr(faiss, "GpuMultipleClonerOptions")
assert faiss.get_num_gpus() == 2
PY
~~~

The NVIDIA driver must support CUDA 12.1 or newer. `RETRIEVER_FAISS_GPU=false`
is a supported CPU-index fallback after this installation; the E5 query encoder
can still run on GPU, but QReCC throughput will be lower.

The examples below use CUDA devices 0 and 1. These are per-process
assignments: retriever, simulator, and training memory use is additive. For a
GPU-resident QReCC index, use a separate pair of GPUs for 7B training when
available; otherwise use CPU FAISS or ensure the two cards have sufficient
headroom for both the index and the training workers.

~~~bash
CUDA_VISIBLE_DEVICES=0,1 \
RETRIEVER_FAISS_GPU=true \
RETRIEVER_INDEX_PATH=$PWD/collection/qrecc/e5_Flat.index \
RETRIEVER_CORPUS_PATH=$PWD/collection/qrecc/qrecc_index.jsonl \
RETRIEVER_MODEL_PATH=intfloat/e5-base-v2 \
INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
bash scripts/run_local_retriever_server.sh > logs/qrecc_retriever.log 2>&1 &
~~~

For InsCiT, use the same command but replace both `qrecc` paths with `inscit`.
With GPU FAISS enabled, expose at least two retrieval GPUs; the server shards
the index over the visible GPUs. If the cards cannot accommodate the index,
set `RETRIEVER_FAISS_GPU=false` and expect lower throughput.

For **TurnPPO**, the Qwen-32B simulator and training launchers use the same
CUDA `0,1` convention. Do not assume the simulator, GPU-FAISS retrieval, and
training can coexist on one physical pair: their memory footprints are
additive. When they share a host, start the retriever with
`RETRIEVER_FAISS_GPU=false` (the query encoder still sees CUDA 0 and 1), and
place the simulator behind the configured API endpoint if local memory is
insufficient.

### Dynamic dialogue Parquets (required by TurnPPO only)

**Yes—this conversion belongs in the setup instructions.** ConvAgent and
ChatR1 consume the static Parquets directly and never need this step. TurnPPO
instead replays complete dialogues against the user simulator, so it needs
one-dialogue-per-row dynamic Parquets. The conversion keeps the complete raw
conversation, canonical source labels, and an audit file; it does not call a
model or retrieve passages.

For the released InsCiT TurnPPO workflow, place the original full-dialogue
JSON files at `data/raw/inscit_train.json` and `data/raw/inscit_test.json`,
then run:

~~~bash
python scripts/prepare_simulated_user_inscit.py \
  --input data/raw/inscit_train.json \
  --output data/sim_user_inscit_train.parquet \
  --split train

python scripts/prepare_simulated_user_inscit.py \
  --input data/raw/inscit_test.json \
  --output data/sim_user_inscit_test.parquet \
  --split test
~~~

The TurnPPO launcher then deterministically creates its own dialogue-disjoint
10% train/monitor split (seed 42, first source user question), matching the
baseline monitor split. If dynamic Parquets already exist, do not reconvert
them unless the original raw dialogues changed.


Start the dataset-matched retriever before a **static baseline** run. InsCiT example:

~~~bash
CUDA_VISIBLE_DEVICES=0,1 \
RETRIEVER_FAISS_GPU=true \
RETRIEVER_INDEX_PATH=$PWD/collection/inscit/e5_Flat.index \
RETRIEVER_CORPUS_PATH=$PWD/collection/inscit/inscit_index.jsonl \
RETRIEVER_MODEL_PATH=intfloat/e5-base-v2 \
INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
bash scripts/run_local_retriever_server.sh > logs/inscit_retriever.log 2>&1 &
~~~

## Latest static baseline suites

The complete suites are serial: **ConvAgent 3B → ConvAgent 7B → ChatR1 3B → ChatR1 7B**. The QReCC launchers also support **ConvAgent 3B → ConvAgent 7B**, and the 7B-only pair **ConvAgent 7B → ChatR1 7B**. For every enabled stage they train, source-test evaluate the selected checkpoint, export only the selected actor, then upload it if `HF_UPLOAD_ACTOR_ONLY=true`.

### InsCiT train/test

~~~bash
nohup env \
  CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
  MODEL_3B_PATH=$PWD/models/Qwen2.5-3B-Instruct \
  MODEL_7B_PATH=$PWD/models/Qwen2.5-7B-Instruct \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  WANDB_ENABLED=true WANDB_PROJECT=interactivechat-r1 \
  WANDB_RUN_GROUP=latest_inscit_baselines \
  HF_UPLOAD_ACTOR_ONLY=true HF_UPLOAD_NUM_WORKERS=4 \
  bash scripts/run_latest_inscit_static_baselines_3b_7b_export_upload.sh \
  > logs/latest_inscit_static_baselines.log 2>&1 &
~~~

### QReCC train/test

Start the retriever with the QReCC index/corpus on CUDA 0 and 1. The example
keeps the large FAISS index in host memory so it can share the CUDA numbering
convention with the serial training job:

~~~bash
nohup env \
  CUDA_VISIBLE_DEVICES=0,1 \
  RETRIEVER_FAISS_GPU=false \
  RETRIEVER_INDEX_PATH=$PWD/collection/qrecc/e5_Flat.index \
  RETRIEVER_CORPUS_PATH=$PWD/collection/qrecc/qrecc_index.jsonl \
  RETRIEVER_MODEL_PATH=intfloat/e5-base-v2 \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  bash scripts/run_local_retriever_server.sh \
  > logs/qrecc_retriever.log 2>&1 &
~~~

Then run only ConvAgent 3B -> ConvAgent 7B serially on GPUs 0 and 1:

~~~bash
nohup env \
  CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
  MODEL_3B_PATH=$PWD/models/Qwen2.5-3B-Instruct \
  MODEL_7B_PATH=$PWD/models/Qwen2.5-7B-Instruct \
  HOLDOUT_FRACTION=0.10 SPLIT_SEED=42 \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  WANDB_ENABLED=true WANDB_PROJECT=interactivechat-r1 \
  WANDB_RUN_GROUP=qrecc_convagent_3b_7b_cuda01_v1 \
  WANDB_LOG_VAL_GENERATIONS=0 \
  HF_UPLOAD_ACTOR_ONLY=true HF_UPLOAD_NUM_WORKERS=4 \
  bash scripts/run_latest_qrecc_convagent_3b_7b_cuda01_export_upload.sh \
  > logs/qrecc_convagent_3b_7b_cuda01_v1.log 2>&1 &
~~~

The launcher hard-disables both ChatR1 stages. Set
`RUN_CONVAGENT_3B=false` to resume with only the 7B stage. The existing
`run_latest_qrecc_static_baselines_3b_7b_cuda01_export_upload.sh` remains the
four-stage ConvAgent + ChatR1 entrypoint when that full comparison is needed.

#### QReCC 7B: ConvAgent then ChatR1

This 7B-only launcher does not require the 3B model directory. It runs
**ConvAgent 7B → ChatR1 7B** serially, with a separate 7B actor and critic for
ChatR1:

~~~bash
nohup env \
  CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
  MODEL_7B_PATH=$PWD/models/Qwen2.5-7B-Instruct \
  HOLDOUT_FRACTION=0.10 SPLIT_SEED=42 \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  WANDB_ENABLED=true WANDB_PROJECT=interactivechat-r1 \
  WANDB_RUN_GROUP=qrecc_convagent_chatr1_7b_cuda01_v1 \
  WANDB_LOG_VAL_GENERATIONS=0 \
  HF_UPLOAD_ACTOR_ONLY=true HF_UPLOAD_NUM_WORKERS=4 \
  HF_CONVAGENT_7B_REPO_ID=DrewZhang/interactivechat-r1-static-convagent-qrecc-qwen25-7b \
  HF_CHATR1_7B_REPO_ID=DrewZhang/interactivechat-r1-static-chatr1-qrecc-qwen25-7b \
  bash scripts/run_latest_qrecc_convagent_chatr1_7b_cuda01_export_upload.sh \
  > logs/qrecc_convagent_chatr1_7b_cuda01_v1.log 2>&1 &
~~~

For either QReCC GPU-FAISS launcher, first run the verification above, then
start the GPU retriever with `RETRIEVER_FAISS_GPU=true`. Do not run the 7B
training pair on the same two 80-GB cards as the GPU-sharded QReCC index.

Each method validates every five updates and selects by the equal-weight
max-reference composite `(F1 + BERTScore-F1 + NDCG@3) / 3`. Immediately after
a newly selected best checkpoint is fully saved, the trainer removes its older
real `global_step_*` siblings; therefore, only the current best full
actor/critic/optimizer checkpoint is retained during training. This limits
disk growth, but does not reduce GPU-memory usage. Confirm a cleanup with:

~~~bash
grep -F "[Checkpoint pruning]" logs/qrecc_convagent_3b_7b_cuda01_v1.log
~~~

Default Hub names:

~~~text
DrewZhang/interactivechat-r1-static-convagent-{inscit,qrecc}-qwen25-{3b,7b}
DrewZhang/interactivechat-r1-static-chatr1-{inscit,qrecc}-qwen25-{3b,7b}
~~~

To resume only a later stage:

~~~bash
RUN_CONVAGENT_3B=false RUN_CONVAGENT_7B=false \
RUN_CHATR1_3B=true RUN_CHATR1_7B=true \
DATASET=inscit bash scripts/run_latest_static_baselines_suite.sh
~~~

## TurnPPO InsCiT 3B

Start the 32B simulator on GPUs 0 and 1. The training command below now uses
the same CUDA 0 and 1 convention; run them together only when the physical
allocation has enough memory, otherwise serve the simulator remotely:

~~~bash
CUDA_VISIBLE_DEVICES=0,1 \
SIMULATOR_MODEL_PATH=$PWD/models/Qwen2.5-32B-Instruct \
SIMULATOR_MODEL_NAME=qwen32b-user-simulator \
SIMULATOR_PORT=8010 SIMULATOR_TP_SIZE=2 \
SIMULATOR_GPU_MEMORY_UTILIZATION=0.65 \
SIMULATOR_MAX_MODEL_LEN=8192 SIMULATOR_MAX_NUM_SEQS=1 \
SIMULATOR_MAX_NUM_BATCHED_TOKENS=8192 \
bash scripts/run_user_simulator_server.sh > logs/qwen32b_user_simulator.log 2>&1 &
~~~

When the retriever and simulator must share GPUs 0 and 1, use this
**TurnPPO-only replacement** for the retriever. It keeps the FAISS index in
host RAM and uses GPU 0 only for the small E5 query encoder; do not leave the
GPU-FAISS static-baseline retriever running on port 8002 at the same time.

~~~bash
CUDA_VISIBLE_DEVICES=0,1 \
RETRIEVER_FAISS_GPU=false \
RETRIEVER_INDEX_PATH=$PWD/collection/inscit/e5_Flat.index \
RETRIEVER_CORPUS_PATH=$PWD/collection/inscit/inscit_index.jsonl \
RETRIEVER_MODEL_PATH=intfloat/e5-base-v2 \
INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
bash scripts/run_local_retriever_server.sh \
  > logs/inscit_turnppo_cpu_faiss_retriever.log 2>&1 &
~~~

With the CPU-FAISS InsCiT retriever and simulator ready:

~~~bash
nohup env \
  CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
  MODEL_PATH=$PWD/models/Qwen2.5-3B-Instruct \
  USER_SIMULATOR_BASE_URL=http://127.0.0.1:8010 \
  USER_SIMULATOR_MODEL=qwen32b-user-simulator \
  HOLDOUT_FRACTION=0.10 SPLIT_SEED=42 \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  WANDB_ENABLED=true WANDB_PROJECT=interactivechat-r1 \
  WANDB_RUN_GROUP=turnppo_inscit_3b \
  HF_TURN_PPO_REPO_ID=DrewZhang/interactivechat-r1-turn-ppo-inscit-qwen25-3b \
  HF_UPLOAD_ACTOR_ONLY=true HF_UPLOAD_NUM_WORKERS=4 \
  EXPERIMENT_NAME=turn_ppo_inscit_qwen25_3b_latest \
  bash scripts/run_turn_ppo_inscit_3b_export_upload.sh \
  > logs/turn_ppo_inscit_3b_latest.log 2>&1 &
~~~

TurnPPO retains:

~~~text
eval_log/inscit/<experiment>/metrics_summary.json
eval_log/inscit/<experiment>_test/metrics_summary.json
eval_log/inscit/<experiment>_static_inscit/metrics_summary.json
~~~

## Feedback-GRPO InsCiT 3B

Feedback-GRPO trains **two separate Qwen2.5-3B-Instruct policies** in one
process: the System policy and the learned User-feedback policy. It does not
use the external 32B TurnPPO user-simulator service. Both policies start from
the original 3B weights, have independent optimizers, and are exported to two
separate Hugging Face repositories.

The run needs the original static ConvAgent InsCiT source Parquets and the
dataset-matched InsCiT E5 index/corpus. Download/build the InsCiT assets in
[Passage corpora and E5 FAISS indexes](#passage-corpora-and-e5-faiss-indexes)
if they are not already present, then verify the expected inputs:

~~~bash
test -f data/static_convagent_raw/ConvAgent/inscit/inscit_train.parquet
test -f data/static_convagent_raw/ConvAgent/inscit/inscit_test.parquet
test -f collection/inscit/e5_Flat.index
test -f collection/inscit/inscit_index.jsonl
~~~

Start the **InsCiT retriever on GPUs 0 and 1**. `RETRIEVER_FAISS_GPU=true`
requires the GPU-capable FAISS installation described above; it shards the
index across the two visible GPUs. Do not point this server at a QReCC index.

~~~bash
mkdir -p logs
nohup env \
  CUDA_VISIBLE_DEVICES=0,1 \
  RETRIEVER_FAISS_GPU=true \
  RETRIEVER_INDEX_PATH=$PWD/collection/inscit/e5_Flat.index \
  RETRIEVER_CORPUS_PATH=$PWD/collection/inscit/inscit_index.jsonl \
  RETRIEVER_MODEL_PATH=intfloat/e5-base-v2 \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  bash scripts/run_local_retriever_server.sh \
  > logs/inscit_feedback_grpo_retriever.log 2>&1 &

tail -f logs/inscit_feedback_grpo_retriever.log
~~~

Wait for `Uvicorn running on http://127.0.0.1:8002`; the Feedback-GRPO launcher
also sends a `/retrieve` readiness probe before it starts training.

The following is the **feedback-refinement** experiment: it selects and
early-stops on the second System response after learned User feedback. This is
a two-round result and must not be presented as a direct one-response
comparison with ConvAgent/ChatR1. It uses GPUs 2 and 3 for training, performs
monitor validation every five System updates, tests both direct and second
response views at the selected pair, automatically keeps only that paired
checkpoint, exports both policies, then uploads both when `HF_UPLOAD=true`.

~~~bash
nohup env \
  CUDA_VISIBLE_DEVICES=2,3 \
  N_GPUS=2 \
  ULYSSES_SEQUENCE_PARALLEL_SIZE=1 \
  SYSTEM_MODEL_PATH=$PWD/models/Qwen2.5-3B-Instruct \
  USER_MODEL_PATH=$PWD/models/Qwen2.5-3B-Instruct \
  SELECTION_SETTING=feedback-refinement \
  EXPERIMENT_NAME=feedback_grpo_inscit_qwen25_3b_feedback_second_only_cuda23_v1 \
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
  WANDB_RUN_GROUP=inscit_feedback_grpo_3b_feedback_second_only_cuda23 \
  HF_UPLOAD=true \
  HF_UPLOAD_NUM_WORKERS=4 \
  HF_SYSTEM_REPO_ID=DrewZhang/interactivechat-r1-feedback-grpo-inscit-qwen25-3b-feedback-second-only-cuda23-system \
  HF_USER_REPO_ID=DrewZhang/interactivechat-r1-feedback-grpo-inscit-qwen25-3b-feedback-second-only-cuda23-user \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  bash scripts/run_feedback_grpo_inscit_3b_train_eval_export_upload.sh \
  > logs/feedback_grpo_inscit_3b_feedback_second_only_cuda23_v1.log 2>&1 &
~~~

System training has one retried first response, one frozen-User feedback, and
eight same-prompt revised System samples per source: **9 System episodes**, not
the former `8 + 8 x 8 = 72`. Invalid first responses are retried at most twice;
sources with no legal terminal action are skipped. Invalid revised samples are
filtered and a revised group needs at least two valid candidates. Only revised
tokens have System loss weight 1; the first response has loss weight 0 but is
logged and evaluated.

For a direct one-response experiment, start a **fresh** run with
`SELECTION_SETTING=direct-response`, a distinct `EXPERIMENT_NAME`, W&B group,
and two different HF repository IDs. It selects/early-stops using the first
response; feedback-refinement selects/early-stops using the second response.
Do not resume an old pre-`second-only` Feedback-GRPO experiment into this
recipe.

The selected pair and both evaluation views are stored here:

~~~text
outputs/feedback_grpo/<experiment>/final_checkpoint.txt
exports/feedback_grpo/<experiment>/{system,user}/
eval_log/feedback_grpo/<experiment>/monitor/system_step_*/
  first_response_metrics/metrics_summary.json
  feedback_round_two_metrics/metrics_summary.json
eval_log/feedback_grpo/<experiment>/official_test/
  first_response_metrics/metrics_summary.json
  feedback_round_two_metrics/metrics_summary.json
~~~

See [docs/feedback_grpo.md](docs/feedback_grpo.md) for the reward equations,
group filtering, checkpoint/resume behavior, and detailed W&B diagnostics.

## W&B and actor-only exports

Set WANDB_ENABLED=true. Runs use WANDB_RUN_GROUP for grouping and EXPERIMENT_NAME for run identity:

~~~text
https://wandb.ai/<your-entity>/interactivechat-r1
~~~

Useful panels include actor/pg_loss, actor/entropy_loss, actor/ppo_kl, critic/reward, critic/vf_loss, response length, terminal reward, and val/selection/*. W&B reports after a completed PPO update, so it can lag rollout progress.

Actor-only export removes optimizer, trainer/FSDP state, reference, and critic while preserving the selected actor weights exactly. Inference/static-evaluation performance is unchanged. Full two-rank global_step checkpoints require two ranks; use the exported HF directory for one-GPU inference.
