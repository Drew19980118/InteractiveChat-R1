# InteractiveChat-R1

This repository provides the current protocol-matched comparison among **ConvAgent**, **ChatR1**, and online **TurnPPO**. Older static-baseline entrypoints are retired; use only the latest scripts below.

## Current protocol

| Setting | Value |
|---|---:|
| Policy initialization | Qwen2.5-3B-Instruct or Qwen2.5-7B-Instruct |
| Static baseline rollout count | n=8 |
| TurnPPO rollout count | n=1 |
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

## Setup

~~~bash
MAX_JOBS=8 bash scripts/install_h100_eval_env.sh interactivechat-r1
conda activate interactivechat-r1
python -m pip install -r requirements-eval-metrics.txt
python -m pip install wandb       # optional, for dashboards
wandb login                       # optional, once per server
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

Use GPUs 0 and 1 for retrieval and GPUs 2 and 3 for policy training. This is
the resource layout used by the static ConvAgent/ChatR1 suites (the simulator
is not running for those static baselines):

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

For **TurnPPO**, the Qwen-32B simulator also uses GPUs 0 and 1, while training
stays on GPUs 2 and 3. Do not run GPU-FAISS retrieval and the 32B simulator on
GPUs 0 and 1 at the same time: their memory footprints are additive. If only
these four GPUs are available, start the retriever with
`RETRIEVER_FAISS_GPU=false` (the query encoder still uses the visible GPUs)
or reserve another GPU pair for GPU-FAISS retrieval.

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

Each suite is serial: **ConvAgent 3B → ConvAgent 7B → ChatR1 3B → ChatR1 7B**. For every stage it trains, source-test evaluates the selected checkpoint, exports only the selected actor, then uploads it if HF_UPLOAD_ACTOR_ONLY=true.

### InsCiT train/test

~~~bash
nohup env \
  CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2 ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
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

Restart the retriever with the QReCC index/corpus, then run:

~~~bash
nohup env \
  CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2 ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
  MODEL_3B_PATH=$PWD/models/Qwen2.5-3B-Instruct \
  MODEL_7B_PATH=$PWD/models/Qwen2.5-7B-Instruct \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  WANDB_ENABLED=true WANDB_PROJECT=interactivechat-r1 \
  WANDB_RUN_GROUP=latest_qrecc_baselines \
  HF_UPLOAD_ACTOR_ONLY=true HF_UPLOAD_NUM_WORKERS=4 \
  bash scripts/run_latest_qrecc_static_baselines_3b_7b_export_upload.sh \
  > logs/latest_qrecc_static_baselines.log 2>&1 &
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

Start the 32B simulator on GPUs 0 and 1; training remains on GPUs 2 and 3:

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
  CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2 ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
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

## W&B and actor-only exports

Set WANDB_ENABLED=true. Runs use WANDB_RUN_GROUP for grouping and EXPERIMENT_NAME for run identity:

~~~text
https://wandb.ai/<your-entity>/interactivechat-r1
~~~

Useful panels include actor/pg_loss, actor/entropy_loss, actor/ppo_kl, critic/reward, critic/vf_loss, response length, terminal reward, and val/selection/*. W&B reports after a completed PPO update, so it can lag rollout progress.

Actor-only export removes optimizer, trainer/FSDP state, reference, and critic while preserving the selected actor weights exactly. Inference/static-evaluation performance is unchanged. Full two-rank global_step checkpoints require two ranks; use the exported HF directory for one-GPU inference.

