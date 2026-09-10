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
data/sim_user_inscit_train.parquet
data/sim_user_inscit_test.parquet
collection/{inscit,qrecc}/
~~~

Download the released static Parquets:

~~~bash
hf download DrewZhang/conv --repo-type dataset \
  --include "ConvAgent/**" --local-dir data/static_convagent_raw
hf download DrewZhang/conv --repo-type dataset \
  --include "ChatR1/**" --local-dir data/static_chatr1_raw
~~~

Start the dataset-matched retriever before a run. InsCiT example:

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

Start the 32B simulator on GPUs distinct from the training pair:

~~~bash
CUDA_VISIBLE_DEVICES=4,5 \
SIMULATOR_MODEL_PATH=$PWD/models/Qwen2.5-32B-Instruct \
SIMULATOR_MODEL_NAME=qwen32b-user-simulator \
SIMULATOR_PORT=8010 SIMULATOR_TP_SIZE=2 \
SIMULATOR_GPU_MEMORY_UTILIZATION=0.65 \
SIMULATOR_MAX_MODEL_LEN=8192 SIMULATOR_MAX_NUM_SEQS=1 \
SIMULATOR_MAX_NUM_BATCHED_TOKENS=8192 \
bash scripts/run_user_simulator_server.sh > logs/qwen32b_user_simulator.log 2>&1 &
~~~

With InsCiT retriever and simulator ready:

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

