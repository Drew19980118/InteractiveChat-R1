# InteractiveChat-R1

InteractiveChat-R1 is an online, user-centric reinforcement-learning framework for conversational search.  It turns a static conversational-search benchmark into a dynamic environment in which a policy agent interacts with a frozen LLM user simulator while solving each original turn-level **sub-task**.

The released code is a self-contained research implementation built on the `verl` FSDP/vLLM training runtime.  It contains the implementation needed for data conversion, local retrieval, simulated-user rollout, sparse GRPO training, online validation, and metric calculation.  It does **not** distribute benchmark data, dense indexes, or model weights.

## What is implemented

For every dialogue, the online collector samples `n=8` policy trajectories.  A source dialogue is processed sub-task by sub-task.  The policy may issue up to four tool calls; each tool call has exactly one search query and returns top-3 passages.  A terminal action is one of `answer`, `clarify`, or `nonanswer` when supported by the dataset.

- **InsCiT:** `answer`, `clarify`, and `nonanswer`.
- **TopiOCQA:** `answer` and `nonanswer`.
- **QReCC / CoRAL:** `answer` only; the prompt never exposes `clarify` or `nonanswer`.

For an answer-labelled sub-task, a correct initial `answer` is assessed by the frozen user simulator.  A level-2/3 judgement yields concise feedback and a retry prompt that forces the policy to finish with `answer` (it may retrieve more evidence first).  A wrong action or invalid format triggers a canonical-gold fallback before the next source sub-task.

The canonical reward recipe is **UCI + interactive task/user channels**:

\[
r = r_{\mathrm{action}} + r_{\mathrm{UCI}} + r_{\mathrm{clarity}}
    + r_{\mathrm{patience}} + r_{\mathrm{format}} + r_{\mathrm{clarify\text{-}F1}}.
\]

Each channel is independently normalized among rollouts that reached the same `(dialogue, subtask, response-depth)` group, then summed.  The terminal user-facing reward is propagated backwards only through policy actions in that same source sub-task, so prior tool-query actions receive credit without leaking reward into the next original turn.  The canonical configuration deliberately disables answer-stripped evidence utility and search-efficiency shaping.

## Repository layout

```text
InteractiveChat-R1/
├── scrl/                         # online simulated-user environment and prompts
├── verl/                         # FSDP/vLLM GRPO backend (modified for this project)
├── scripts/
│   ├── prepare_simulated_user_*.py       # static JSON -> dynamic Parquet adapters
│   ├── run_local_retriever_server.sh     # FAISS /retrieve service
│   ├── run_user_simulator_server.sh      # frozen Qwen32B OpenAI-compatible server
│   ├── run_{inscit,qrecc}_{3b,7b}_uci_train.sh
│   ├── run_{inscit,qrecc}_{3b,7b}_uci_val.sh
│   ├── run_topiocqa_from_inscit_*_uci_val.sh
│   └── run_coral_from_qrecc_*_uci_val.sh
├── data/                         # create locally; raw JSON and converted Parquet
├── models/                       # create locally; Qwen checkpoints
├── collection/                   # create locally; corpus and FAISS index
├── outputs/                      # generated FSDP checkpoints
└── eval_log/                     # generated online trajectories and metric summaries
```

`data/`, `models/`, `collection/`, `outputs/`, `eval_log/`, cache, and logs are intentionally ignored by Git.

## 1. Hardware and software

The supplied environment lock is designed for a clean **Linux x86_64, Python 3.10, NVIDIA H100/H200** installation with CUDA 12.1 wheels, PyTorch 2.4.0, vLLM 0.6.3, Ray 2.10.0, Transformers 4.49.0, and FlashAttention 2.5.8.  An NVIDIA driver at least `525.60.13` is required.

The recommended single-node layout has eight H100 80GB GPUs:

| Role | Recommended GPUs | Notes |
|---|---:|---|
| Local FAISS retriever | `0,1` | May use one GPU if the index fits. |
| Frozen Qwen2.5-32B user simulator | `2,3` | Tensor parallel size 2. |
| Policy / reference / critic / rollout | `4,5,6,7` | FSDP, global train context batch 128. |

Two policy GPUs are supported by setting `N_GPUS=2` and `ULYSSES_SEQUENCE_PARALLEL_SIZE=2`; keep the same global batch and reduce only `ROLLOUT_GPU_MEMORY_UTILIZATION` if necessary.  The 7B launchers use a conservative `0.04` vLLM cache fraction by default.  This does not change the 8192-token model context window or the optimization recipe.

### Create the locked environment from scratch

The commands below are the supported public-repository path.  They assume no
previous IGPO checkout, Conda environment, model directory, or retriever
service.  Do not substitute an old project's environment in these instructions.

```bash
git clone https://github.com/Drew19980118/InteractiveChat-R1.git InteractiveChat-R1
cd InteractiveChat-R1

MAX_JOBS=8 bash scripts/install_h100_eval_env.sh interactivechat-r1
conda activate interactivechat-r1
python -m pip install -r requirements-eval-metrics.txt
python -m pip install -r requirements-retriever.txt
python -m pip install -r requirements-dev.txt
python -m pip check
```

Do **not** install the broad `requirements.txt` into this environment: it can replace the locked PyTorch/vLLM runtime.  `flash-attn` is compiled by the installer, so set `MAX_JOBS` if the login node has a restrictive CPU quota.  The `faiss-gpu-cu12` package in `requirements-retriever.txt` is intentionally separate because the exact FAISS wheel must be compatible with the host driver/CUDA stack.  W&B logging and the legacy web-agent browser are optional and can be installed, if needed, with `python -m pip install -e '.[tracking,web-agent]'`.

Quick import test:

```bash
python -m pytest -q tests/trainer/ppo/test_simulated_user_sparse_grpo.py
```

## 2. Download the policy and simulator models

The default scripts assume the following local layout; all paths can be overridden with environment variables.

```bash
mkdir -p models

hf download Qwen/Qwen2.5-3B-Instruct  --local-dir models/Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct  --local-dir models/Qwen2.5-7B-Instruct
hf download Qwen/Qwen2.5-32B-Instruct --local-dir models/Qwen2.5-32B-Instruct
```

The 32B checkpoint is frozen and used only as the user simulator.  The 3B or 7B checkpoint is both the actor initialization and the reference/critic model path required by the FSDP runner.

## 3. Convert static benchmarks into dynamic environments

Place the original, unmodified source JSON files as follows:

```text
data/raw/
├── inscit_train.json      ├── inscit_test.json
├── qrecc_train.json       ├── qrecc_test.json
├── coral_train.json       ├── coral_test.json
└── topiocqa_train.json    └── topiocqa_test.json
```

Then run:

```bash
bash scripts/prepare_all_dynamic_benchmarks.sh
```

This produces one Parquet row per complete source dialogue, for example `data/sim_user_inscit_train.parquet`.  The original dialogue, source context, answer, action label, and gold passage information are retained in `reward_model.simulated_dialogue`; no online data are generated in this conversion step.

Dataset-specific handling is fixed and audited:

- **InsCiT:** `directAnswer` and `noAnswerButRelevantInfo` map to `answer`; `clarification` maps to `clarify`; `noAnswerNoRelevantInfo` maps to `nonanswer`.  The script writes a canonical-label audit JSONL for any next-context mismatch.
- **TopiOCQA:** literal `UNANSWERABLE` maps to `nonanswer`; gold passage text is retained because source IDs and local-retriever IDs may differ.
- **QReCC:** grouped by `Conversation_no`; only `answer` actions; turns with empty `Truth_answer` are skipped and auditable.
- **CoRAL:** grouped by `conv_id`; only `answer` actions; turns with an empty response are skipped and auditable.

The converter names each audit file `<output-stem>.canonical_audit.jsonl`.  Review it before reporting results.

## 4. Set up the local retriever

All rollout scripts require a local HTTP service at `http://127.0.0.1:8002/retrieve`.  Its request contract is:

```json
{"queries": ["one query"], "topk": 3, "return_scores": true}
```

and each result must expose a `document` with `passage_id`, `passage_text`, and optional `title`.  The project includes a compatible FAISS server at `scripts/local_retriever.py`.

### Prepare a QReCC FAISS collection (optional helper)

The original QReCC corpus/index is large (about 210GB); it is not committed.  The helper downloads it outside the training environment:

```bash
tmux new -s qrecc-download
bash scripts/download_qrecc_index.sh
```

The retriever needs a **single** FAISS file plus a JSONL corpus aligned with its row IDs.  After downloading, create the corpus with the existing converter and concatenate index shards only if the source archive stores them as binary pieces:

```bash
# Creates collection/qrecc/qrecc_index.jsonl from part_*.parquet without
# downloading anything again.
python scripts/build_qrecc_corpus.py --collection-dir collection/qrecc

# If the downloaded repository contains e5_Flat.index.part_{aa,ab,ac,ad},
# assemble its binary index once.
cat collection/qrecc/e5_Flat.index.part_aa \
    collection/qrecc/e5_Flat.index.part_ab \
    collection/qrecc/e5_Flat.index.part_ac \
    collection/qrecc/e5_Flat.index.part_ad \
    > collection/qrecc/e5_Flat.index
```

For InsCiT, TopiOCQA, and CoRAL, point the server to the **same retrieval collection used by your experiment**.  The important requirement is that the corpus covers the benchmark’s evidence passages and the FAISS index and JSONL corpus have identical row order.  TopiOCQA NDCG@3 uses normalized passage-text matching rather than raw passage IDs to accommodate differing source/retriever ID namespaces.

Launch and test the service:

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0,1 \
RETRIEVER_INDEX_PATH=$PWD/collection/qrecc/e5_Flat.index \
RETRIEVER_CORPUS_PATH=$PWD/collection/qrecc/qrecc_index.jsonl \
RETRIEVER_MODEL_PATH=intfloat/e5-base-v2 \
bash scripts/run_local_retriever_server.sh \
  > logs/retriever.log 2>&1 &

python scripts/test_local_retriever.py --topk 3 \
  --query "What is the capital of France?"
```

The readiness check in every training/validation launcher deliberately uses `return_scores=true`; do not change the retriever to return a different response shape.

## 5. Launch the frozen user simulator

The policy’s private thought/tool/evidence trace is never sent to the simulator.  The simulator receives only the current sub-task, its permitted source context, and the user-visible `<answer>` content; it returns structured level 1/2/3 feedback.

```bash
mkdir -p logs
nohup env \
  CUDA_VISIBLE_DEVICES=2,3 \
  SIMULATOR_GPU_MEMORY_UTILIZATION=0.45 \
  SIMULATOR_MAX_MODEL_LEN=8192 \
  SIMULATOR_MAX_NUM_SEQS=1 \
  SIMULATOR_MAX_NUM_BATCHED_TOKENS=8192 \
  SIMULATOR_MODEL_PATH=$PWD/models/Qwen2.5-32B-Instruct \
  SIMULATOR_MODEL_NAME=qwen32b-user-simulator \
  SIMULATOR_TP_SIZE=2 \
  bash scripts/run_user_simulator_server.sh \
  > logs/qwen32b_user_simulator.log 2>&1 &

curl http://127.0.0.1:8010/v1/models
```

The expected model ID is `qwen32b-user-simulator`.  If port 8010 is occupied, either stop the process that owns that port or set `SIMULATOR_PORT=<free-port>` and use the same port in `USER_SIMULATOR_BASE_URL` below.  Never stop the retriever on port 8002 when fixing the simulator.

## 6. Canonical training configuration

All provided UCI training launchers use exactly the current formal setting:

| Setting | InsCiT | QReCC |
|---|---:|---:|
| Policy model | Qwen2.5-3B or Qwen2.5-7B Instruct | Qwen2.5-3B or Qwen2.5-7B Instruct |
| Global context batch | 128 | 128 |
| PPO mini-batch | 64 | 64 |
| GRPO siblings per dialogue | 8 | 8 |
| Epochs | 3 | 3 |
| Training steps | 15 | 30 |
| Checkpoint frequency | every 5 steps | every 5 steps |
| Final validation batch size | 256 | 256 |
| Search | 1 query/tool call, top-3 passages | 1 query/tool call, top-3 passages |
| Tool-call / answer-depth limit | 4 / 3 | 4 / 3 |
| Model context / max response | 8192 / 500 | 8192 / 500 |
| Reward | action + UCI + clarity + patience + format + clarify-F1 | action + UCI + clarity + patience + format |

`EXACT_CONTEXT_BATCH=true` packs complete dialogues so that exactly 128 original sub-task contexts contribute to each update, without splitting a dialogue when calculating sibling advantages.  The collected trajectory count is therefore `128 × 8 = 1024` rollout rows per update.

### InsCiT 3B: train and final online validation

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
N_GPUS=4 ULYSSES_SEQUENCE_PARALLEL_SIZE=4 \
USER_SIMULATOR_BASE_URL=http://127.0.0.1:8010 \
USER_SIMULATOR_MODEL=qwen32b-user-simulator \
INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
bash scripts/run_inscit_3b_uci_train.sh
```

### InsCiT 7B

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
N_GPUS=4 ULYSSES_SEQUENCE_PARALLEL_SIZE=4 \
USER_SIMULATOR_BASE_URL=http://127.0.0.1:8010 \
USER_SIMULATOR_MODEL=qwen32b-user-simulator \
INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
bash scripts/run_inscit_7b_uci_train.sh
```

### QReCC 3B / 7B

Replace the final script name with the required model scale:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
N_GPUS=4 ULYSSES_SEQUENCE_PARALLEL_SIZE=4 \
USER_SIMULATOR_BASE_URL=http://127.0.0.1:8010 \
USER_SIMULATOR_MODEL=qwen32b-user-simulator \
INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
bash scripts/run_qrecc_3b_uci_train.sh

# Qwen2.5-7B:
# bash scripts/run_qrecc_7b_uci_train.sh
```

Each run automatically performs the full online test after the final step and then computes F1, BERTScore, NDCG@3, action accuracy (where applicable), format success rate, user satisfaction, simulator fallback rate, mean retry depth, and mean tool calls.

For a controlled UCI ablation that preserves all other interactive reward channels but replaces UCI with token-set answer F1:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 N_GPUS=4 ULYSSES_SEQUENCE_PARALLEL_SIZE=4 \
USER_SIMULATOR_BASE_URL=http://127.0.0.1:8010 \
USER_SIMULATOR_MODEL=qwen32b-user-simulator \
bash scripts/run_inscit_3b_uci_to_f1_train.sh
```

For the static-context/no-feedback ablation, every sub-task starts from the
source dataset's gold dialogue prefix; no simulator feedback or retry is used.
It retains action, UCI, format, and clarification-F1 rewards:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 N_GPUS=4 ULYSSES_SEQUENCE_PARALLEL_SIZE=4 \
bash scripts/run_inscit_3b_static_uci_no_feedback_train.sh
```

`run_inscit_3b_ablation_suite.sh` runs these two ablations serially.  Keep the
retriever running for both runs and the user simulator running for the first.

### Resume an interrupted run

The checkpoint `global_step_9` means the completed updates are 0–9.  Resume to a total of 15 steps as follows:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 N_GPUS=4 ULYSSES_SEQUENCE_PARALLEL_SIZE=4 \
USER_SIMULATOR_BASE_URL=http://127.0.0.1:8010 \
USER_SIMULATOR_MODEL=qwen32b-user-simulator \
RESUME_MODE=resume_path \
RESUME_FROM_PATH=$PWD/outputs/inscit/interactivechat_r1_inscit_3b_uci_15steps/global_step_9 \
TOTAL_STEPS=15 \
bash scripts/run_inscit_3b_uci_train.sh
```

## 7. Online validation / transfer evaluation

All validation launchers run the same interactive environment: the simulator can give feedback, policy history contains its own previous answers or canonical fallback responses, and top-3 retrieval happens at every valid tool call.  They write complete per-subtask online traces to `eval_log/`.

Use the same running retriever and user simulator.  For example, evaluate an InsCiT-trained 3B actor on TopiOCQA:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
N_GPUS=4 ULYSSES_SEQUENCE_PARALLEL_SIZE=4 \
USER_SIMULATOR_BASE_URL=http://127.0.0.1:8010 \
USER_SIMULATOR_MODEL=qwen32b-user-simulator \
CHECKPOINT_PATH=$PWD/outputs/inscit/interactivechat_r1_inscit_3b_uci_15steps/global_step_14 \
bash scripts/run_topiocqa_from_inscit_3b_uci_val.sh
```

Available launchers:

| Source checkpoint | In-domain online test | Transfer online test |
|---|---|---|
| InsCiT 3B | `run_inscit_3b_uci_val.sh` | `run_topiocqa_from_inscit_3b_uci_val.sh` |
| InsCiT 7B | `run_inscit_7b_uci_val.sh` | `run_topiocqa_from_inscit_7b_uci_val.sh` |
| QReCC 3B | `run_qrecc_3b_uci_val.sh` | `run_coral_from_qrecc_3b_uci_val.sh` |
| QReCC 7B | `run_qrecc_7b_uci_val.sh` | `run_coral_from_qrecc_7b_uci_val.sh` |

## 8. Outputs and metrics

For experiment `<name>` and data source `<dataset>`:

```text
outputs/<dataset>/<name>/global_step_{4,9,14,...}/  # FSDP checkpoints
outputs/<dataset>/<name>/rollout/                   # complete online training traces
eval_log/<dataset>/<name>/<step>.jsonl              # complete online validation traces
eval_log/<dataset>/<name>/metrics_per_sample.jsonl
eval_log/<dataset>/<name>/metrics_summary.json
```

Metric semantics:

- **F1 / BERTScore / NDCG@3:** averaged only across answer-labelled sub-tasks with a nonempty gold answer.  Answer text comes from the terminal valid `<answer>`; otherwise the latest valid `<answer>` in the same sub-task is used; no valid answer scores zero.
- **NDCG@3:** retrieval after every valid tool call is logged, and the final per-subtask top-3 is evaluated.  QReCC/CoRAL/InsCiT use passage IDs; TopiOCQA uses normalized passage text because its source and retrieval IDs differ.
- **Action accuracy:** exact expected terminal action among action-labelled data (`answer`, and optionally `clarify`/`nonanswer`).
- **Format success rate:** fraction of sub-tasks whose terminal event passes the strict XML parser.
- **User satisfaction:** formerly `level_1_rate`; among answer-labelled sub-tasks, fraction whose final simulator judgement is level 1 (the user is satisfied).
- **Mean retry depth / mean tool calls:** average number of answer attempts and valid executed searches per source sub-task.

To recompute a metric summary for any existing trace:

```bash
python scripts/compute_convagent_eval_metrics.py \
  --input eval_log/inscit/<name>/14.jsonl \
  --output-dir eval_log/inscit/<name> \
  --bert-score-device cuda \
  --bert-score-batch-size 64
```

## Troubleshooting

- **`/retrieve` returns 500:** verify the body includes `return_scores: true`; confirm the corpus rows and FAISS IDs are aligned; then run `scripts/test_local_retriever.py`.
- **Simulator cannot start / port 8010 in use:** use `curl http://127.0.0.1:8010/v1/models` to identify a healthy existing server.  Do not kill the retriever on 8002.
- **FSDP checkpoint shape mismatch:** `MODEL_PATH` must match the checkpoint scale exactly (3B checkpoint with Qwen2.5-3B, 7B checkpoint with Qwen2.5-7B); use a `global_step_*` directory as `CHECKPOINT_PATH`.
- **CUDA OOM:** do not reduce the 8192 model context.  First reserve clean policy GPUs, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and lower only `ROLLOUT_GPU_MEMORY_UTILIZATION` (e.g. `0.04`).
- **`set: pipefail: invalid option name`:** the shell file has Windows CRLF endings.  On the Linux server run `sed -i 's/\r$//' scripts/<script>.sh` once.

## License and attribution

The FSDP/vLLM backend derives from the Apache-2.0 `verl` runtime.  Its original license is retained in [`LICENSE`](LICENSE).  Please preserve this attribution and separately comply with the licenses and terms of the benchmark datasets, dense retrieval collection, and Qwen checkpoints.
