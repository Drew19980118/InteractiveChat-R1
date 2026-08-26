# InteractiveChat-R1

> A user-centric reinforcement-learning framework for online conversational search.

InteractiveChat-R1 trains retrieval-augmented conversational agents in an
environment where each system response changes what a user sees and may change
what the user asks next.  It turns an annotated dialogue into a sequence of
interactive turn-level **sub-tasks**, and optimizes both answer utility and the
quality of the user interaction.

The repository provides an end-to-end implementation built on the `verl`
FSDP/vLLM runtime: dynamic benchmark loading, local dense retrieval,
simulated-user rollout, sparse GRPO training, online validation, and metric
calculation.  Ready-to-run dynamic benchmark Parquet files are released in
[`DrewZhang/conv`](https://huggingface.co/datasets/DrewZhang/conv); model
weights and passage collections are downloaded separately.

## Method overview

1. **Online environment.** A source dialogue is processed sub-task by
   sub-task.  The collector samples `n=8` policy trajectories per dialogue.
2. **Retrieval-augmented interaction.** Each policy tool call emits exactly one
   query and retrieves the top-3 passages.  A sub-task permits at most four
   tool calls.
3. **User feedback.** For answer-labelled sub-tasks, a frozen user simulator
   scores the user-visible answer.  Level-2/3 feedback launches a repair turn;
   the repair prompt requires termination with `answer`.  An invalid response
   or action mismatch falls back to the canonical response before the next
   source turn.
4. **Sparse GRPO.** Reward channels are normalized among sibling rollouts at
   the same `(dialogue, sub-task, response-depth)`, then returned only through
   actions in that original sub-task.

Terminal actions are dataset-specific:

| Dataset | Supported terminal actions |
|---|---|
| InsCiT | `answer`, `clarify`, `nonanswer` |
| TopiOCQA | `answer`, `nonanswer` |
| QReCC / CoRAL | `answer` |

The canonical objective combines answer correctness with interactive
user-satisfaction channels:

\[
r = r_{\mathrm{action}} + r_{\mathrm{answer\text{-}F1}} + r_{\mathrm{clarity}}
    + r_{\mathrm{patience}} + r_{\mathrm{format}} + r_{\mathrm{clarify\text{-}F1}}.
\]

For a valid `answer`, token-set answer F1 is the reference-answer correctness
proxy.  The frozen user simulator independently evaluates clarity and remaining
repair burden: level 1 rewards task resolution, while levels 2/3 incur clarity
and patience penalties and trigger answer-only retries. Each channel is
normalized independently within its active sibling group. The terminal
user-facing return is propagated only through policy actions in the same source
sub-task, so retrieval queries receive credit without reward crossing into a
future user turn.

The canonical configuration disables frozen-model likelihood rewards,
answer-stripped evidence-utility, and search-efficiency shaping.

## Codebase

```text
InteractiveChat-R1/
├── scrl/                         # online simulated-user environment and prompts
├── verl/                         # FSDP/vLLM GRPO backend (modified for this project)
├── scripts/
│   ├── prepare_simulated_user_*.py       # static JSON -> dynamic Parquet adapters
│   ├── run_local_retriever_server.sh     # FAISS /retrieve service
│   ├── run_user_simulator_server.sh      # frozen Qwen32B OpenAI-compatible server
│   ├── run_{inscit,qrecc}_{3b,7b}_full_train.sh
│   ├── run_{inscit,qrecc}_{3b,7b}_full_val.sh
│   ├── run_topiocqa_from_inscit_*_full_val.sh
│   └── run_coral_from_qrecc_*_full_val.sh
├── data/                         # create locally; downloaded dynamic benchmark Parquet
├── models/                       # create locally; Qwen checkpoints
├── collection/                   # create locally; corpus and FAISS index
├── outputs/                      # generated FSDP checkpoints
└── eval_log/                     # generated online trajectories and metric summaries
```

`data/`, `models/`, `collection/`, `outputs/`, `eval_log/`, cache, and logs are intentionally ignored by Git.

## Reproducing InteractiveChat-R1

### 1. Hardware and software

The supplied environment lock is designed for a clean **Linux x86_64, Python 3.10, NVIDIA H100/H200** installation with CUDA 12.1 wheels, PyTorch 2.4.0, vLLM 0.6.3, Ray 2.10.0, Transformers 4.49.0, and FlashAttention 2.5.8.  An NVIDIA driver at least `525.60.13` is required.

The reference single-node layout has four H200 141GB GPUs:

| Role | Recommended GPUs | Notes |
|---|---:|---|
| Local FAISS retriever | `0,1` | Co-located with the simulator; may use one GPU if the index fits. |
| Frozen Qwen2.5-32B user simulator | `0,1` | Tensor parallel size 2. |
| Policy / reference / critic / rollout | `2,3` | FSDP, global train context batch 128. |

Two policy GPUs are supported by setting `N_GPUS=2` and `ULYSSES_SEQUENCE_PARALLEL_SIZE=2`; keep the same global batch and reduce only `ROLLOUT_GPU_MEMORY_UTILIZATION` if necessary.  The 7B launchers use a conservative `0.04` vLLM cache fraction by default.  This does not change the 8192-token model context window or the optimization recipe.

Thus, keep GPUs `0,1` for the retriever and simulator, and use GPUs `2,3`
for policy training and validation:

```bash
export CUDA_VISIBLE_DEVICES=2,3
export N_GPUS=2
export ULYSSES_SEQUENCE_PARALLEL_SIZE=2
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.08
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Run any training or validation launcher below in this shell.
```

### 2. Create the locked environment

The commands below create a clean, reproducible installation with its own
Conda environment, model directory, and local services.

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

Do **not** install the broad `requirements.txt` into this environment: it can replace the locked PyTorch/vLLM runtime.  `flash-attn` is compiled by the installer, so set `MAX_JOBS` if the login node has a restrictive CPU quota.  The `faiss-gpu-cu12==1.9.0.0` package in `requirements-retriever.txt` is intentionally separate because the exact FAISS wheel must be compatible with the host driver/CUDA stack; this release supports Hopper and pins NumPy below 2, which is required by vLLM 0.6.3.  Optional tracking and browser-agent dependencies can be installed with `python -m pip install -e '.[tracking,web-agent]'`.

Quick import test:

```bash
python -m pytest -q tests/trainer/ppo/test_simulated_user_sparse_grpo.py
```

### 3. Download policy and simulator checkpoints

The default scripts assume the following local layout; all paths can be overridden with environment variables.

```bash
mkdir -p models

hf download Qwen/Qwen2.5-3B-Instruct  --local-dir models/Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct  --local-dir models/Qwen2.5-7B-Instruct
hf download Qwen/Qwen2.5-32B-Instruct --local-dir models/Qwen2.5-32B-Instruct
```

The 32B checkpoint is frozen and used only as the user simulator.  The 3B or 7B checkpoint is both the actor initialization and the reference/critic model path required by the FSDP runner.

### 4. Download prepared dynamic benchmarks

The recommended path is to download the exact prepared dynamic benchmarks used
by the provided launchers.  They are public in
[`DrewZhang/conv/InteractiveChat-R1`](https://huggingface.co/datasets/DrewZhang/conv/tree/main/InteractiveChat-R1).
This avoids rebuilding labels from raw sources and guarantees that all
canonical-label decisions match the released experiments.

```bash
# No login is required for this public dataset repository.
mkdir -p data/_hf

# Use separate commands for compatibility with both recent and older hf CLIs.
hf download DrewZhang/conv \
  --repo-type dataset \
  --include "InteractiveChat-R1/sim_user_*.parquet" \
  --local-dir data/_hf

hf download DrewZhang/conv \
  --repo-type dataset \
  --include "InteractiveChat-R1/sim_user_*.canonical_audit.jsonl" \
  --local-dir data/_hf

# The Hub preserves the repository directory under --local-dir; flatten only
# the released dynamic data files into the project data directory.
mv data/_hf/InteractiveChat-R1/sim_user_* data/

ls -lh data/sim_user_*.parquet
```

The launchers directly consume the eight files below:

```text
data/
├── sim_user_inscit_train.parquet      ├── sim_user_inscit_test.parquet
├── sim_user_qrecc_train.parquet       ├── sim_user_qrecc_test.parquet
├── sim_user_coral_train.parquet       ├── sim_user_coral_test.parquet
├── sim_user_topiocqa_train.parquet    └── sim_user_topiocqa_test.parquet
```

The accompanying `*.canonical_audit.jsonl` files are included for provenance
and debugging only; training and validation do not read them.  The temporary
`data/_hf/` directory may be retained as a download cache or removed after
verifying that the eight Parquet files are present.

### 5. Set up the local retriever

All rollout scripts require a local HTTP service at `http://127.0.0.1:8002/retrieve`.  Its request contract is:

```json
{"queries": ["one query"], "topk": 3, "return_scores": true}
```

and each result must expose a `document` with `passage_id`, `passage_text`, and optional `title`.  The project includes a compatible FAISS server at `scripts/local_retriever.py`.

#### Download passage collections and FAISS indexes

The dynamic benchmark Parquets contain labels and dialogue state, but **not**
the retrieval corpus or dense index.  Before any online rollout, download the
collection that matches the experiment.  Every released collection contains:

- `part_*.parquet`: passage shards, in the exact row order used to build the
  index;
- `e5_Flat.index.part_*`: binary chunks of one FAISS index.

The local retriever requires the reassembled index and a JSONL corpus created
from the same, complete set of passage shards.  Do not mix files across
repositories or reorder the shards.

#### InsCiT

Download the complete released [InsCiT passage collection and index](https://huggingface.co/datasets/DrewZhang/inscit-passages-index).
It is approximately 159 GB before the generated JSONL corpus, so reserve at
least 250 GB of free local storage.

```bash
mkdir -p collection/inscit

# Use one command per pattern so older hf CLIs do not discard an earlier
# --include option.  The first command downloads every InsCiT passage shard;
# the second downloads every FAISS binary chunk (part_aa through part_an).
hf download DrewZhang/inscit-passages-index \
  --repo-type dataset \
  --include "part_*.parquet" \
  --local-dir collection/inscit \
  --max-workers 8

hf download DrewZhang/inscit-passages-index \
  --repo-type dataset \
  --include "e5_Flat.index.part_*" \
  --local-dir collection/inscit \
  --max-workers 8

# The existing converter is row-order preserving.  Give an explicit output
# name because the same utility is also used for QReCC.
python scripts/build_qrecc_corpus.py \
  --collection-dir collection/inscit \
  --output collection/inscit/inscit_index.jsonl

# Concatenate chunks in lexical order: aa, ab, ..., an.
cat collection/inscit/e5_Flat.index.part_* > collection/inscit/e5_Flat.index
```

#### QReCC

Download the complete released [QReCC passage collection and index](https://huggingface.co/datasets/DrewZhang/qrecc-passages-index).
It contains exactly `part_001.parquet` through `part_050.parquet`, together
with `e5_Flat.index.part_aa` through `e5_Flat.index.part_ad` (about 210 GB in
total before the generated JSONL).  The supplied resumable helper uses an
isolated download environment and checks the expected 50 passage shards and
four index shards:

```bash
tmux new -s qrecc-download
cd /path/to/InteractiveChat-R1
bash scripts/download_qrecc_index.sh
```

After the helper completes, build the aligned corpus and reassemble the
index:

```bash
python scripts/build_qrecc_corpus.py \
  --collection-dir collection/qrecc \
  --output collection/qrecc/qrecc_index.jsonl

cat collection/qrecc/e5_Flat.index.part_* > collection/qrecc/e5_Flat.index
```

#### TopiOCQA and CoRAL collections

The released collections below follow the same procedure as QReCC: download
all passage and index chunks, build a JSONL corpus in the original shard
order, then concatenate the binary FAISS chunks.  Do not use a corpus from one
dataset with an index from another dataset.

**TopiOCQA** — [DrewZhang/topiocqa-passages-index](https://huggingface.co/datasets/DrewZhang/topiocqa-passages-index)
contains `part_001.parquet` through `part_006.parquet`, plus
`e5_Flat.index.part_aa` and `e5_Flat.index.part_ab` (about 86 GB total).

```bash
mkdir -p collection/topiocqa

# Separate download commands remain compatible with older hf CLIs.
hf download DrewZhang/topiocqa-passages-index \
  --repo-type dataset \
  --include "part_*.parquet" \
  --local-dir collection/topiocqa \
  --max-workers 8

hf download DrewZhang/topiocqa-passages-index \
  --repo-type dataset \
  --include "e5_Flat.index.part_*" \
  --local-dir collection/topiocqa \
  --max-workers 8

python scripts/build_qrecc_corpus.py \
  --collection-dir collection/topiocqa \
  --output collection/topiocqa/topiocqa_index.jsonl

cat collection/topiocqa/e5_Flat.index.part_* > collection/topiocqa/e5_Flat.index
```

**CoRAL** — [DrewZhang/coral-passages-index](https://huggingface.co/datasets/DrewZhang/coral-passages-index)
contains one passage shard (`part_001.parquet`) and one index chunk
(`e5_Flat.index.part_aa`), about 768 MB total.

```bash
mkdir -p collection/coral

hf download DrewZhang/coral-passages-index \
  --repo-type dataset \
  --include "part_001.parquet" \
  --local-dir collection/coral \
  --max-workers 8

hf download DrewZhang/coral-passages-index \
  --repo-type dataset \
  --include "e5_Flat.index.part_aa" \
  --local-dir collection/coral \
  --max-workers 8

python scripts/build_qrecc_corpus.py \
  --collection-dir collection/coral \
  --output collection/coral/coral_index.jsonl

cat collection/coral/e5_Flat.index.part_aa > collection/coral/e5_Flat.index
```

TopiOCQA NDCG@3 uses normalized passage-text matching rather than raw passage
IDs to accommodate differing source/retriever ID namespaces.  For every
dataset, preserve original passage-shard ordering when creating the JSONL
corpus; FAISS row IDs depend on it.

Launch and test the service:

```bash
mkdir -p logs
# For InsCiT training, use collection/inscit/e5_Flat.index and
# collection/inscit/inscit_index.jsonl.  Replace both paths together for
# QReCC or another aligned collection.
CUDA_VISIBLE_DEVICES=0,1 \
RETRIEVER_INDEX_PATH=$PWD/collection/inscit/e5_Flat.index \
RETRIEVER_CORPUS_PATH=$PWD/collection/inscit/inscit_index.jsonl \
RETRIEVER_MODEL_PATH=intfloat/e5-base-v2 \
bash scripts/run_local_retriever_server.sh \
  > logs/retriever.log 2>&1 &

python scripts/test_local_retriever.py --topk 3 \
  --query "What is the capital of France?"
```

The readiness check in every training/validation launcher deliberately uses `return_scores=true`; do not change the retriever to return a different response shape.

### 6. Launch the frozen user simulator

The policy’s private thought/tool/evidence trace is never sent to the simulator.  The simulator receives only the current sub-task, its permitted source context, and the user-visible `<answer>` content; it returns structured level 1/2/3 feedback.

```bash
mkdir -p logs
nohup env \
  CUDA_VISIBLE_DEVICES=0,1 \
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

## Training

### Canonical configuration

All provided full-method training launchers use exactly the current formal setting:

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
| Reward | action + answer-F1 + clarity + patience + format + clarify-F1 | action + answer-F1 + clarity + patience + format |

`EXACT_CONTEXT_BATCH=true` packs complete dialogues so that exactly 128 original sub-task contexts contribute to each update, without splitting a dialogue when calculating sibling advantages.  The collected trajectory count is therefore `128 × 8 = 1024` rollout rows per update.

### InsCiT: train and final online validation

The commands below assume a four-GPU node where GPUs `0,1` serve the
retriever and the 32B user simulator, and GPUs `2,3` are reserved for policy
training and validation.  On another layout, change these three variables
together: `CUDA_VISIBLE_DEVICES`, `N_GPUS`, and
`ULYSSES_SEQUENCE_PARALLEL_SIZE`.

```bash
# Qwen2.5-3B
CUDA_VISIBLE_DEVICES=2,3 \
N_GPUS=2 ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
USER_SIMULATOR_BASE_URL=http://127.0.0.1:8010 \
USER_SIMULATOR_MODEL=qwen32b-user-simulator \
INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
bash scripts/run_inscit_3b_full_train.sh
```

```bash
# Qwen2.5-7B
CUDA_VISIBLE_DEVICES=2,3 \
N_GPUS=2 ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
USER_SIMULATOR_BASE_URL=http://127.0.0.1:8010 \
USER_SIMULATOR_MODEL=qwen32b-user-simulator \
INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
bash scripts/run_inscit_7b_full_train.sh
```

### QReCC: train and final online validation

```bash
# Qwen2.5-3B
CUDA_VISIBLE_DEVICES=2,3 \
N_GPUS=2 ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
USER_SIMULATOR_BASE_URL=http://127.0.0.1:8010 \
USER_SIMULATOR_MODEL=qwen32b-user-simulator \
INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
bash scripts/run_qrecc_3b_full_train.sh
```

```bash
# Qwen2.5-7B
CUDA_VISIBLE_DEVICES=2,3 \
N_GPUS=2 ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
USER_SIMULATOR_BASE_URL=http://127.0.0.1:8010 \
USER_SIMULATOR_MODEL=qwen32b-user-simulator \
INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
bash scripts/run_qrecc_7b_full_train.sh
```

Each run automatically performs the full online test after the final step and then computes F1, BERTScore, NDCG@3, action accuracy (where applicable), format success rate, user satisfaction, simulator fallback rate, mean retry depth, and mean tool calls.

### Controlled ablations

The static-context/no-feedback study starts every sub-task from the canonical
dialogue prefix and uses no simulator feedback or retry. It retains answer-F1,
action, format, and clarification-F1 rewards:

```bash
CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2 ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
bash scripts/run_inscit_3b_static_no_feedback_train.sh
```

The canonical full method uses online, model-generated dialogue context and
simulator feedback; this control removes both while retaining the same
answer-correctness, action, and format supervision.

### Resume an interrupted run

The checkpoint `global_step_9` means the completed updates are 0–9.  Resume to a total of 15 steps as follows:

```bash
CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2 ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
USER_SIMULATOR_BASE_URL=http://127.0.0.1:8010 \
USER_SIMULATOR_MODEL=qwen32b-user-simulator \
RESUME_MODE=resume_path \
RESUME_FROM_PATH=$PWD/outputs/inscit/interactivechat_r1_inscit_3b_full_f1_satisfaction_15steps/global_step_9 \
TOTAL_STEPS=15 \
bash scripts/run_inscit_3b_full_train.sh
```

## Online evaluation

All validation launchers run the same interactive environment: the simulator can give feedback, policy history contains its own previous answers or canonical fallback responses, and top-3 retrieval happens at every valid tool call.  They write complete per-subtask online traces to `eval_log/`.

Use the same running retriever and user simulator.  For example, evaluate an InsCiT-trained 3B actor on TopiOCQA:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
N_GPUS=2 ULYSSES_SEQUENCE_PARALLEL_SIZE=2 \
USER_SIMULATOR_BASE_URL=http://127.0.0.1:8010 \
USER_SIMULATOR_MODEL=qwen32b-user-simulator \
CHECKPOINT_PATH=$PWD/outputs/inscit/interactivechat_r1_inscit_3b_full_f1_satisfaction_15steps/global_step_14 \
bash scripts/run_topiocqa_from_inscit_3b_full_val.sh
```

Available launchers:

| Source checkpoint | In-domain online test | Transfer online test |
|---|---|---|
| InsCiT 3B | `run_inscit_3b_full_val.sh` | `run_topiocqa_from_inscit_3b_full_val.sh` |
| InsCiT 7B | `run_inscit_7b_full_val.sh` | `run_topiocqa_from_inscit_7b_full_val.sh` |
| QReCC 3B | `run_qrecc_3b_full_val.sh` | `run_coral_from_qrecc_3b_full_val.sh` |
| QReCC 7B | `run_qrecc_7b_full_val.sh` | `run_coral_from_qrecc_7b_full_val.sh` |

## Outputs and metrics

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
- **`ModuleNotFoundError: tensordict` or `cannot import name ForkingPickler`:**
  do not upgrade TensorDict alone.  The released runtime is coupled to
  **PyTorch `2.4.0+cu121` + TensorDict `0.5.0`**; the latter is incompatible
  with PyTorch 2.5+.  Pull the latest repository version and rerun the full
  installer from a base shell:
  ```bash
  conda activate base
  MAX_JOBS=8 bash scripts/install_h100_eval_env.sh interactivechat-r1
  conda activate interactivechat-r1
  python -m pytest -q tests/trainer/ppo/test_simulated_user_sparse_grpo.py
  ```
  To inspect an existing environment before repairing it, run
  `python -c "import torch; print(torch.__version__)"`; it must print
  `2.4.0+cu121`.
- **CUDA OOM:** do not reduce the 8192 model context.  First reserve clean policy GPUs, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and lower only `ROLLOUT_GPU_MEMORY_UTILIZATION` (e.g. `0.04`).
- **`set: pipefail: invalid option name`:** the shell file has Windows CRLF endings.  On the Linux server run `sed -i 's/\r$//' scripts/<script>.sh` once.

## License and attribution

The FSDP/vLLM backend derives from the Apache-2.0 `verl` runtime.  Its original license is retained in [`LICENSE`](LICENSE).  Please preserve this attribution and separately comply with the licenses and terms of the benchmark datasets, dense retrieval collection, and Qwen checkpoints.
