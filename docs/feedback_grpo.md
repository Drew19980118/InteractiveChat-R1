# Alternating user/system Feedback GRPO on static InsCiT

This experimental recipe trains two independent Qwen2.5-3B-Instruct policies:
the system answers the current query, and the user generates feedback. The
user is a language policy, not a PPO value critic. Each policy has its own
parameters and optimizer. Training uses GRPO, without a learned value head.
The other policy is frozen during each phase.

The source is the static ConvAgent InsCiT dataset. Each example retains its
original conversation history and current query. Feedback introduces exactly
one additional response within that example; generated responses never become
the history of another source example. Both responses share the same labels.

## Reward and grouping

System reward follows the existing F1-plus-action ablation:

\[
R(y)=F(y)+0.5M(y),\qquad
M(y)=\begin{cases}1&\text{valid permissible terminal action}\\-0.5&\text{wrong or malformed action}.\end{cases}
\]

For a valid answer, `F` is the maximum token-set F1 of the single prediction
against the allowed answer references. Clarification and non-answer actions
receive no text F1. The action contribution to total reward is therefore
`+0.5` or `-0.25`. Retrieval reward and the separate format-reward ablation are
disabled. Search remains available to the system.

**User phase:** generate one shared first system response, sample `n` feedbacks,
then one second system response for each feedback. If the shared first
response has no legal terminal action, retry at most twice (three attempts in
total). Skip the group if all attempts fail. Remove feedback candidates whose
second system response is malformed. Skip groups with fewer than two valid
candidates. A legal but incorrect terminal action is retained.

For answer-to-answer branches, feedback reward is `R(second) - R(first)`.
For all other legal terminal-action pairs, it is only
`0.5 * (M(second) - M(first))`. Normalize within the retained feedback group
and update only the generated feedback tokens.

The user sees the original history, current query and extracted terminal
content. It does not see the system's thinking/search trace or the gold
answer. A fixed instruction to answer again using the action format is
appended by the program and is excluded from the user loss.

**System phase:** sample one shared first response for each source row. If it
has no legal terminal action, retry it at most twice; a source that remains
malformed is skipped. The frozen User policy generates one feedback for each
valid first response. That exact `first response + feedback` prompt is then
used to sample `n` second responses, which form one same-prompt GRPO group.
Malformed second responses are removed; groups with fewer than two valid
responses are skipped. Only second-response tokens receive a System-policy
gradient. The first response remains visible in diagnostics and in the
Direct-response evaluation, but has loss weight zero.

This is a coherent alternating policy-optimization experiment. It is not a
guarantee of convergence or an improvement over static baselines. GRPO rewards
are relative within a group: a negative absolute improvement can still have a
positive advantage if it is the least harmful candidate. A constant-reward
group has no reward-driven update. A single second-response sample makes the
feedback reward noisy; monitor improvement distributions and valid-group
rates, not only mean reward.

## Defaults and resource cost

| Setting | Default |
| --- | --- |
| GPUs | two visible GPUs; one full GPU per policy; sequence parallelism 1 |
| System / user initialization | Separate Qwen2.5-3B-Instruct policies |
| Source training batch / validation batch | 128 / 256 |
| Group size | 8 second responses per System feedback prompt |
| Alternation | 5 user batches, then 5 system batches |
| Holdout | 10% source-conversation split, seed 42 |
| Validation | Every 5 system updates |
| Selection score | Mean of F1, BERTScore F1, NDCG@3, each clipped to [0,1] |
| Early stopping | 3 consecutive validations without strict improvement |

With `n=8`, a complete System-phase source example produces one first-round
and eight second-round System episodes, plus one feedback: 9 System episodes
per source. A 128-source batch therefore produces at most 1,152 System
episodes before tool-call turns, rather than the former 9,216. Rollout and
updates remain chunked. Smaller `ROLLOUT_N` or source batches are useful for
an initial smoke run; they define a different training configuration and
should be logged.

The learned user policy is internal to the recipe. Do not start a separate
32B or 3B vLLM user-simulator service for this run. The native InsCiT retriever
must already respond at `http://127.0.0.1:8002/retrieve`. Use the project
retriever instructions with the InsCiT index/corpus, not a QReCC index.

## Setup, authentication, and retriever

Run from the InteractiveChat-R1 project root in the GPU server environment.
Required local inputs are the Qwen2.5-3B-Instruct model, raw ConvAgent InsCiT
train/test Parquet files, the InsCiT E5 FAISS index/corpus, and a running local
retriever. The recipe creates the shared holdout from the train source; do not
pass an already-split train file.

Install the project environment and evaluation/retrieval dependencies once:

```bash
MAX_JOBS=8 bash scripts/install_h100_eval_env.sh interactivechat-r1
conda activate interactivechat-r1
python -m pip install -r requirements-eval-metrics.txt
python -m pip install -r requirements-retriever.txt
python -m pip install -U wandb huggingface_hub
```

Authenticate W&B and Hugging Face once per server. The Python login path is
compatible with both older and newer Hugging Face CLI installations; never put
your token in a shell command or tracked file.

```bash
wandb login
python - <<'PY'
from huggingface_hub import HfApi, login
login()
print("Hugging Face account:", HfApi().whoami()["name"])
PY
```

Start the **InsCiT** GPU-FAISS retriever on GPUs 0 and 1 before training. It
must use the InsCiT index and row-aligned corpus, never the QReCC pair:

```bash
mkdir -p logs
nohup env \
  CUDA_VISIBLE_DEVICES=0,1 \
  RETRIEVER_FAISS_GPU=true \
  RETRIEVER_INDEX_PATH="$PWD/collection/inscit/e5_Flat.index" \
  RETRIEVER_CORPUS_PATH="$PWD/collection/inscit/inscit_index.jsonl" \
  RETRIEVER_MODEL_PATH=intfloat/e5-base-v2 \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  bash scripts/run_local_retriever_server.sh \
  > logs/inscit_feedback_grpo_retriever.log 2>&1 &
```

The launcher performs a `/retrieve` readiness probe. If the GPU FAISS binding
is not available, follow the GPU-FAISS installation and verification section
in the repository README before launching it.

## Start training, test, export and upload

```bash
mkdir -p logs
nohup env \
  CUDA_VISIBLE_DEVICES=2,3 \
  N_GPUS=2 \
  ULYSSES_SEQUENCE_PARALLEL_SIZE=1 \
  SYSTEM_MODEL_PATH="$PWD/models/Qwen2.5-3B-Instruct" \
  USER_MODEL_PATH="$PWD/models/Qwen2.5-3B-Instruct" \
  SELECTION_SETTING=feedback-refinement \
  EXPERIMENT_NAME=feedback_grpo_inscit_qwen25_3b_feedback_second_only_cuda23_v1 \
  HOLDOUT_FRACTION=0.10 \
  SPLIT_SEED=42 \
  WANDB_ENABLED=true \
  WANDB_PROJECT=interactivechat-r1 \
  WANDB_RUN_GROUP=inscit_feedback_grpo_3b_feedback_second_only_cuda23 \
  HF_UPLOAD=true \
  HF_SYSTEM_REPO_ID=DrewZhang/interactivechat-r1-feedback-grpo-inscit-qwen25-3b-feedback-second-only-cuda23-system \
  HF_USER_REPO_ID=DrewZhang/interactivechat-r1-feedback-grpo-inscit-qwen25-3b-feedback-second-only-cuda23-user \
  INTERACTIVECHAT_CONDA_ENV=interactivechat-r1 \
  bash scripts/run_feedback_grpo_inscit_3b_train_eval_export_upload.sh \
  > logs/feedback_grpo_inscit_3b_feedback_second_only_cuda23_v1.log 2>&1 &
```

`HF_UPLOAD=false` skips authentication and both uploads while retaining local
export. When enabled, the launcher authenticates before training, checks the
installed CLI help to select `--repo-type` or `--type`, and uploads system then
user sequentially. These are two separate Hub transactions; if one fails,
retry its upload from the existing exported folder. Never assume the pair is
fully available until both uploads succeed. Keep credentials in the local HF
login cache or `HF_TOKEN`, not in command-line arguments.

The single-policy weights, tokenizer and config are uploaded for both roles,
in distinct repositories. Every run uploads its one selected System/User pair.
Optimizer and phase/sampler state remain in the local checkpoint for resuming
training and are not uploaded as policy weights.

## Validation, best checkpoints and resume

Each run evaluates both response views, but selects **only one** of them using
the mean of the normalized F1, BERTScore F1 and NDCG@3 values. Choose the
setting before starting a fresh run:

- `SELECTION_SETTING=direct-response`: select and early-stop using the first
  System response. This is directly comparable with one-response ConvAgent and
  ChatR1. The System loss still comes only from the second-round GRPO groups;
  direct response is a separate selection/evaluation track, not a direct
  first-round loss term.
- `SELECTION_SETTING=feedback-refinement`: select and early-stop using the
  strict second response after one learned User-feedback turn. Report it as a
  two-round interaction result, not as a like-for-like static-baseline score.

The runs must use distinct experiment names and start from the original model
weights. Recipe version 4 records the `singleton_first_second_only_v1` mode in
its configuration signature. It intentionally rejects `RESUME=true` for the
older `*_v1` experiment, whose System phase used the former `n + n^2` sampling
and mixed first/second loss. Early stopping happens when the selected setting
has not improved for the configured patience. Two-round evaluation never
selects the better of the first and second answer: a malformed second terminal
response receives zero answer score.

Artifacts are under these roots (overridable by the same uppercase variables):

```text
outputs/feedback_grpo/<experiment>/
  final_checkpoint.txt                 # best pair for this run's selected setting
  global_step_<selected_step>/
    system/     # model, optimizer and extra-state shards
    user/       # model, optimizer and extra-state shards
    state.json  # trainer counters, phase and deterministic data state
eval_log/feedback_grpo/<experiment>/
exports/feedback_grpo/<experiment>/
  system/
  user/
```

During training, save a new best into a staging directory, verify complete
shards for both policies, then atomically publish its completed pair and best
pointer. Only after that successful replacement are older complete checkpoint
directories owned by this recipe pruned. Foreign/unowned directories and
symlinks are not followed or removed. A failed save preserves the old best.
Space for the old best and one candidate pair is necessary while saving;
keeping one best does not eliminate this temporary disk-space requirement.

Set `RESUME=true` and retain the same experiment name/configuration and
selection setting to load its selected pair and optimizer/coordinator state. It
resumes the selected pair, not unsaved updates after that checkpoint. Use a
fresh experiment name for the other setting or an independent run. A process
crash may leave an explicit save lock; do not delete it while any save process
is running.

W&B is enabled by default. The run contains separate user/system progress and
GRPO diagnostics, plus both validation modes. The deterministic static score,
feedback reward deltas, valid/filtered group fractions and response format
rates should be interpreted together.
