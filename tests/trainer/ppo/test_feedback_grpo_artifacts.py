"""Checkpoint safety tests do not load Torch or require accelerator hardware."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[3] / "verl/recipe/feedback_grpo/artifacts.py"
_SPEC = importlib.util.spec_from_file_location("feedback_grpo_artifacts", _MODULE_PATH)
artifacts = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(artifacts)


def _save_policy(role, path, step):
    for rank in range(2):
        for kind in ("model", "optim", "extra_state"):
            (path / f"{kind}_world_size_2_rank_{rank}.pt").write_bytes(f"{role}:{kind}:{step}:{rank}".encode())
    (path / "huggingface").mkdir()
    (path / "huggingface/config.json").write_text('{"model_type": "qwen2"}', encoding="utf-8")


def test_new_best_swaps_paired_pointer_then_prunes_previous(tmp_path):
    store = artifacts.PairCheckpointStore(tmp_path / "checkpoints")
    old = store.save(5, {"system_step": 5, "user_step": 10, "phase": "system"}, _save_policy)
    foreign = store.root / "global_step_1"
    foreign.mkdir()
    (foreign / "valuable.txt").write_text("not owned by the recipe")
    newest = store.save(10, {"system_step": 10, "user_step": 15, "phase": "user", "rng": [1, 2, 3], "cursor": 128}, _save_policy)
    assert not old.exists()
    assert foreign.is_dir()
    pair, state = artifacts.PairCheckpointStore(store.root).load_state()
    assert pair == newest
    assert state == {"system_step": 10, "user_step": 15, "phase": "user", "rng": [1, 2, 3], "cursor": 128}
    for role in ("system", "user"):
        assert (pair / role / "optim_world_size_2_rank_1.pt").read_bytes() == f"{role}:optim:10:1".encode()


def test_partial_second_policy_failure_keeps_old_best_and_cleans_only_stage(tmp_path):
    store = artifacts.PairCheckpointStore(tmp_path / "checkpoints")
    old = store.save(5, {"step": 5}, _save_policy)

    def fail_user(role, path, step):
        _save_policy(role, path, step)
        if role == "user":
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        store.save(10, {"step": 10}, fail_user)
    assert store.load_state() == (old, {"step": 5})
    assert not (store.root / "global_step_10").exists()
    assert not list(store.root.glob(".feedback_grpo_stage_*"))


def test_missing_optimizer_shard_never_publishes_or_prunes(tmp_path):
    store = artifacts.PairCheckpointStore(tmp_path / "checkpoints")
    old = store.save(1, {}, _save_policy)

    def missing_optimizer(role, path, step):
        _save_policy(role, path, step)
        (path / "optim_world_size_2_rank_1.pt").unlink()

    with pytest.raises(ValueError, match="Incomplete optim"):
        store.save(2, {}, missing_optimizer)
    assert store.load_state()[0] == old


def test_pointer_publish_failure_preserves_old_checkpoint(tmp_path, monkeypatch):
    store = artifacts.PairCheckpointStore(tmp_path / "checkpoints")
    old = store.save(1, {"step": 1}, _save_policy)
    original = artifacts._atomic_write

    def fail_pointer(path, payload):
        if path.name == "final_checkpoint.txt":
            raise OSError("pointer write failed")
        return original(path, payload)

    monkeypatch.setattr(artifacts, "_atomic_write", fail_pointer)
    with pytest.raises(OSError, match="pointer write failed"):
        store.save(2, {"step": 2}, _save_policy)
    assert store.load_state() == (old, {"step": 1})
    assert (store.root / "global_step_2").is_dir()


def test_load_rejects_out_of_root_pointer_and_state_tampering(tmp_path):
    store = artifacts.PairCheckpointStore(tmp_path / "checkpoints")
    best = store.save(1, {"system_step": 1}, _save_policy)
    pointer = store.root / "final_checkpoint.txt"
    pointer.write_text(str(tmp_path / "global_step_1"))
    with pytest.raises(ValueError, match="direct global_step"):
        store.load_state()
    pointer.write_text(str(best))
    (best / "state.json").write_text('{"system_step": 999}')
    with pytest.raises(ValueError, match="integrity"):
        store.load_state()


def test_load_rejects_modified_policy_inventory(tmp_path):
    store = artifacts.PairCheckpointStore(tmp_path / "checkpoints")
    best = store.save(1, {}, _save_policy)
    (best / "user/model_world_size_2_rank_0.pt").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="inventory"):
        store.load_state()


def test_world_size_mismatch_rejects_pair(tmp_path):
    store = artifacts.PairCheckpointStore(tmp_path / "checkpoints")

    def mismatched(role, path, step):
        if role == "system":
            _save_policy(role, path, step)
        else:
            for kind in ("model", "optim", "extra_state"):
                (path / f"{kind}_world_size_1_rank_0.pt").write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="different FSDP world sizes"):
        store.save(1, {}, mismatched)
    assert not (store.root / "final_checkpoint.txt").exists()


def test_refuses_existing_step_and_does_not_overwrite_state(tmp_path):
    store = artifacts.PairCheckpointStore(tmp_path / "checkpoints")
    best = store.save(1, {"value": "original"}, _save_policy)
    with pytest.raises(FileExistsError):
        store.save(1, {"value": "overwritten"}, _save_policy)
    assert store.load_state() == (best, {"value": "original"})


def test_foreign_store_complete_pair_is_not_pruned(tmp_path):
    store = artifacts.PairCheckpointStore(tmp_path / "checkpoints")
    other = artifacts.PairCheckpointStore(tmp_path / "other")
    foreign = other.save(1, {"other": True}, _save_policy)
    foreign.rename(store.root / "global_step_1")
    store.save(2, {}, _save_policy)
    assert (store.root / "global_step_1/system/model_world_size_2_rank_0.pt").is_file()


@pytest.mark.parametrize("directory", [Path.cwd(), Path.home(), Path(Path.cwd().anchor)])
def test_refuses_broad_checkpoint_roots(directory):
    with pytest.raises(ValueError, match="dedicated checkpoint"):
        artifacts.PairCheckpointStore(directory)


def _symlink_or_skip(link, destination):
    try:
        link.symlink_to(destination, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this host")


def test_refuses_symlink_root_and_selected_checkpoint(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    _symlink_or_skip(linked, actual)
    with pytest.raises(ValueError, match="symlinks"):
        artifacts.PairCheckpointStore(linked)
    store = artifacts.PairCheckpointStore(tmp_path / "checkpoints")
    best = store.save(1, {}, _save_policy)
    alias = store.root / "global_step_9"
    _symlink_or_skip(alias, best)
    (store.root / "final_checkpoint.txt").write_text(str(alias))
    with pytest.raises(ValueError, match="symlinks"):
        store.load_state()


def test_pruning_does_not_follow_unowned_symlink(tmp_path):
    store = artifacts.PairCheckpointStore(tmp_path / "checkpoints")
    external = tmp_path / "external"
    external.mkdir()
    preserved = external / "user_data"
    preserved.write_text("keep")
    _symlink_or_skip(store.root / "global_step_1", external)
    store.save(2, {}, _save_policy)
    assert preserved.read_text() == "keep"


def test_native_evaluator_uses_current_python_and_max_reference(tmp_path, monkeypatch):
    source = tmp_path / "predictions.jsonl"
    source.write_text("{}\n")
    output = tmp_path / "metrics"
    expected = {"input_jsonl": str(source.resolve()), "metrics": {"f1": 0.4, "bertscore_f1": 0.8, "ndcg_at_3": 0.3}}
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append((command, cwd, check))
        output.mkdir()
        (output / "metrics_summary.json").write_text(json.dumps(expected))

    monkeypatch.setattr(artifacts.subprocess, "run", fake_run)
    assert artifacts.evaluate_jsonl(source, output, device="cuda:0", batch_size=16) == expected
    command, cwd, check = calls[0]
    assert command[0] == artifacts.sys.executable
    assert Path(command[1]) == cwd / "scripts/compute_convagent_eval_metrics.py"
    assert "--multi-reference" in command
    assert command[command.index("--bert-score-device") + 1] == "cuda:0"
    assert command[command.index("--bert-score-batch-size") + 1] == "16"
    assert check is True


def test_native_evaluator_does_not_return_stale_summary_on_failure(tmp_path, monkeypatch):
    source = tmp_path / "predictions.jsonl"
    source.write_text("{}\n")
    output = tmp_path / "metrics"
    output.mkdir()
    (output / "metrics_summary.json").write_text('{"metrics": {"f1": 1}}')

    def fail_run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(artifacts.subprocess, "run", fail_run)
    with pytest.raises(subprocess.CalledProcessError):
        artifacts.evaluate_jsonl(source, output)
