"""Paired, resumable best checkpoints and the existing ConvAgent evaluator.

Only a newly selected best pair is saved.  The old pair stays available until
both policies and the coordinator state have been committed and the pointer
has been atomically replaced.  No arbitrary checkpoint cleanup is performed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Callable
import uuid
import warnings


_RECIPE = "feedback_grpo"
_VERSION = 1
_STORE_MARKER = ".feedback_grpo_store.json"
_OWNER_MARKER = ".feedback_grpo_owner.json"
_COMPLETE_MARKER = ".feedback_grpo_complete.json"
_STEP = re.compile(r"global_step_(0|[1-9][0-9]*)\Z")
_SHARD = re.compile(r"(model|optim|extra_state)_world_size_([1-9][0-9]*)_rank_([0-9]+)\.pt\Z")
_ROLES = ("system", "user")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    # Windows junctions/reparse points must not redirect checkpoint cleanup.
    try:
        attrs = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _reject_link_components(path: Path) -> None:
    for component in (path, *path.parents):
        if _is_link(component):
            raise ValueError(f"Checkpoint paths must not contain symlinks or junctions: {component}")


def _sync_directory(path: Path) -> None:
    # Windows does not expose directory fsync through os.open. File fsync and
    # atomic replace still protect its metadata files from partial writes.
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    _reject_link_components(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict:
    _reject_link_components(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _regular_files(directory: Path) -> list[Path]:
    _reject_link_components(directory)
    if not directory.is_dir():
        raise ValueError(f"Missing checkpoint directory: {directory}")
    result = []
    for parent, directories, files in os.walk(directory, followlinks=False):
        for name in directories + files:
            item = Path(parent) / name
            if _is_link(item):
                raise ValueError(f"Checkpoint contents must not include symlinks: {item}")
        for name in files:
            item = Path(parent) / name
            if not stat.S_ISREG(item.stat().st_mode):
                raise ValueError(f"Checkpoint contains a non-regular file: {item}")
            result.append(item)
    return result


def _policy_inventory(path: Path) -> dict[str, int]:
    """Require all model, optimizer and extra-state FSDP shards for resume."""
    files = _regular_files(path)
    shards: dict[str, set[tuple[int, int]]] = {kind: set() for kind in ("model", "optim", "extra_state")}
    for item in files:
        match = _SHARD.fullmatch(item.name)
        if match and item.parent == path:
            kind, world_size, rank = match.groups()
            shards[kind].add((int(world_size), int(rank)))
    if not shards["model"]:
        raise ValueError(f"No FSDP actor model shards in {path}")
    world_sizes = {world for world, _ in shards["model"]}
    if len(world_sizes) != 1:
        raise ValueError(f"Mixed FSDP world sizes in {path}")
    world_size = next(iter(world_sizes))
    expected = {(world_size, rank) for rank in range(world_size)}
    for kind, found in shards.items():
        if found != expected:
            raise ValueError(f"Incomplete {kind} shards in {path}; expected {sorted(expected)}, found {sorted(found)}")
    inventory = {str(item.relative_to(path).as_posix()): item.stat().st_size for item in files}
    if any(size <= 0 for name, size in inventory.items() if _SHARD.fullmatch(name)):
        raise ValueError(f"Empty FSDP checkpoint shard in {path}")
    return inventory


class PairCheckpointStore:
    """Save the system/user pair selected together by holdout validation.

    ``save_policy(role, path, step)`` must synchronously save a full FSDP actor
    checkpoint directly into ``path`` (including optimizer and extra state).
    Coordinator counters, phase, sampler state and RNG state belong in the
    JSON-serializable ``state`` object. Each policy's RNG is also in extra state.
    A save needs temporary space for the old best and the new candidate pair.
    """

    def __init__(self, root: str | Path):
        root = Path(root).expanduser().absolute()
        _reject_link_components(root)
        root = root.resolve()
        repository = Path(__file__).resolve().parents[3]
        home = Path.home().resolve()
        if root == Path(root.anchor) or root == Path.cwd().resolve() or root == home or root in repository.parents or root == repository:
            raise ValueError(f"Use a dedicated checkpoint directory, not a broad directory: {root}")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        marker_path = root / _STORE_MARKER
        if not marker_path.exists():
            marker = {"recipe": _RECIPE, "version": _VERSION, "store_id": uuid.uuid4().hex}
            # Exclusive creation prevents two independent stores from claiming
            # the same directory with different ownership IDs.
            try:
                with marker_path.open("xb") as stream:
                    stream.write(_json_bytes(marker))
                    stream.flush()
                    os.fsync(stream.fileno())
                _sync_directory(root)
            except FileExistsError:
                pass
        marker = _read_json(marker_path)
        if marker.get("recipe") != _RECIPE or marker.get("version") != _VERSION or not isinstance(marker.get("store_id"), str):
            raise ValueError(f"Unrecognized checkpoint store: {marker_path}")
        self.store_id = marker["store_id"]

    def _owner(self, step: int) -> dict:
        return {"recipe": _RECIPE, "version": _VERSION, "store_id": self.store_id, "step": step}

    def _assert_root(self) -> None:
        _reject_link_components(self.root)
        marker = _read_json(self.root / _STORE_MARKER)
        if marker.get("store_id") != self.store_id or marker.get("recipe") != _RECIPE:
            raise ValueError("Checkpoint store ownership changed")

    def _check_pair(self, path: Path) -> tuple[dict, dict]:
        self._assert_root()
        _reject_link_components(path)
        if not path.is_absolute() or path.parent != self.root or not _STEP.fullmatch(path.name):
            raise ValueError(f"Checkpoint must be a direct global_step_* child of {self.root}: {path}")
        owner = self._owner(int(path.name.removeprefix("global_step_")))
        if _read_json(path / _OWNER_MARKER) != owner:
            raise ValueError(f"Checkpoint is not owned by this store: {path}")
        complete = _read_json(path / _COMPLETE_MARKER)
        if any(complete.get(key) != value for key, value in owner.items()):
            raise ValueError(f"Invalid paired checkpoint marker: {path}")
        _reject_link_components(path / "state.json")
        state_payload = (path / "state.json").read_bytes()
        if hashlib.sha256(state_payload).hexdigest() != complete.get("state_sha256"):
            raise ValueError(f"Checkpoint coordinator state failed integrity check: {path}")
        state = json.loads(state_payload)
        if not isinstance(state, dict):
            raise ValueError(f"Invalid coordinator state: {path}")
        inventories = {}
        for role in _ROLES:
            inventories[role] = _policy_inventory(path / role)
            if inventories[role] != complete.get("policies", {}).get(role):
                raise ValueError(f"Checkpoint {role} files differ from completed inventory: {path}")
        self._check_matching_world_sizes(inventories)
        return complete, state

    @staticmethod
    def _check_matching_world_sizes(policies: dict) -> None:
        model_shards = [{name for name in policies[role] if name.startswith("model_world_size_")} for role in _ROLES]
        if model_shards[0] != model_shards[1]:
            raise ValueError("System and user checkpoints have different FSDP world sizes")

    def _remove_owned(self, path: Path, owner: dict) -> None:
        self._assert_root()
        if path.parent != self.root or _read_json(path / _OWNER_MARKER) != owner:
            raise ValueError(f"Refusing cleanup outside this checkpoint store: {path}")
        _regular_files(path)  # Refuse links, junctions and special files before deletion.
        shutil.rmtree(path)
        _sync_directory(self.root)

    def save(self, step: int, state: dict, save_policy: Callable[[str, Path, int], None]) -> Path:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Checkpoint step must be a nonnegative integer")
        if not isinstance(state, dict):
            raise TypeError("Coordinator state must be a dict")
        state_payload = _json_bytes(state)
        self._assert_root()
        final = self.root / f"global_step_{step}"
        if final.exists() or final.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing checkpoint: {final}")
        lock = self.root / ".feedback_grpo_save.lock"
        _reject_link_components(lock)
        # Never auto-delete another process's lock, even if it looks stale.
        lock_stream = lock.open("x")
        stage = self.root / f".feedback_grpo_stage_{step}_{uuid.uuid4().hex}"
        owner = self._owner(step)
        try:
            stage.mkdir()
            _atomic_write(stage / _OWNER_MARKER, _json_bytes(owner))
            policies = {}
            for role in _ROLES:
                destination = stage / role
                destination.mkdir()
                save_policy(role, destination, step)
                policies[role] = _policy_inventory(destination)
            self._check_matching_world_sizes(policies)
            _atomic_write(stage / "state.json", state_payload)
            # torch.save has closed its files; sync the completed files before
            # publishing a marker claiming that this pair can be resumed.
            for item in _regular_files(stage):
                with item.open("r+b") as stream:
                    os.fsync(stream.fileno())
            complete = {**owner, "state_sha256": hashlib.sha256(state_payload).hexdigest(), "policies": policies}
            _atomic_write(stage / _COMPLETE_MARKER, _json_bytes(complete))
            for role in _ROLES:
                _sync_directory(stage / role)
            _sync_directory(stage)
            stage.rename(final)
            _sync_directory(self.root)
            _atomic_write(self.root / "final_checkpoint.txt", (str(final) + "\n").encode("utf-8"))
            # Only older complete checkpoints bearing this exact store ID can
            # be pruned; foreign/unowned directories are left untouched.
            for candidate in self.root.iterdir():
                match = _STEP.fullmatch(candidate.name)
                if not match or int(match.group(1)) >= step or _is_link(candidate):
                    continue
                try:
                    self._check_pair(candidate)
                except (OSError, ValueError, TypeError):
                    continue
                try:
                    self._remove_owned(candidate, self._owner(int(match.group(1))))
                except OSError as error:
                    warnings.warn(f"Best checkpoint saved, but old checkpoint cleanup failed for {candidate}: {error}")
            return final
        finally:
            if stage.exists():
                try:
                    self._remove_owned(stage, owner)
                except (OSError, ValueError) as error:
                    warnings.warn(f"Incomplete checkpoint retained for inspection at {stage}: {error}")
            lock_stream.close()
            lock.unlink()

    def load_state(self) -> tuple[Path, dict]:
        self._assert_root()
        pointer = self.root / "final_checkpoint.txt"
        _reject_link_components(pointer)
        path = Path(pointer.read_text(encoding="utf-8").strip())
        _, state = self._check_pair(path)
        return path, state


def evaluate_jsonl(
    input_path: str | Path,
    output_dir: str | Path,
    model: str = "roberta-large",
    device: str = "cpu",
    batch_size: int = 8,
) -> dict:
    """Run the native evaluator with the shared maximum-reference protocol."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("BERTScore batch_size must be a positive integer")
    input_path = Path(input_path).resolve(strict=True)
    if not input_path.is_file():
        raise ValueError(f"Evaluation input must be a JSONL file: {input_path}")
    output_dir = Path(output_dir).resolve()
    repository = Path(__file__).resolve().parents[3]
    evaluator = repository / "scripts" / "compute_convagent_eval_metrics.py"
    subprocess.run(
        [sys.executable, str(evaluator), "--input", str(input_path), "--output-dir", str(output_dir),
         "--bert-score-model", model, "--bert-score-device", device,
         "--bert-score-batch-size", str(batch_size), "--multi-reference"],
        cwd=repository,
        check=True,
    )
    summary = _read_json(output_dir / "metrics_summary.json")
    if Path(summary.get("input_jsonl", "")).resolve() != input_path:
        raise ValueError("Evaluator summary does not describe the requested JSONL")
    if not isinstance(summary.get("metrics"), dict):
        raise ValueError("Evaluator summary is missing metrics")
    return summary
