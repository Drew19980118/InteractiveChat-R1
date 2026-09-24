"""CPU-only tests for the two explicit Feedback-GRPO selection settings."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def recipe_main(monkeypatch):
    """Load coordinator selection helpers without Ray, Torch or veRL imports."""
    for package in ("verl", "verl.recipe", "verl.recipe.feedback_grpo"):
        module = types.ModuleType(package)
        module.__path__ = [str(ROOT.joinpath(*package.split(".")))]
        monkeypatch.setitem(sys.modules, package, module)

    artifacts = types.ModuleType("verl.recipe.feedback_grpo.artifacts")
    artifacts.PairCheckpointStore = object
    artifacts.evaluate_jsonl = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, artifacts.__name__, artifacts)

    engine = types.ModuleType("verl.recipe.feedback_grpo.engine")
    engine.Episode = object
    engine.SystemRequest = object
    engine.TwoRoundCollector = object
    monkeypatch.setitem(sys.modules, engine.__name__, engine)

    protocol = types.ModuleType("verl.recipe.feedback_grpo.protocol")
    protocol.StaticExample = object
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)

    name = "verl.recipe.feedback_grpo.main"
    spec = importlib.util.spec_from_file_location(name, ROOT / "verl/recipe/feedback_grpo/main.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_each_fresh_run_has_one_explicit_selection_mode(recipe_main):
    assert recipe_main.SELECTION_SETTINGS == {
        "direct-response": "first_response",
        "feedback-refinement": "feedback_round_two",
    }


def test_direct_and_feedback_runs_select_different_summaries(recipe_main):
    first = {"metrics": {"f1": 0.6, "bertscore_f1": 0.8, "ndcg_at_3": 0.4}}
    revised = {"metrics": {"f1": 0.4, "bertscore_f1": 0.7, "ndcg_at_3": 0.9}}
    assert recipe_main._selection_summary("direct-response", first, revised) is first
    assert recipe_main._selection_summary("feedback-refinement", first, revised) is revised
    with pytest.raises(ValueError, match="Unknown selection setting"):
        recipe_main._selection_summary("both", first, revised)


def test_selection_score_is_unweighted_mean_of_bounded_metrics(recipe_main):
    summary = {"metrics": {"f1": 0.3, "bertscore_f1": 0.9, "ndcg_at_3": 0.6}}
    assert recipe_main._selection_score(summary) == pytest.approx(0.6)


def test_parser_rejects_unknown_selection_setting(recipe_main, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    data = tmp_path / "data.parquet"
    data.write_bytes(b"placeholder")
    with pytest.raises(SystemExit):
        recipe_main.parse_args([
            "--system-model", str(model), "--user-model", str(model),
            "--train-file", str(data), "--test-file", str(data),
            "--experiment", "test", "--selection-setting", "both",
        ])


def test_empty_system_batch_does_not_call_system_update_or_advance_its_counter(recipe_main, monkeypatch, tmp_path):
    """The coordinator must not turn an all-invalid collector result into an update."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    data = tmp_path / "data.parquet"
    data.write_bytes(b"placeholder")
    args = recipe_main.parse_args([
        "--system-model", str(model), "--user-model", str(model),
        "--train-file", str(data), "--test-file", str(data),
        "--experiment", "empty-system", "--user-updates-per-phase", "1",
        "--system-updates-per-phase", "1", "--max-system-updates", "1",
        "--max-empty-system-batches", "1", "--output-root", str(tmp_path / "outputs"),
        "--eval-root", str(tmp_path / "eval"), "--export-root", str(tmp_path / "exports"),
    ])

    class FakeStore:
        def __init__(self, root):
            self.root = root
            root.mkdir(parents=True, exist_ok=True)

    class FakeCollector:
        def __init__(self, *_args, **_kwargs):
            pass

        def user_batch(self, _examples, _batch_id):
            return [object()], {"user/valid_groups": 1.0}, []

        def system_batch(self, _examples, _batch_id):
            return [], {"system/valid_groups": 0.0}, []

    class FakeBackend:
        instance = None

        def __init__(self, _args):
            self.updated = []
            self.closed = False
            type(self).instance = self

        def update(self, role, _items):
            self.updated.append(role)
            return {}

        def close(self):
            self.closed = True

    class FakeTracking:
        instances = []

        def __init__(self, **_kwargs):
            self.events = []
            type(self).instances.append(self)

        def log(self, values, step):
            self.events.append((values, step))

    backend_module = types.ModuleType("verl.recipe.feedback_grpo.backend")
    backend_module.RayBackend = FakeBackend
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)
    utils_module = types.ModuleType("verl.utils")
    utils_module.__path__ = []
    tracking_module = types.ModuleType("verl.utils.tracking")
    tracking_module.Tracking = FakeTracking
    monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
    monkeypatch.setitem(sys.modules, tracking_module.__name__, tracking_module)
    monkeypatch.setattr(recipe_main, "PairCheckpointStore", FakeStore)
    monkeypatch.setattr(recipe_main, "TwoRoundCollector", FakeCollector)
    monkeypatch.setattr(recipe_main, "_split_examples", lambda _args: ([object()], [object()], [object()], {"test": True}))

    with pytest.raises(RuntimeError, match="Too many consecutive empty system batches"):
        recipe_main.run(args)

    assert FakeBackend.instance.updated == ["user"]
    assert FakeBackend.instance.closed
    assert any(values.get("system/empty_effective_batch") == 1.0
               for values, _step in FakeTracking.instances[0].events)
