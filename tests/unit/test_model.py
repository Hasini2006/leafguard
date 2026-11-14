"""Unit tests for the promotion gate rules and the serving wrapper.

The gate is what stops a worse model going live, so each rule is checked on its own. The
wrapper is exercised with a tiny TorchScript network, which proves the exact code the model
server runs can decode, resize and score an image without MLflow or a real model.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from conftest import make_leaf

torch = pytest.importorskip("torch")

from leafguard.training.registry import decide  # noqa: E402

RULES = {"min_accuracy": 0.70, "max_drop_vs_champion": 0.02, "max_latency_ms": 400}


def metrics(accuracy: float = 0.9, latency: float = 20.0) -> dict:
    return {"accuracy": accuracy, "latency_p95_ms": latency}


# ----------------------------------------------------------------- promotion gate


def test_a_good_first_model_is_promoted() -> None:
    passed, reasons = decide(metrics(), RULES, champion_accuracy=None)
    assert passed
    assert any("no champion exists yet" in line for line in reasons)
    assert len(reasons) == 3  # one readable reason per rule


def test_a_model_below_the_accuracy_floor_is_refused() -> None:
    passed, reasons = decide(metrics(accuracy=0.5), RULES, champion_accuracy=None)
    assert not passed
    assert any("below the minimum" in line for line in reasons)


def test_a_model_worse_than_the_champion_is_refused() -> None:
    passed, reasons = decide(metrics(accuracy=0.85), RULES, champion_accuracy=0.90)
    assert not passed
    assert any("dropped" in line for line in reasons)


def test_a_small_regression_within_the_allowance_is_accepted() -> None:
    passed, _ = decide(metrics(accuracy=0.89), RULES, champion_accuracy=0.90)
    assert passed


def test_a_slow_model_is_refused_even_when_accurate() -> None:
    passed, reasons = decide(metrics(latency=900), RULES, champion_accuracy=None)
    assert not passed
    assert any("exceeds" in line for line in reasons)


# ----------------------------------------------------------------- serving wrapper


class TinyNet(torch.nn.Module):
    """Averages each colour channel and maps it to three classes."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.head(x.mean(dim=(2, 3)))


@pytest.fixture
def loaded_wrapper(tmp_path: Path):
    from leafguard.training.wrapper import LeafModel

    weights = tmp_path / "model.pt"
    torch.jit.trace(TinyNet().eval(), torch.zeros(1, 3, 32, 32)).save(str(weights))
    classes = tmp_path / "classes.json"
    classes.write_text(json.dumps({"classes": ["a", "b", "c"], "image_size": 32}))

    model = LeafModel()
    model.load_context(
        SimpleNamespace(artifacts={"weights": str(weights), "classes": str(classes)})
    )
    return model


def test_the_wrapper_returns_one_probability_per_class(loaded_wrapper) -> None:
    frame = pd.DataFrame({"image_b64": [base64.b64encode(make_leaf()).decode()]})
    scores = loaded_wrapper.predict(None, frame)
    assert list(scores.columns) == ["a", "b", "c"]
    assert scores.iloc[0].sum() == pytest.approx(1.0, abs=1e-4)


def test_the_wrapper_resizes_to_the_trained_size(loaded_wrapper) -> None:
    tensor = loaded_wrapper.prepare(make_leaf(size=150, image_format="PNG"))
    assert tuple(tensor.shape) == (1, 3, 32, 32)
