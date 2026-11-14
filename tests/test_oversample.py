"""Offline checks of the oversampling study on small synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agender_mtl import oversample, step2
from test_benchmark import small_lat

NAMES = ["CALLS", "MB", "BRAND=A", "BRAND=B", "BRAND=C", "NET=3G", "NET=LTE"]


def toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    numbers = rng.normal(size=(n, 2))
    brand = np.eye(3)[rng.integers(0, 3, n)]
    net = np.eye(3)[rng.integers(0, 3, n)][:, :2]          # third value not kept: both columns 0
    X = np.column_stack([numbers, brand, net])
    age = rng.choice(4, n, p=[0.05, 0.45, 0.4, 0.1])
    male = rng.integers(0, 2, n)
    return X, male, age


def test_layout_round_trip_and_snap():
    X, _, _ = toy()
    layout = oversample.Layout(NAMES)
    assert layout.numeric == [0, 1] and list(layout.groups) == ["BRAND", "NET"]
    numbers, codes = layout.decode(X)
    assert np.allclose(layout.encode(numbers, codes), X)
    blurred = X.copy()
    blurred[:, 2:5] = [0.2, 0.7, 0.1]
    blurred[:, 5:7] = [0.3, 0.1]
    snapped = layout.snap(blurred)
    assert (snapped[:, 2:5] == [0, 1, 0]).all() and (snapped[:, 5:7] == 0).all()


def test_targets():
    labels = np.array([0] * 50 + [1] * 30 + [2] * 10 + [3] * 1)
    assert oversample.targets(labels, "median") == {2: 20}        # class 3 has one row: left alone
    assert oversample.targets(labels, "full") == {1: 50, 2: 50}
    assert oversample.targets(np.array([0] * 50 + [1] * 30 + [2] * 10 + [3] * 4), "two smallest") == {2: 20, 3: 20}


@pytest.mark.parametrize("method", [m for m in oversample.METHODS if m != "CTGAN"])
def test_every_method_keeps_real_rows_and_gives_both_labels(method):
    X, male, age = toy()
    X_out, male_out, age_out, record = oversample.resample(method, X, male, age, 4, NAMES, seed=1)
    n = len(X)
    assert np.array_equal(X_out[:n], X) and np.array_equal(male_out[:n], male) and np.array_equal(age_out[:n], age)
    assert len(X_out) == len(male_out) == len(age_out) == n + record["rows_added"]
    assert set(np.unique(male_out)) <= {0, 1}
    new = X_out[n:]
    if len(new):
        assert ((new[:, 2:5].sum(axis=1) == 1)).all()                  # exactly one brand per new row
        assert set(np.unique(new[:, 2:7])) <= {0.0, 1.0}
    if method == "none":
        assert record["rows_added"] == 0
    else:
        assert record["rows_added"] > 0
    # random oversampling and SMOTE-NC reach the target exactly (ADASYN and Borderline-SMOTE only aim at it)
    if method in ("random oversampling", "SMOTE-NC", "SMOTE-NC, full balance"):
        before = np.bincount(male * 4 + age, minlength=8)
        after = np.bincount(male_out * 4 + age_out, minlength=8)
        goal = before.max() if method.endswith("full balance") else int(np.median(before[before > 0]))
        assert (after[before >= 2] >= goal).all()
        assert np.isclose(male_out.mean(), male.mean(), atol=0.05)     # gender balance barely moves


def test_ctgan_if_installed():
    pytest.importorskip("ctgan")
    X, male, age = toy()
    X_out, male_out, age_out, record = oversample.resample("CTGAN", X, male, age, 4, NAMES, seed=1, ctgan_epochs=2)
    assert record["rows_added"] > 0 and np.array_equal(X_out[:len(X)], X)


def test_fidelity_sees_copies():
    X, male, age = toy()
    y = male * 4 + age
    copies = X[y == 1][:20]
    result = oversample.fidelity(X, y, copies, np.ones(20, int))
    assert result["distance_ratio"] == 0.0


def test_paired_difference():
    r = step2.paired([0.30, 0.32, 0.31, 0.29, 0.33], [0.28, 0.29, 0.30, 0.27, 0.30])
    assert np.isclose(r["difference"], 0.022) and r["low"] < 0.022 < r["high"] and r["seeds_higher"] == 5


def test_step2_end_to_end(tmp_path):
    raw = small_lat(n=600)
    prepared = step2.Prepared(raw)
    for method in ("none", "SMOTE-NC", "SMOTE, earlier setting"):
        result = step2.run_method(prepared, "6", method, tmp_path, seeds=(42, 7), epochs=3)
        assert len(result["rows_added"]) == 2
    assert step2.chosen_attributes(prepared, tmp_path) == 2
    summary = step2.combine(tmp_path)
    assert summary["parts"] == 3
    paired = pd.read_csv(tmp_path / "paired.csv")
    assert set(paired["method"]) == {"SMOTE-NC", "SMOTE, earlier setting"}
    scores = pd.read_csv(tmp_path / "scores.csv")
    assert "recall <18" in scores.columns and scores.query("task == 'age'")["rare_recall"].notna().all()
