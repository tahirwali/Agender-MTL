"""Offline checks: metrics, QUBO, the multi-task network and the benchmark pipeline."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agender_mtl import benchmark, data, metrics, qubo, step1
from agender_mtl.mtl import MultiTaskNet, class_weights


def test_qubo_matrix_gives_the_energy():
    rng = np.random.default_rng(0)
    d, k, alpha, beta, lam = 6, 3, 1.0, 0.05, 1.5
    relevance = rng.random(d)
    redundancy = rng.random((d, d))
    redundancy = (redundancy + redundancy.T) / 2
    matrix, constant = qubo.qubo_matrix(relevance, redundancy, k, alpha, beta, lam)
    for bits in itertools.product([0, 1], repeat=d):
        x = np.array(bits)
        pairs = sum(redundancy[i, j] * x[i] * x[j] for i in range(d) for j in range(i + 1, d))
        expected = -alpha * relevance @ x + beta * pairs + lam * (x.sum() - k) ** 2
        assert np.isclose(qubo.energy(x, matrix, constant), expected)


def test_annealing_finds_the_best_subset_of_a_small_problem():
    rng = np.random.default_rng(1)
    d = 8
    relevance, redundancy = rng.random(d), rng.random((d, d)) * 0.3
    redundancy = (redundancy + redundancy.T) / 2
    matrix, constant = qubo.qubo_matrix(relevance, redundancy, 3)
    best = min(qubo.energy(np.array(b), matrix, constant) for b in itertools.product([0, 1], repeat=d))
    x, e = qubo.anneal(matrix, constant, restarts=2, iterations=3000, seed=0)
    assert np.isclose(e, best) and np.isclose(qubo.energy(x, matrix, constant), e) and x.sum() == 3


def test_relevance_finds_the_informative_column():
    rng = np.random.default_rng(2)
    label = rng.integers(0, 3, 600)
    X = np.column_stack([label + rng.normal(0, 0.3, 600), rng.normal(size=600), rng.normal(size=600)])
    chosen, _ = qubo.select(X, label, k=1, iterations=2000, restarts=1)
    assert list(chosen) == [0]


def test_metrics():
    p = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    y = np.array([0, 1, 1, 2])
    scores = metrics.age_scores(p, y)
    assert scores["accuracy"] == 0.5 and scores["within_1"] == 0.75
    # perfectly calibrated toy: confidence 1 and always right -> ECE 0
    assert metrics.ece(np.eye(3)[[0, 1, 2]], [0, 1, 2]) == 0.0
    g = metrics.gender_scores([0.9, 0.2, 0.6, 0.4], [1, 0, 1, 1])
    assert g["accuracy"] == 0.75 and g["auc"] == 1.0


def test_age_schemes_use_the_stated_bins():
    ages = np.array([13, 14, 15, 27, 28, 39, 40, 54, 55, 60, 65])
    assert list(benchmark.age_classes(ages, "3")) == [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    assert list(benchmark.age_classes(ages, "10")) == [0, 0, 1, 3, 3, 5, 6, 8, 9, 9, 9]
    assert list(benchmark.age_classes(ages, "48")) == [0, 1, 2, 14, 15, 26, 27, 41, 42, 47, 47]
    assert benchmark.age_classes(np.array([17, 18, 60, 61]), "6").tolist() == [0, 1, 4, 5]


def test_class_weights_soften_inverse_frequency():
    labels = np.array([0] * 90 + [1] * 10)
    w = class_weights(labels, 2)
    assert np.isclose(w[1] / w[0], 3.0)                 # sqrt(90 / 10)
    assert np.isclose(w[labels].mean(), 1.0)


def test_network_learns_a_simple_problem():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(800, 4))
    gender = (X[:, 0] > 0).astype(int)
    age = np.digitize(X[:, 1], [-0.5, 0.5])
    net = MultiTaskNet(4, 3, seed=0).fit(X[:600], gender[:600], age[:600], X[600:], gender[600:], age[600:], 3, epochs=150)
    p_gender, p_age = net.predict(X[600:])
    assert ((p_gender > 0.5) == gender[600:]).mean() > 0.9
    assert (p_age.argmax(1) == age[600:]).mean() > 0.8


def small_lat(n=400, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(13, 66, n)
    table = pd.DataFrame({"AGE": age, "GENDER": rng.choice(["F", "M"], n),
                          "PAYMENT_METHOD": rng.choice([f"PMethod {i}" for i in range(15)], n),
                          "DEVICE_CAPABILITY": rng.choice(["3G", "LTE", None], n),
                          "RECHARGES": rng.poisson(3, n), "INTERNET_MB": rng.exponential(100, n),
                          "MADE_CALLS": rng.poisson(20 + age, n)})
    for i in range(4):
        table[f"NOISE_{i}"] = rng.normal(size=n)
    return table


def test_encoder_uses_only_training_rows():
    raw = small_lat()
    prepared = benchmark.prepare(raw, 42)
    names = prepared["names"]
    assert "DEVICE_CAPABILITY=missing" in names and "PAYMENT_METHOD_freq" in names   # missing kept; 15 values -> frequency
    train = prepared["train"]
    numeric = names.index("MADE_CALLS")
    assert np.isclose(prepared["X"][train, numeric].mean(), 0, atol=1e-9)          # standardised on train only
    sizes = [len(prepared[p]) for p in ("train", "validation", "test")]
    assert sizes == [280, 60, 60]


def test_benchmark_runs_end_to_end(tmp_path):
    raw = small_lat()
    result = step1.run_scheme_all_seeds(raw, "6", tmp_path, seeds=(42,), epochs=3)
    assert "age / MTL" in result["test_means"] and (tmp_path / "scores_6class.csv").exists()
    assert (tmp_path / "mtl_loss_6class.csv").exists() and (tmp_path / "qubo_6class.json").exists()


def test_lat_description_counts_groups():
    summary = data.describe_lat(small_lat())
    assert summary["rows"] == 400 and sum(summary["six_groups"].values()) == 400
    assert summary["missing_cells"]["DEVICE_CAPABILITY"] > 0
