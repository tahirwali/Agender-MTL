"""Step F on small synthetic outputs of Steps 1-3."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agender_mtl import benchmark, figures, step1, step2, step3
from test_benchmark import small_lat


def test_figures_from_saved_outputs(tmp_path, monkeypatch):
    raw = small_lat(n=500)
    (tmp_path / "s1").mkdir()
    monkeypatch.setattr(benchmark, "SWEEP", {"k": [4, 6], "beta": [0.05], "lam": [1.5], "alpha": [1.0]})
    for scheme in ("6", "10"):
        step1.run_scheme_all_seeds(raw, scheme, tmp_path / "s1", seeds=(42, 7), epochs=3)
    step1.run_sweep_all_seeds(raw, tmp_path / "s1", seeds=(42, 7), epochs=3)
    prepared = step2.Prepared(raw)
    for scheme in ("6", "10"):
        for method in ("none", "SMOTE-NC"):
            step2.run_method(prepared, scheme, method, tmp_path / "s2", seeds=(42, 7), epochs=3)
    step2.combine(tmp_path / "s2")
    rng = np.random.default_rng(0)
    table = pd.DataFrame({"AGE": rng.integers(0, 5, 300), "A": rng.normal(size=300), "B": rng.poisson(4, 300)})
    step3.run(table, tmp_path / "s3", seeds=(42,), epochs=3, methods=["none", "SMOTE-NC"])

    numbers = figures.run(raw, tmp_path / "s1", tmp_path / "s2", tmp_path / "s3", tmp_path / "out", epochs=3)
    out = tmp_path / "out"
    for name in ("fig_data_distribution", "fig_correlation", "fig_loss_6class", "fig_qubo_sweep", "fig_confusion_age_6class",
                 "fig_confusion_gender_6class", "fig_roc_age_10class", "fig_qubo_relevance", "fig_importance",
                 "fig_oversampling_recall_10class", "fig_oversampling_paired"):
        assert (out / "figures" / f"{name}.pdf").exists(), name
    for name in ("table_lat_6class", "table_ece", "table_schemes", "table_sweep", "table_oversampling",
                 "table_oversampling_paired_6class", "table_synthetic_rows", "table_iranian"):
        assert (out / "tables" / f"{name}.tex").read_text().startswith(r"\begin{table"), name
    assert r"$\pm$" in (out / "tables" / "table_lat_6class.tex").read_text()
    assert json.loads((out / "numbers.json").read_text())["data"]["rows"] == 500
    assert numbers["step3"]["k_chosen"]
