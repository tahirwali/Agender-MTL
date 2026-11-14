"""
Step 1 driver: runs the benchmark scheme by scheme and writes every result as soon as it exists,
so a run stopped by the time limit still leaves usable files.
"""
import json
from pathlib import Path

import pandas as pd

from agender_mtl import benchmark, data


def run_scheme_all_seeds(raw, scheme, out_dir, seeds=benchmark.SEEDS, epochs=150):
    out_dir = Path(out_dir)
    scores, infos, histories = [], [], []
    for seed in seeds:
        prepared = benchmark.prepare(raw, seed)
        rows, info, history = benchmark.run_scheme(prepared, scheme, seed, epochs=epochs)
        scores += rows
        infos.append(info)
        histories.append(history)
    pd.DataFrame(scores).to_csv(out_dir / f"scores_{scheme}class.csv", index=False)
    pd.concat(histories).to_csv(out_dir / f"mtl_loss_{scheme}class.csv", index_label="epoch")
    with open(out_dir / f"qubo_{scheme}class.json", "w") as f:
        json.dump(infos, f, indent=1, default=float)
    test = pd.DataFrame(scores).query("part == 'test'")
    means = {}
    for (task, model), group in test.groupby(["task", "model"]):
        values = group.drop(columns=["seed"]).mean(numeric_only=True).dropna().round(3)
        means[f"{task} / {model}"] = values.to_dict()
    return {"rows": len(scores), "mtl_best_epochs": [i["mtl_best_epoch"] for i in infos], "test_means": means}


def run_sweep_all_seeds(raw, out_dir, seeds=benchmark.SEEDS, epochs=150):
    rows = []
    for seed in seeds:
        rows += benchmark.run_sweep(benchmark.prepare(raw, seed), seed, epochs=epochs)
    table = pd.DataFrame(rows)
    table.to_csv(Path(out_dir) / "qubo_sweep.csv", index=False)
    means = table.groupby(["k", "beta", "lam", "alpha"])["validation_macro_f1"].agg(["mean", "std"]).reset_index()
    means.to_csv(Path(out_dir) / "qubo_sweep_means.csv", index=False)
    return means.round(3).to_dict("records")


def raw_lat(lat_path):
    return data.read_lat(lat_path)
