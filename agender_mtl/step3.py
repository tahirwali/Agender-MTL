"""
Step 3 driver: the age part of the pipeline on a second operator's data (Iranian Churn).

Same steps as on LAT, with the changes this table needs: no gender, so the network trains its age head
alone (gender weight 0) and the QUBO relevance is measured against the age group; 11 attributes instead of
29, so the number kept by the QUBO (k) is chosen on the validation rows of each seed, from 4, 6, 8, 10 and
all 11. Then the oversampling methods of Step 2 are compared in the same way (without gender, the joint
label is the age group itself).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agender_mtl import benchmark, churn, metrics, oversample, qubo, step2
from agender_mtl.mtl import MultiTaskNet

K_GRID = (4, 6, 8, 10, 11)
N_AGE = 5
METHODS = [m for m in oversample.METHODS if m != "SMOTE-NC, age label only"]   # the same as SMOTE-NC here


def prepare(table, seed):
    rows = np.arange(len(table))
    labels = table["AGE"].to_numpy()
    train, rest = train_test_split(rows, test_size=0.30, stratify=labels, random_state=seed)
    validation, test = train_test_split(rest, test_size=0.50, stratify=labels[rest], random_state=seed)
    encoder = benchmark.Encoder().fit(table.iloc[train])
    return {"X": encoder.transform(table), "names": encoder.names, "age": labels,
            "train": train, "validation": validation, "test": test}


def network(X_fit, age_fit, X_val, age_val, seed, epochs):
    net = MultiTaskNet(X_fit.shape[1], N_AGE, seed=seed, gender_weight=0.0)
    zeros = np.zeros(len(X_fit), int)
    return net.fit(X_fit, zeros, age_fit, X_val, np.zeros(len(X_val), int), age_val, N_AGE, epochs=epochs)


def choose_k(data, seed, epochs):
    X, age, train, validation = data["X"], data["age"], data["train"], data["validation"]
    scores = {}
    for k in K_GRID:
        k = min(k, X.shape[1])
        chosen, _ = qubo.select(X[train], age[train], k, seed=seed)
        net = network(X[train][:, chosen], age[train], X[validation][:, chosen], age[validation], seed, epochs)
        scores[k] = metrics.age_scores(net.predict(X[validation][:, chosen])[1], age[validation])["macro_f1"]
    best = max(scores, key=scores.get)
    chosen, _ = qubo.select(X[train], age[train], best, seed=seed)
    return best, chosen, scores


def score(probabilities, labels, rare):
    row = metrics.age_scores(probabilities, labels)
    recalls = [float(np.mean(probabilities.argmax(1)[labels == c] == c)) for c in range(N_AGE)]
    for c in range(N_AGE):
        row[f"recall {churn.GROUP_NAMES[c]}"] = recalls[c]
    row["rare_recall"] = float(np.mean([recalls[c] for c in rare]))
    return row


def run_seed(table, seed, epochs=150, methods=METHODS, ctgan_epochs=100):
    data = prepare(table, seed)
    X, age, train, validation, test = data["X"], data["age"], data["train"], data["validation"], data["test"]
    k, chosen, k_scores = choose_k(data, seed, epochs)
    Xs = X[:, chosen]
    names = [data["names"][i] for i in chosen]
    rare = step2.rare_classes(age[train], N_AGE)
    rows, records = [], []

    for name, model in benchmark.baselines("6", seed).items():           # the 6-class settings: few classes
        fitted = model.fit(Xs[train], age[train])
        p = benchmark.full_probabilities(fitted, Xs[test], N_AGE)
        rows.append({"model": name, "method": "none", **score(p, age[test], rare)})

    for method in methods:
        no_gender = np.zeros(len(train), int)
        X_fit, _, age_fit, record = oversample.resample(method, Xs[train], no_gender, age[train], N_AGE, names, seed,
                                                        ctgan_epochs=ctgan_epochs)
        if record["rows_added"]:
            record.update(oversample.fidelity(Xs[train], age[train], X_fit[len(train):], age_fit[len(train):], seed))
        net = network(X_fit, age_fit, Xs[validation], age[validation], seed, epochs)
        rows.append({"model": "MTL (age head)", "method": method, **score(net.predict(Xs[test])[1], age[test], rare)})
        if method != "none":                      # Gradient Boosting without new rows is already among the baselines
            booster = benchmark.baselines("6", seed)["Gradient Boosting"].fit(X_fit, age_fit)
            rows.append({"model": "Gradient Boosting", "method": method,
                         **score(benchmark.full_probabilities(booster, Xs[test], N_AGE), age[test], rare)})
        records.append({**record, "seed": seed})
    for row in rows:
        row["seed"] = seed
    info = {"seed": seed, "k": k, "validation_macro_f1_per_k": k_scores, "chosen": names,
            "rare": [churn.GROUP_NAMES[c] for c in rare]}
    return rows, records, info


def run(table, out_dir, seeds=benchmark.SEEDS, epochs=150, methods=METHODS, ctgan_epochs=100):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, records, infos = [], [], []
    for seed in seeds:
        r, rec, info = run_seed(table, seed, epochs, methods, ctgan_epochs)
        rows += r
        records += rec
        infos.append(info)
        pd.DataFrame(rows).to_csv(out_dir / "scores.csv", index=False)          # rewritten after every seed
    pd.DataFrame(records).to_csv(out_dir / "records.csv", index=False)
    with open(out_dir / "qubo_choice.json", "w") as f:
        json.dump(infos, f, indent=1, default=float)
    scores = pd.DataFrame(rows)
    measures = ["accuracy", "within_1", "macro_f1", "weighted_f1", "balanced", "auc_ovr", "ece", "rare_recall"]
    means = scores.groupby(["model", "method"])[measures].agg(["mean", "std"])
    means.columns = [f"{a} {b}" for a, b in means.columns]
    means.reset_index().to_csv(out_dir / "means.csv", index=False)

    comparisons = []
    for model, group in scores.groupby("model"):
        base = group[group["method"] == "none"].set_index("seed")
        if base.empty:
            continue
        for method, other in group[group["method"] != "none"].groupby("method"):
            other = other.set_index("seed")
            for measure in ("macro_f1", "balanced", "rare_recall", "ece"):
                comparisons.append({"model": model, "method": method, "measure": measure,
                                    **step2.paired(other[measure], base.loc[other.index, measure])})
    pd.DataFrame(comparisons).to_csv(out_dir / "paired.csv", index=False)
    headline = means.reset_index()
    return {"k_chosen": [i["k"] for i in infos],
            "test_means": {f"{r['model']} / {r['method']}": {m: round(r[f"{m} mean"], 3) for m in ("macro_f1", "balanced", "rare_recall", "ece")}
                           for _, r in headline.iterrows()}}
