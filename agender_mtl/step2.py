"""
Step 2 driver: does oversampling the training rows help the rare age groups?

For each age scheme and seed, the split and the QUBO choice are the same as in Step 1, and they are made
once, before any oversampling, so every method starts from the same rows and the same attributes. Then
each method adds synthetic training rows, the multi-task network and Gradient Boosting are trained on
them, and both are scored on the untouched test rows. "none" is the network as in Step 1 (class weights
only, no new rows).

Each (scheme, method) is written to its own file as soon as it is done, so a run stopped by the time limit
still leaves usable results; `combine` then builds the tables from whatever files exist.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import recall_score

from agender_mtl import benchmark, metrics, oversample, qubo
from agender_mtl.mtl import MultiTaskNet

SCHEMES = ("6", "10")
CLASS_NAMES = {
    "3": ["<28", "28-39", "40+"],
    "6": benchmark.AGE_NAMES,
    "10": ["<15"] + [f"{a}-{a + 4}" for a in range(15, 55, 5)] + ["55+"],
    "48": [str(a) for a in range(13, 60)] + ["60+"],
}
MEASURES = ["macro_f1", "balanced", "rare_recall", "accuracy", "within_1", "auc_ovr", "ece"]
GENDER_MEASURES = ["auc", "accuracy", "ece"]


class Prepared:
    """Split, encoding and QUBO choice per (scheme, seed), made once and shared by every method."""

    def __init__(self, raw):
        self.raw, self.data, self.chosen = raw, {}, {}

    def get(self, scheme, seed):
        if seed not in self.data:
            self.data[seed] = benchmark.prepare(self.raw, seed)
        data = self.data[seed]
        if (scheme, seed) not in self.chosen:
            age = benchmark.age_classes(data["age"], scheme)
            joint = data["male"] * benchmark.N_CLASSES[scheme] + age
            train = data["train"]
            chosen, _ = qubo.select(data["X"][train], joint[train], **benchmark.DEFAULTS, seed=seed)
            self.chosen[(scheme, seed)] = chosen
        return data, self.chosen[(scheme, seed)]


def rare_classes(age_train, n_age):
    """The two smallest age groups in the training rows."""
    counts = np.bincount(age_train, minlength=n_age)
    return [int(c) for c in np.argsort(counts) if counts[c] > 0][:2]


def age_row(probabilities, labels, scheme, rare):
    n_age = probabilities.shape[1]
    row = benchmark.score_age(probabilities, labels, scheme)
    predicted = probabilities.argmax(axis=1)
    recalls = recall_score(labels, predicted, labels=range(n_age), average=None, zero_division=0)
    names = CLASS_NAMES[scheme]
    for c in range(n_age):
        row[f"recall {names[c]}"] = recalls[c]
    row["rare_recall"] = float(np.mean(recalls[rare]))
    row["predicted_rare_share"] = float(np.isin(predicted, rare).mean())
    return row


def run_one(prepared, scheme, method, seed, epochs=150, ctgan_epochs=100):
    data, chosen = prepared.get(scheme, seed)
    n_age = benchmark.N_CLASSES[scheme]
    names = [data["names"][i] for i in chosen]
    X = data["X"][:, chosen]
    train, validation, test = data["train"], data["validation"], data["test"]
    age = benchmark.age_classes(data["age"], scheme)
    male = data["male"]
    rare = rare_classes(age[train], n_age)

    X_fit, male_fit, age_fit, record = oversample.resample(method, X[train], male[train], age[train], n_age,
                                                           names, seed, ctgan_epochs=ctgan_epochs)
    n = len(train)
    if record["rows_added"]:
        sampler, label, _ = oversample.METHODS[method]
        y_real = male[train] * n_age + age[train] if label == "joint" else age[train]
        y_new = male_fit[n:] * n_age + age_fit[n:] if label == "joint" else age_fit[n:]
        record.update(oversample.fidelity(X[train], y_real, X_fit[n:], y_new, seed=seed))
    added = np.bincount(age_fit[n:], minlength=n_age)
    record.update(scheme=scheme, seed=seed, rare=[CLASS_NAMES[scheme][c] for c in rare],
                  added_per_age_group={CLASS_NAMES[scheme][c]: int(added[c]) for c in range(n_age) if added[c]})

    rows = []
    net = MultiTaskNet(X.shape[1], n_age, seed=seed)
    net.fit(X_fit, male_fit, age_fit, X[validation], male[validation], age[validation], n_age, epochs=epochs)
    p_male, p_age = net.predict(X[test])
    rows.append({"model": "MTL", "task": "gender", **metrics.gender_scores(p_male, male[test])})
    rows.append({"model": "MTL", "task": "age", **age_row(p_age, age[test], scheme, rare)})
    record["mtl_best_epoch"] = net.best_epoch

    booster = benchmark.baselines(scheme, seed)["Gradient Boosting"]
    fitted = booster.fit(X_fit, male_fit)
    rows.append({"model": "Gradient Boosting", "task": "gender",
                 **metrics.gender_scores(benchmark.full_probabilities(fitted, X[test], 2)[:, 1], male[test])})
    fitted = booster.fit(X_fit, age_fit)
    rows.append({"model": "Gradient Boosting", "task": "age",
                 **age_row(benchmark.full_probabilities(fitted, X[test], n_age), age[test], scheme, rare)})
    for row in rows:
        row.update(scheme=scheme, method=method, seed=seed)
    return rows, record


def safe_name(text):
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()


def run_method(prepared, scheme, method, out_dir, seeds=benchmark.SEEDS, epochs=150, ctgan_epochs=100):
    """One scheme and one method over every seed; writes scores and records at once."""
    out_dir = Path(out_dir)
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    rows, records = [], []
    for seed in seeds:
        r, record = run_one(prepared, scheme, method, seed, epochs, ctgan_epochs)
        rows += r
        records.append(record)
    stem = f"{scheme}class_{safe_name(method)}"
    pd.DataFrame(rows).to_csv(out_dir / "parts" / f"scores_{stem}.csv", index=False)
    with open(out_dir / "parts" / f"records_{stem}.json", "w") as f:
        json.dump(records, f, indent=1, default=float)
    table = pd.DataFrame(rows).query("task == 'age'").groupby("model")[["macro_f1", "rare_recall", "ece"]].mean()
    return {"rows_added": [r["rows_added"] for r in records], "test_age_means": table.round(3).to_dict("index")}


def chosen_attributes(prepared, out_dir):
    """The attributes QUBO chose for each (scheme, seed); the same for every method."""
    table = {f"{scheme}class_seed{seed}": [prepared.data[seed]["names"][i] for i in chosen]
             for (scheme, seed), chosen in prepared.chosen.items()}
    with open(Path(out_dir) / "qubo_chosen.json", "w") as f:
        json.dump(table, f, indent=1)
    return len(table)


# ---- tables from whatever parts exist ----

def paired(values, reference):
    """Method minus "none" over the same seeds: mean, 95 % t interval, paired t-test, seeds better/worse."""
    d = np.asarray(values, float) - np.asarray(reference, float)
    n = len(d)
    mean = d.mean()
    if n < 2 or np.allclose(d, d[0]):
        return {"difference": mean, "low": mean, "high": mean, "p": np.nan,
                "seeds_higher": int((d > 0).sum()), "seeds_lower": int((d < 0).sum())}
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return {"difference": mean, "low": mean - half, "high": mean + half, "p": stats.ttest_rel(values, reference).pvalue,
            "seeds_higher": int((d > 0).sum()), "seeds_lower": int((d < 0).sum())}


def combine(out_dir):
    out_dir = Path(out_dir)
    parts = sorted((out_dir / "parts").glob("scores_*.csv"))
    if not parts:
        return {"note": "no parts written"}
    scores = pd.concat([pd.read_csv(p, dtype={"scheme": str}) for p in parts], ignore_index=True)
    scores.to_csv(out_dir / "scores.csv", index=False)
    records = []
    for p in sorted((out_dir / "parts").glob("records_*.json")):
        records += json.loads(p.read_text())
    pd.DataFrame(records).to_csv(out_dir / "records.csv", index=False)

    means = scores.drop(columns="seed").groupby(["scheme", "task", "model", "method"]).agg(["mean", "std"])
    means.columns = [f"{a} {b}" for a, b in means.columns]
    means = means.dropna(axis=1, how="all").reset_index()
    means.to_csv(out_dir / "means.csv", index=False)

    comparisons = []
    for (scheme, task, model), group in scores.groupby(["scheme", "task", "model"]):
        base = group[group["method"] == "none"].set_index("seed")
        if base.empty:
            continue
        for method, other in group[group["method"] != "none"].groupby("method"):
            other = other.set_index("seed")
            seeds = base.index.intersection(other.index)
            for measure in (MEASURES if task == "age" else GENDER_MEASURES):
                comparisons.append({"scheme": scheme, "task": task, "model": model, "method": method,
                                    "measure": measure, "seeds": len(seeds),
                                    **paired(other.loc[seeds, measure], base.loc[seeds, measure])})
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(out_dir / "paired.csv", index=False)

    headline = {}
    age = means[means["task"] == "age"]
    for _, row in age.iterrows():
        headline[f"{row['scheme']}class / {row['model']} / {row['method']}"] = {
            m: round(row[f"{m} mean"], 3) for m in ("macro_f1", "balanced", "rare_recall", "ece")}
    return {"parts": len(parts), "age_means": headline}
