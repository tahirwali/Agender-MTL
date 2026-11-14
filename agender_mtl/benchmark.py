"""
The benchmark: one multi-task network against single-task baselines, on LAT.

Preprocessing: the 29 raw LAT attributes; log(1+x) and standardisation for skewed numbers, one-hot
encoding for columns with few values and frequency encoding for columns with many, missing values as their
own category, all fitted on the training rows only. Split: 70 / 15 / 15, stratified by the six age groups.
Then QUBO attribute selection, the multi-task network, Logistic Regression, Random Forest and Gradient
Boosting, under four age schemes (3, 6, 10 and 48 classes), and a sensitivity sweep of the QUBO weights.
"""
import time

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from agender_mtl import metrics, qubo
from agender_mtl.mtl import MultiTaskNet

SEEDS = (42, 7, 123, 2024, 99)
AGE_EDGES = [-np.inf, 17, 25, 35, 45, 60, np.inf]
AGE_NAMES = ["<18", "18-25", "26-35", "36-45", "46-60", "60+"]
LABELS = ("AGE", "GENDER")
ONE_HOT_LIMIT = 10          # a categorical column with at most this many values is one-hot encoded


# ---- age-class schemes ----

def age_classes(age, scheme):
    age = np.asarray(age)
    if scheme == "3":                                   # Al-Zuabi et al.: <28, 28-39, 40+
        return np.digitize(age, [28, 40])
    if scheme == "6":                                   # <18, 18-25, 26-35, 36-45, 46-60, 60+
        return pd.cut(pd.Series(age), AGE_EDGES, labels=AGE_NAMES).cat.codes.to_numpy()
    if scheme == "10":                                  # Kuang et al.: <15, 15-19, ..., 50-54, 55+
        return np.digitize(age, list(range(15, 60, 5)))
    if scheme == "48":                                  # one class per year, 13 ... 60+
        return np.clip(age, 13, 60) - 13
    raise ValueError(scheme)


N_CLASSES = {"3": 3, "6": 6, "10": 10, "48": 48}
TOLERANCE = {"48": 5}                                   # the 48-class scheme is also scored within +-5 years


# ---- preprocessing ----

class Encoder:
    """Fitted on the training rows only, then applied unchanged to validation and test rows."""

    def fit(self, table):
        self.numeric, self.one_hot, self.frequency = [], {}, {}
        self.log, self.mean, self.std = {}, {}, {}
        for column in table.columns:
            if column in LABELS:
                continue
            values = table[column]
            if is_numeric_dtype(values):
                self.numeric.append(column)
                use_log = bool(values.min() >= 0 and values.skew() > 1)
                x = np.log1p(values) if use_log else values.astype(float)
                self.log[column], self.mean[column] = use_log, x.mean()
                self.std[column] = x.std() or 1.0
            else:
                values = values.fillna("missing").astype(str)
                if values.nunique() <= ONE_HOT_LIMIT:
                    self.one_hot[column] = sorted(values.unique())
                else:
                    self.frequency[column] = values.value_counts(normalize=True).to_dict()
        return self

    def transform(self, table):
        parts, names = [], []
        for column in self.numeric:
            x = table[column].astype(float)
            x = np.log1p(x) if self.log[column] else x
            parts.append(((x - self.mean[column]) / self.std[column]).to_numpy())
            names.append(column)
        for column, categories in self.one_hot.items():
            values = table[column].fillna("missing").astype(str)
            for category in categories:
                parts.append((values == category).astype(float).to_numpy())
                names.append(f"{column}={category}")
        for column, share in self.frequency.items():
            values = table[column].fillna("missing").astype(str)
            parts.append(values.map(share).fillna(0.0).astype(float).to_numpy())
            names.append(f"{column}_freq")
        self.names = names
        return np.column_stack(parts)


def split(raw, seed):
    """70 / 15 / 15, stratified by the six age groups."""
    strata = age_classes(raw["AGE"], "6")
    rows = np.arange(len(raw))
    train, rest = train_test_split(rows, test_size=0.30, stratify=strata, random_state=seed)
    validation, test = train_test_split(rest, test_size=0.50, stratify=strata[rest], random_state=seed)
    return train, validation, test


def prepare(raw, seed):
    train, validation, test = split(raw, seed)
    encoder = Encoder().fit(raw.iloc[train])
    X = encoder.transform(raw)
    return {"X": X, "names": encoder.names, "train": train, "validation": validation, "test": test,
            "male": (raw["GENDER"] == "M").astype(int).to_numpy(), "age": raw["AGE"].to_numpy()}


# ---- models ----

def baselines(scheme, seed):
    many_classes = N_CLASSES[scheme] > 12
    boosting = (HistGradientBoostingClassifier(random_state=seed) if many_classes
                else GradientBoostingClassifier(random_state=seed))     # histogram version: 48 classes are too many for the plain one
    return {"Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
            "Gradient Boosting": boosting}


def full_probabilities(model, X, n_classes):
    """predict_proba with a column for every class, also for classes missing from the training rows."""
    out = np.zeros((len(X), n_classes))
    out[:, model.classes_] = model.predict_proba(X)
    return out


def score_age(probabilities, labels, scheme):
    return metrics.age_scores(probabilities, labels, tolerance=TOLERANCE.get(scheme, 1))


def run_scheme(data, scheme, seed, k=25, alpha=1.0, beta=0.05, lam=1.5, epochs=150, with_baselines=True):
    """One scheme, one seed. Returns the score rows and the chosen attributes."""
    X, train, validation, test = data["X"], data["train"], data["validation"], data["test"]
    n_age = N_CLASSES[scheme]
    age = age_classes(data["age"], scheme)
    male = data["male"]
    joint = male * n_age + age                                      # one joint label: gender x age group
    chosen, terms = qubo.select(X[train], joint[train], k, alpha, beta, lam, seed=seed)
    Xs = X[:, chosen]

    rows = []
    started = time.time()
    net = MultiTaskNet(Xs.shape[1], n_age, seed=seed)
    net.fit(Xs[train], male[train], age[train], Xs[validation], male[validation], age[validation],
            n_age, epochs=epochs)
    for part, index in (("validation", validation), ("test", test)):
        p_male, p_age = net.predict(Xs[index])
        rows.append({"model": "MTL", "part": part, "task": "gender", **metrics.gender_scores(p_male, male[index])})
        rows.append({"model": "MTL", "part": part, "task": "age", **score_age(p_age, age[index], scheme)})
    mtl_seconds = time.time() - started

    if with_baselines:
        for name, model in baselines(scheme, seed).items():
            for task, target, n in (("gender", male, 2), ("age", age, n_age)):
                fitted = model.fit(Xs[train], target[train])
                p = full_probabilities(fitted, Xs[test], n)
                scores = (metrics.gender_scores(p[:, 1], target[test]) if task == "gender"
                          else score_age(p, target[test], scheme))
                rows.append({"model": name, "part": "test", "task": task, **scores})

    for row in rows:
        row.update(scheme=scheme, seed=seed)
    info = {"scheme": scheme, "seed": seed, "chosen": [data["names"][i] for i in chosen],
            "qubo_energy": terms["energy"], "mtl_best_epoch": net.best_epoch, "mtl_seconds": round(mtl_seconds, 1),
            "relevance": dict(zip(data["names"], np.round(terms["relevance"], 5)))}
    history = pd.DataFrame(net.history, columns=["train_loss", "validation_loss"]).assign(scheme=scheme, seed=seed)
    return rows, info, history


# ---- QUBO sensitivity: one coefficient at a time, validation age macro-F1 (6 classes) ----

DEFAULTS = {"k": 25, "beta": 0.05, "lam": 1.5, "alpha": 1.0}
SWEEP = {"k": [15, 20, 25, 30, 35], "beta": [0.0, 0.05, 0.10, 0.20], "lam": [0.5, 1.5, 3.0], "alpha": [0.5, 1.0, 2.0]}


def sweep_settings():
    seen, settings = set(), []
    for name, values in SWEEP.items():
        for value in values:
            setting = {**DEFAULTS, name: value}
            key = tuple(sorted(setting.items()))
            if key not in seen:
                seen.add(key)
                settings.append(setting)
    return settings


def run_sweep(data, seed, epochs=150):
    rows = []
    for setting in sweep_settings():
        scores, _, _ = run_scheme(data, "6", seed, epochs=epochs, with_baselines=False, **setting)
        val_age = next(r for r in scores if r["part"] == "validation" and r["task"] == "age")
        rows.append({**setting, "seed": seed, "validation_macro_f1": val_age["macro_f1"]})
    return rows
