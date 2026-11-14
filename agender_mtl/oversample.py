"""
Oversampling of the training rows, for the study of rare age groups.

Every method works on the attributes the QUBO step chose, after encoding, and only on the training rows.
Validation and test rows are never touched.

Which label is balanced. The network predicts gender and age together, so a new row needs both labels.
Most methods here balance the joint label (gender x age group): a synthetic row is made from rows of the
same gender and the same age group, so it carries both labels and the gender balance does not move.
Two methods balance the age label alone, as the earlier SMOTE study did; there the gender of a new row is
decided by its neighbours, like any other attribute.

How many rows are added. "median": every class smaller than the median class is raised to the median size.
"full": every class is raised to the size of the largest class.

One-hot attributes. A one-hot attribute (e.g. DEVICE_BRAND=Samsung) is one category of an original
column. SMOTE-NC and CTGAN see each original column as one categorical value. The methods that only know
numbers (SMOTE, Borderline-SMOTE, ADASYN) interpolate the 0/1 columns too; afterwards each new row gets
back exactly one category per column (the largest value), or none when no value reaches 0.5, which means
a category the QUBO step did not keep.
"""
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors

METHODS = {
    # name: (sampler, label, amount)
    "none": (None, None, None),
    "SMOTE, earlier setting": ("smote", "age", "two smallest"),
    "random oversampling": ("random", "joint", "median"),
    "SMOTE-NC": ("smotenc", "joint", "median"),
    "SMOTE-NC, age label only": ("smotenc", "age", "median"),
    "SMOTE-NC, full balance": ("smotenc", "joint", "full"),
    "Borderline-SMOTE": ("borderline", "joint", "median"),
    "ADASYN": ("adasyn", "joint", "median"),
    "CTGAN": ("ctgan", "joint", "median"),
}
NEIGHBOURS = 5


class Layout:
    """Which encoded columns are numbers and which are one-hot categories of the same original column."""

    def __init__(self, names):
        self.groups = {}
        for i, name in enumerate(names):
            if "=" in name:
                self.groups.setdefault(name.split("=", 1)[0], []).append(i)
        grouped = {i for members in self.groups.values() for i in members}
        self.numeric = [i for i in range(len(names)) if i not in grouped]
        self.width = len(names)

    def decode(self, X):
        """Numbers, then one code per original column (len(group) = a category that was not kept)."""
        codes = []
        for members in self.groups.values():
            block = X[:, members]
            code = block.argmax(axis=1)
            code[block.max(axis=1) < 0.5] = len(members)
            codes.append(code)
        return X[:, self.numeric], (np.column_stack(codes) if codes else np.zeros((len(X), 0), int))

    def encode(self, numbers, codes):
        X = np.zeros((len(numbers), self.width))
        X[:, self.numeric] = numbers
        for j, members in enumerate(self.groups.values()):
            for position, column in enumerate(members):
                X[:, column] = (codes[:, j] == position).astype(float)
        return X

    def snap(self, X):
        """Give every row exactly one category per original column, or none."""
        numbers, codes = self.decode(X)
        return self.encode(numbers, codes)


def targets(labels, amount, n_classes=None):
    """Desired number of rows per class; only classes that grow are listed."""
    counts = np.bincount(labels, minlength=n_classes or labels.max() + 1)
    present = counts[counts > 0]
    if amount == "full":
        goal = {c: present.max() for c in range(len(counts))}
    elif amount == "median":
        goal = {c: int(np.median(present)) for c in range(len(counts))}
    elif amount == "two smallest":
        smallest = [c for c in np.argsort(counts) if counts[c] > 0][:2]
        goal = {c: int(np.median(present)) for c in smallest}
    else:
        raise ValueError(amount)
    # a class with a single row has no neighbour to interpolate with, so it is left as it is
    return {int(c): int(n) for c, n in goal.items() if 2 <= counts[c] < n}


def _neighbours(labels, wanted):
    smallest = min(np.sum(labels == c) for c in wanted)
    return max(1, min(NEIGHBOURS, smallest - 1))


def _imblearn(sampler, X, y, wanted, layout, seed):
    from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, BorderlineSMOTE, RandomOverSampler

    k = _neighbours(y, wanted)
    if sampler == "random":
        return RandomOverSampler(sampling_strategy=wanted, random_state=seed).fit_resample(X, y)
    if sampler == "smote":
        return SMOTE(sampling_strategy=wanted, k_neighbors=k, random_state=seed).fit_resample(X, y)
    if sampler == "borderline":
        return BorderlineSMOTE(sampling_strategy=wanted, k_neighbors=k, m_neighbors=10,
                               random_state=seed).fit_resample(X, y)
    if sampler == "adasyn":
        # ADASYN adds rows only near rare rows that have neighbours of other classes. A class without such
        # rows gets none, and imblearn then refuses the whole call, so the classes are raised one by one.
        new_rows, new_labels = [], []
        for c, goal in wanted.items():
            try:
                out, y_out = ADASYN(sampling_strategy={c: goal}, n_neighbors=k, random_state=seed).fit_resample(X, y)
            except (ValueError, RuntimeError):
                continue
            new_rows.append(out[len(X):])
            new_labels.append(y_out[len(X):])
        return np.vstack([X] + new_rows), np.concatenate([y] + new_labels)
    if sampler == "smotenc":
        numbers, codes = layout.decode(X[:, :layout.width])
        extra = X[:, layout.width:]                     # gender, when only the age label is balanced
        table = np.column_stack([numbers, codes, extra])
        categorical = list(range(numbers.shape[1], table.shape[1]))
        if not categorical:                             # nothing categorical: SMOTE-NC is plain SMOTE
            return SMOTE(sampling_strategy=wanted, k_neighbors=k, random_state=seed).fit_resample(X, y)
        out, y_out = SMOTENC(categorical_features=categorical, sampling_strategy=wanted, k_neighbors=k,
                             random_state=seed).fit_resample(table, y)
        n_numbers, n_codes = numbers.shape[1], codes.shape[1]
        X_out = layout.encode(out[:, :n_numbers], out[:, n_numbers:n_numbers + n_codes].astype(int))
        return np.column_stack([X_out, out[:, n_numbers + n_codes:]]), y_out
    raise ValueError(sampler)


def _ctgan(X, y, wanted, layout, seed, epochs):
    import pandas as pd
    from ctgan import CTGAN

    numbers, codes = layout.decode(X)
    frame = pd.DataFrame(numbers, columns=[f"n{i}" for i in range(numbers.shape[1])])
    discrete = []
    for j in range(codes.shape[1]):
        frame[f"c{j}"] = codes[:, j].astype(str)
        discrete.append(f"c{j}")
    frame["label"] = y.astype(str)
    try:
        model = CTGAN(epochs=epochs, batch_size=500, enable_gpu=False)
    except TypeError:                                   # older ctgan versions call the switch `cuda`
        model = CTGAN(epochs=epochs, batch_size=500, cuda=False)
    model.set_random_state(seed)
    model.fit(frame, discrete_columns=discrete + ["label"])

    new_rows, new_labels, matched, asked = [], [], 0, 0
    for c, goal in wanted.items():
        need = goal - int(np.sum(y == c))
        kept = []
        for _ in range(5):                              # keep rows whose generated label is the one asked for
            sample = model.sample(3 * need, condition_column="label", condition_value=str(c))
            kept.append(sample[sample["label"] == str(c)])
            if sum(len(k) for k in kept) >= need:
                break
        sample = pd.concat(kept).head(need)
        matched += len(sample)
        asked += need
        if len(sample) < need:                          # rarely: fill up with rows made for this label
            sample = pd.concat([sample, model.sample(need - len(sample), condition_column="label",
                                                     condition_value=str(c))])
        new_numbers = sample[[f"n{i}" for i in range(numbers.shape[1])]].to_numpy(float)
        new_codes = sample[discrete].astype(int).to_numpy() if discrete else np.zeros((need, 0), int)
        new_rows.append(layout.encode(new_numbers, new_codes))
        new_labels.append(np.full(need, c))
    X_out = np.vstack([X] + new_rows)
    y_out = np.concatenate([y] + new_labels)
    return X_out, y_out, {"ctgan_label_match": round(matched / max(asked, 1), 3)}


def resample(method, X, male, age, n_age, names, seed, ctgan_epochs=100):
    """Return the training rows with the synthetic rows appended, their two labels, and a record."""
    sampler, label, amount = METHODS[method]
    record = {"method": method, "rows_before": len(X), "male_share_before": round(float(male.mean()), 4)}
    if sampler is None:
        return X, male, age, {**record, "rows_added": 0, "seconds": 0.0, "male_share_after": record["male_share_before"]}
    layout = Layout(names)
    started = time.time()
    if label == "joint":
        y = male * n_age + age
        wanted = targets(y, amount, 2 * n_age)
        if sampler == "ctgan":
            X_out, y_out, extra = _ctgan(X, y, wanted, layout, seed, ctgan_epochs)
            record.update(extra)
        else:
            X_out, y_out = _imblearn(sampler, X, y, wanted, layout, seed)
        male_out, age_out = y_out // n_age, y_out % n_age
    else:                                               # age label only; gender travels as an attribute
        wanted = targets(age, amount, n_age)
        X_out, age_out = _imblearn(sampler, np.column_stack([X, male]), age, wanted, layout, seed)
        male_out = np.rint(X_out[:, -1]).clip(0, 1).astype(int)
        X_out = X_out[:, :-1]
    n = len(X)
    assert np.allclose(X_out[:n], X), "the original rows must come first and unchanged"
    if sampler in ("smote", "borderline", "adasyn"):
        X_out = np.vstack([X, layout.snap(X_out[n:])])
    record.update(rows_added=len(X_out) - n, seconds=round(time.time() - started, 1),
                  male_share_after=round(float(male_out.mean()), 4),
                  classes_raised=len(wanted))
    return X_out, male_out.astype(int), age_out.astype(int), record


# ---- how the synthetic rows relate to the real ones ----

def fidelity(X_real, y_real, X_new, y_new, seed=0):
    """Two plain checks of the synthetic rows, inside the classes that were raised.

    distance_ratio: median distance from a synthetic row to the closest real row of its class, divided by
    the median distance from a real row to the closest other real row of its class. Near 0 means copies;
    near 1 means new rows lie as close to the real ones as real rows lie to each other.
    discriminator_auc: how well gradient boosting tells synthetic rows from real rows of the same classes
    (3-fold cross-validated ROC-AUC). 0.5 means they cannot be told apart.
    """
    if len(X_new) < 10:
        return {"distance_ratio": np.nan, "discriminator_auc": np.nan}
    synthetic, real = [], []
    for c in np.unique(y_new):
        own = X_real[y_real == c]
        if len(own) < 2:
            continue
        finder = NearestNeighbors(n_neighbors=2).fit(own)
        real.append(finder.kneighbors(own)[0][:, 1])
        synthetic.append(finder.kneighbors(X_new[y_new == c], n_neighbors=1)[0][:, 0])
    ratio = float(np.median(np.concatenate(synthetic)) / max(np.median(np.concatenate(real)), 1e-12))
    same_classes = np.isin(y_real, np.unique(y_new))
    rows = np.vstack([X_real[same_classes], X_new])
    is_new = np.r_[np.zeros(same_classes.sum()), np.ones(len(X_new))]
    auc = cross_val_score(HistGradientBoostingClassifier(max_iter=100, random_state=seed), rows, is_new,
                          cv=3, scoring="roc_auc").mean()
    return {"distance_ratio": round(ratio, 4), "discriminator_auc": round(float(auc), 4)}
