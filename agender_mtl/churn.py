"""
The Iranian Churn dataset (UCI 563, CC BY 4.0), used as an external check of the age part of the pipeline.

3,150 customers of an Iranian operator, aggregated over nine months: call failures, complaints,
subscription length, charge amount, seconds of use, frequency of use and of SMS, distinct called numbers,
tariff plan, status, customer value, churn. Age comes in five groups.

Two facts shape the use of this table:
- the column `Age` holds one value per age group (15, 25, 30, 45, 55), so it is the label written as a
  number and is dropped;
- some rows are exact copies of others; a copy in the training rows and its twin in the test rows would
  make the test look easier than it is, so copies are removed before splitting.
- `Customer Value` is a linear formula of the usage columns whose coefficients differ between age groups
  (inside each group the formula fits exactly), so it also carries the label and is dropped;
  `formula_check` measures this.
There is no gender column, so only age is predicted here.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import radius_neighbors_graph

from agender_mtl import data

LEAKING = ["AGE", "CUSTOMER_VALUE"]                      # after renaming: the age value and the customer value

URL = "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip"
MEMBER = "Customer Churn.csv"
GROUP_NAMES = ["group 1 (15)", "group 2 (25)", "group 3 (30)", "group 4 (45)", "group 5 (55)"]


def fetch(folder):
    return data.download(URL, Path(folder) / "iranian_churn.zip")


def read_raw(path):
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            return pd.read_csv(io.BytesIO(z.read(MEMBER)))
    return pd.read_csv(path)


def tidy(raw):
    table = raw.copy()
    table.columns = [" ".join(c.split()).upper().replace(" ", "_") for c in table.columns]
    return table


def clean(raw):
    """Drop `Age` and `Customer Value`, remove exact copies; the label becomes AGE = group - 1 (0 ... 4)."""
    table = tidy(raw).drop_duplicates()
    table = table.drop(columns=LEAKING).rename(columns={"AGE_GROUP": "AGE"})
    table["AGE"] = table["AGE"] - 1
    # rows that differed only in a dropped column are now copies too
    return table.drop_duplicates().reset_index(drop=True)


def formula_check(raw):
    """How exactly customer value follows a linear formula of usage, with and without the age group.

    Fitted inside each age group, the formula is exact (R^2 = 1); one formula for everybody is not, and
    rows equal in every column except customer value can belong to different age groups.
    """
    table = tidy(raw).drop_duplicates()
    usage = table.drop(columns=["AGE", "AGE_GROUP", "CUSTOMER_VALUE"])
    value = table["CUSTOMER_VALUE"]
    inside = {int(g): round(LinearRegression().fit(usage[m], value[m]).score(usage[m], value[m]), 6)
              for g, m in ((g, table["AGE_GROUP"] == g) for g in sorted(table["AGE_GROUP"].unique()))}
    overall = LinearRegression().fit(usage, value).score(usage, value)
    columns = [c for c in table.columns if c not in ("AGE", "AGE_GROUP", "CUSTOMER_VALUE")]
    twins = table[table.duplicated(subset=columns, keep=False)]
    mixed = int(twins.groupby(columns)["AGE_GROUP"].nunique().gt(1).sum())
    return {"linear_r2_inside_each_age_group": inside, "linear_r2_one_formula": round(float(overall), 4),
            "row_sets_equal_except_customer_value_with_different_age_group": mixed}


def near_copy_check(table, radius=0.3, seed=0):
    """Macro-F1 of gradient boosting with a plain stratified 5-fold split, and with rows that lie within
    `radius` of each other (log-scaled, standardised attributes) kept together in one fold."""
    X, y = table.drop(columns="AGE"), table["AGE"]
    logged = np.log1p(X.clip(lower=0))
    Z = (logged - logged.mean()) / logged.std()
    _, groups = connected_components(radius_neighbors_graph(Z, radius), directed=False)
    model = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    plain = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="f1_macro")
    grouped = cross_val_score(model, X, y, groups=groups, scoring="f1_macro",
                              cv=StratifiedGroupKFold(5, shuffle=True, random_state=seed))
    sizes = np.bincount(groups)
    return {"radius": radius, "groups": int(len(sizes)), "rows_with_a_near_copy": round(float((sizes[groups] > 1).mean()), 3),
            "macro_f1_plain_5fold": round(float(plain.mean()), 3), "macro_f1_grouped_5fold": round(float(grouped.mean()), 3)}


def describe(raw, table):
    return {"rows_raw": len(raw), "columns_raw": raw.shape[1], "exact_copies_removed": len(raw) - len(table),
            "dropped_columns": ["Age (one value per age group)", "Customer Value (a formula that depends on the age group)"],
            "rows": len(table), "attributes": [c for c in table.columns if c != "AGE"],
            "age_groups": {GROUP_NAMES[g]: int(n) for g, n in table["AGE"].value_counts().sort_index().items()},
            "age_values_per_group": raw.groupby("Age Group")["Age"].unique().apply(lambda v: sorted(map(int, v))).to_dict(),
            "missing_cells": int(table.isna().sum().sum())}
