"""
Download, read and describe the LAT dataset (Kuang et al. 2022, PLOS ONE, supporting file S1).

Nothing is changed here; the description only counts, so any decision about odd values is taken
after reading it.
"""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

LAT_URL = ("https://journals.plos.org/plosone/article/file?type=supplementary"
           "&id=10.1371/journal.pone.0271714.s001")
LAT_MEMBER = "minimal-dataset/age-gender-pred-dataset.csv"


def file_hash(path, kind="sha256"):
    digest = hashlib.new(kind)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target, timeout=120):
    """Download to a temporary name first, so a broken transfer never looks like a finished file."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(partial, "wb") as f:
            for block in response.iter_content(1 << 20):
                f.write(block)
    partial.replace(target)
    return {"file": target.name, "url": url, "bytes": target.stat().st_size, "sha256": file_hash(target)}


def fetch_lat(folder):
    return download(LAT_URL, Path(folder) / "lat_s1.zip")


def read_lat(path):
    """Read the LAT table from the PLOS zip, or from the csv inside it. Kaggle unpacks zip files when
    a notebook output is saved as a dataset, so later steps usually see the csv."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            raw = z.read(LAT_MEMBER)
    else:
        raw = path.read_bytes()
    # keep_default_na=False so that empty cells stay empty and we decide later what they mean
    return pd.read_csv(io.BytesIO(raw), keep_default_na=False, na_values=[""])


def describe_lat(table):
    age = table["AGE"]
    groups = pd.cut(age, [-float("inf"), 17, 25, 35, 45, 60, float("inf")],
                    labels=["<18", "18-25", "26-35", "36-45", "46-60", "60+"])
    missing = table.isna().sum()
    return {"rows": len(table), "columns": table.shape[1],
            "gender": {str(k): int(v) for k, v in table["GENDER"].value_counts().items()},
            "age_min": int(age.min()), "age_max": int(age.max()), "distinct_ages": int(age.nunique()),
            "six_groups": {str(k): int(v) for k, v in groups.value_counts(sort=False).items()},
            "missing_cells": {c: int(n) for c, n in missing[missing > 0].items()}}


def save_json(record, path):
    with open(path, "w") as f:
        json.dump(record, f, indent=1, default=str)
