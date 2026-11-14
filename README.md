# AGender-MTL

Joint age and gender prediction from aggregated mobile-usage data, with one multi-task network.

Telecom operators often share only summarised, anonymised usage records. This repository measures how well
age and gender can still be predicted from such records. A single network predicts both: its age head also
receives the predicted gender. Attributes are chosen by a QUBO (minimum redundancy, maximum relevance)
solved by simulated annealing on a CPU. The network is compared with single-task Logistic Regression, Random
Forest and Gradient Boosting under four age schemes (3, 6, 10 and 48 classes), over five seeds.

Everything runs on Kaggle's free tier, on the CPU.

## Pipeline

| Step | Notebook | What it does | Run time |
|---|---|---|---|
| 0 | `step0_data.ipynb` | Downloads LAT from PLOS and describes it (rows, labels, age groups, missing cells) | < 1 min |
| 1 | `step1_benchmark.ipynb` | Preprocessing, QUBO selection, multi-task network and baselines for the 3-, 6-, 10- and 48-class age schemes; QUBO sensitivity sweep | 30-45 min |

## Data

LAT (Kuang et al., PLOS ONE 2022, doi 10.1371/journal.pone.0271714, supporting file S1, CC BY 4.0) is
downloaded by Step 0 and is not stored here. See `docs/datasets.md`.

## Running it

1. Zip this folder and upload it to Kaggle as a dataset named `agender-mtl-code`.
2. Import a notebook from `notebooks/` and add `agender-mtl-code` as an input. Step 1 also needs the saved
   output of Step 0 (`agender-mtl-step0`).
3. Use **Save Version -> Save & Run All**.

Every number and figure is written by the notebooks; no result is typed by hand. Settings are chosen on the
validation split, and the test split is scored once per seed.

Tests run locally without internet:

```
pip install -r requirements.txt
python -m pytest tests
```

## Layout

```
agender_mtl/   the package: data, benchmark (preprocessing, models, age schemes, sweep), multi-task network,
               QUBO selection, metrics, step runner
notebooks/     one Kaggle notebook per step
tests/         offline tests
docs/          one note per step and the data note
```

## Licence

MIT (see `LICENSE`). The dataset keeps its own licence.
