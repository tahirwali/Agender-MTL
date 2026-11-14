# Step F - figures and tables

**Notebook:** `notebooks/stepF_figures.ipynb` (CPU, about 5 min). Inputs: `agender-mtl-code` and the outputs
of Steps 0-3 (as datasets, or attached notebooks). Code: `agender_mtl/figures.py`; test `tests/test_figures.py`.

Nothing is typed in. Everything is drawn from the files Steps 1-3 saved. Two small computations happen here:
one repeat of the network (seed 42) for the 6- and 10-class schemes, because confusion matrices, ROC curves
and permutation importance need test predictions (same split, attributes and training as Step 1, so the same
model); and a regression check (gradient boosting regression of age on the QUBO attributes, five seeds:
share within +-5 years and mean absolute error).

## Figures (`figures/`, PDF + PNG, with the plotted values as CSV where useful)

| File | Content |
|---|---|
| `fig_data_distribution` | gender and six age groups of LAT |
| `fig_correlation` | Pearson correlation of the numeric LAT attributes |
| `fig_qubo_sweep` | validation macro-F1 when one QUBO weight changes |
| `fig_loss_6class`, `fig_loss_10class` | training and validation loss, mean and range over seeds |
| `fig_confusion_age_6class`, `fig_confusion_gender_6class`, `fig_confusion_age_10class` | confusion matrices, counts and share of each true class (seed 42) |
| `fig_roc_age_6class`, `fig_roc_gender_6class`, `fig_roc_age_10class` | ROC curves (seed 42) |
| `fig_qubo_relevance` | relevance of every attribute, chosen ones marked |
| `fig_importance` | permutation importance for age (drop in macro-F1) and gender (drop in AUC) |
| `fig_oversampling_recall_6class`, `_10class` | recall in each age group, the network, per oversampling method |
| `fig_oversampling_paired` | each method minus no oversampling, rare-group recall and macro-F1, 95 % intervals |

## Tables (`tables/`, LaTeX + CSV)

`table_lat_6class` (mean +- sd), `table_ece` (LAT gender and age, IC age), `table_schemes` (3, 6, 10, 48
classes), `table_comparison` (our best against published work), `table_10class`, `table_48class` (with the
network after the earlier SMOTE setting), `table_sweep`, `table_oversampling`,
`table_oversampling_paired_6class`, `_10class`, `table_synthetic_rows`, `table_iranian`,
`table_iranian_oversampling`. `numbers.json` holds the values quoted in the text.
