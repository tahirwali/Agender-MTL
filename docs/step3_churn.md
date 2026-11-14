# Step 3 - external check on Iranian Churn

**Notebook:** `notebooks/step3_churn.ipynb` (CPU, internet on, input `agender-mtl-code`, about 15-25 min).
Code: `agender_mtl/churn.py`, `step3.py`.

## Data

Iranian Churn (UCI 563, doi 10.24432/C5JW3Z, CC BY 4.0; Jafari-Marandi et al., Neural Computing and
Applications 2020): 3,150 customers of an Iranian operator, aggregated over nine months, five age groups,
no gender. Downloaded from UCI by the notebook.

Three properties are handled before any model is trained, and each is measured by the code:

| Property | What we found | What we do |
|---|---|---|
| `Age` | one value per age group (15, 25, 30, 45, 55): the label written as a number | dropped |
| `Customer Value` | a linear formula of the usage columns whose coefficients differ between age groups: inside each group the formula fits exactly (R^2 = 1.000); 27 sets of rows agree on every other column and still belong to different age groups (`formula_check`) | dropped |
| exact copies | 300 rows repeat other rows | removed before splitting (2,850 customers remain) |

`near_copy_check` asks whether very similar customers split between training and test could explain the
scores: gradient boosting, 5-fold, plain against folds that keep rows within distance 0.3 (log-scaled,
standardised attributes) together.

## Pipeline

70 / 15 / 15 split stratified by age group, seeds 42, 7, 123, 2024, 99; encoding as for LAT. QUBO relevance to
the age group; the number of attributes kept (4, 6, 8, 10 or all 11) is chosen on the validation rows of each
seed. Models: the network's age head (gender weight 0), Logistic Regression, Random Forest, Gradient
Boosting; then every Step 2 oversampling method for the network and Gradient Boosting (without gender the
joint label is the age group, so "SMOTE-NC, age label only" is left out).

## Output (`/kaggle/working/agender_mtl_outputs/step3/`)

`scores.csv`, `means.csv`, `paired.csv` (each method against no oversampling), `records.csv` (rows added,
checks of the synthetic rows), `qubo_choice.json` (k and attributes per seed), `step3_summary.json` (data
description, the customer-value check and the near-copy check).
