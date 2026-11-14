# Step 2 - oversampling the rare age groups

**Notebook:** `notebooks/step2_oversampling.ipynb` (CPU, internet on for `ctgan`, inputs `agender-mtl-code`
and the Step 0 output, about 2-3 hours). Code: `agender_mtl/oversample.py`, `step2.py`.

## Question

The two smallest age groups are rare in LAT. In the 6-class scheme these are <18 and 60+, which together hold
about 8 % of the subscribers. The network already gives rare groups more weight in its loss. Step 2 asks a
second question: does it help to add synthetic training rows for the rare groups, and which way of making
them helps most?

**Age schemes:** 6 and 10 classes, all methods. 48 classes (one per year of age): only "none" and
"SMOTE, earlier setting", to re-produce the earlier 48-class row; joint oversampling would split each year
of age further by gender, which leaves too few subscribers per class.

## What stays fixed

For each scheme and seed, the split, the encoding and the QUBO choice of attributes are made once, before
any oversampling. They are exactly those of Step 1. Every method therefore starts from the same training
rows and the same attributes. Only the training rows are oversampled. The validation rows (used to stop
training) and the test rows (used to score) are always real subscribers.

## Methods

| Method | How a new row is made | Label balanced | Rows added |
|---|---|---|---|
| none | no new rows; class weights only, as in Step 1 | - | 0 |
| SMOTE, earlier setting | a new row on the straight line between a real row and one of its 5 nearest neighbours of the same class | age group | the two smallest age groups, raised to the median group size |
| random oversampling | a copy of a randomly chosen real row of the same class | gender x age group | every class below the median size, raised to the median size |
| SMOTE-NC | numbers as in SMOTE; each categorical column takes the most common value among the neighbours | gender x age group | to the median |
| SMOTE-NC, age label only | as SMOTE-NC; gender is treated as one more categorical column | age group | to the median |
| SMOTE-NC, full balance | as SMOTE-NC | gender x age group | every class raised to the largest class |
| Borderline-SMOTE | SMOTE, but only from rare rows whose neighbours are mostly of other classes (the rows near a class border) | gender x age group | to the median (at most) |
| ADASYN | SMOTE, with more new rows near rare rows that are hard to classify | gender x age group | about the median |
| CTGAN | a generative adversarial network trained on all training rows, then asked for rows of a given class | gender x age group | to the median |

**Why the joint label.** The network predicts gender and age together, so every new training row needs both
labels. If only the age label is balanced, the new rows can also change the share of men and women.
Balancing the joint label (gender x age group) makes each new row from rows of the same gender and the same
age group, so it has both labels and the gender balance barely moves (the raised classes of men and women grow
to the same size, so the share of men moves a little toward one half). "SMOTE-NC, age label only" measures
what happens without this choice. "SMOTE, earlier setting" repeats the earlier SMOTE study of this work.

**Why these methods.** SMOTE (Chawla et al., 2002) is where most studies start. SMOTE-NC is the version for
mixed number and category columns, which is what LAT has. Borderline-SMOTE (Han et al., 2005) and ADASYN
(He et al., 2008) put the new rows where the classes overlap. CTGAN (Xu et al., 2019) learns the whole
distribution of the table and samples from it. Recent comparisons on tabular data do not agree on a winner:
SMOTE-type methods often match or beat deep generators on small data, and CTGAN is reported to suffer on
small classes. That is why both families are tested here, on the same rows.

**One-hot attributes.** SMOTE-NC and CTGAN treat each original categorical column as one value. SMOTE,
Borderline-SMOTE and ADASYN interpolate the 0/1 columns as numbers; afterwards each new row gets back exactly
one category per original column (the largest value), or none when no value reaches 0.5. "None" means a
category the QUBO step did not keep.

**CTGAN settings.** 100 epochs, batch 500, CPU. Rows are asked for with the class as the condition; only rows
whose generated class equals the asked class are kept (the share kept is recorded as `ctgan_label_match`).

## Models

The multi-task network exactly as in Step 1, and Gradient Boosting, the strongest baseline in Step 1 (one
model for gender and one for age). Both are trained on the oversampled rows. The network keeps its class
weights; after oversampling they are computed on the new class sizes.

## Measures (test rows, five seeds)

- **Age:** macro-F1, balanced accuracy, recall in each age group, the mean recall of the two smallest groups
  in the training rows (`rare_recall`), the share of test rows predicted as one of those two groups, +-1
  accuracy, one-vs-rest AUC, ECE.
- **Gender:** AUC, accuracy, ECE (to see whether oversampling disturbs the gender task).
- **Paired comparison:** each method minus "none" on the same five seeds: mean difference, 95 % t interval,
  paired t-test, and the number of seeds where the method is higher or lower.

## Checks of the synthetic rows

- `distance_ratio`: median distance from a synthetic row to the closest real row of its class, divided by the
  median distance between a real row and the closest other real row of its class. Near 0 means the new rows
  are copies; near 1 means they lie about as close to the real rows as real rows lie to each other.
- `discriminator_auc`: how well gradient boosting tells synthetic rows from real rows of the same classes
  (3-fold cross-validated ROC-AUC). 0.5 means they cannot be told apart.

## Output (`/kaggle/working/agender_mtl_outputs/step2/`)

- `parts/scores_*.csv` and `parts/records_*.json`: one file per (scheme, method), written as soon as it is
  done.
- `scores.csv`: every score of every seed. `means.csv`: mean and standard deviation per method.
- `paired.csv`: every method against "none". `records.csv`: rows added, time, gender share before and after,
  the synthetic-row checks.
- `qubo_chosen.json`: the attributes chosen per (scheme, seed).
- `step2_summary.json`, `steps.json`.

Check: the "none" rows of the network must equal the Step 1 test scores for the same scheme and seed,
because the split, the attributes and the training are the same.
