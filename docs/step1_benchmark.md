# Step 1 - the benchmark

**Notebook:** `notebooks/step1_benchmark.ipynb` (CPU, inputs `agender-mtl-code` and `agender-mtl-step0`,
about 30-45 min). Code: `agender_mtl/benchmark.py`, `step1.py`, `mtl.py`, `qubo.py`, `metrics.py`.

| Part | Setting |
|---|---|
| Attributes | the 29 raw LAT attributes |
| Preprocessing | log(1+x) for skewed non-negative numbers (skew > 1 on the training rows), then standardisation; one-hot for categorical columns with at most 10 values; frequency encoding for the others; missing values as the category `missing`; everything fitted on training rows only |
| Split | 70 / 15 / 15 stratified by the six age groups; seeds 42, 7, 123, 2024, 99 |
| QUBO | R = NMI with the joint label (gender x number of age groups + age group); Q = NMI between attributes (10 quantile bins); alpha 1, beta 0.05, lambda 1.5, k 25; simulated annealing, 6 restarts x 20,000 single-bit flips, T 5 -> 0.01 geometric |
| Network | NumPy, hand-written backpropagation (checked against finite differences); trunk d-64-32 ReLU, dropout 0.3; gender head sigmoid; age head 33-16-C fed the detached gender probability; class weights (1/count)^0.5; loss 2 L_gender + L_age; AdamW 1e-3 -> 1e-5 cosine, weight decay 5e-4, batch 256, 150 epochs, lowest validation loss kept |
| Baselines | Logistic Regression (balanced class weights), Random Forest (300 trees), Gradient Boosting (histogram version for 48 classes); one model per task, on the QUBO-selected columns |
| Age schemes | 3 (<28, 28-39, 40+), 6 (<18, 18-25, 26-35, 36-45, 46-60, 60+), 10 (<15, 15-19, ..., 50-54, 55+), 48 (one per year, 13 ... 60+, also scored within +-5 years) |
| Sweep | one QUBO weight at a time around the defaults; validation macro-F1 of the 6-class age head; five seeds |

## Output (`/kaggle/working/agender_mtl_outputs/step1/`)

- `scores_{3,6,10,48}class.csv`: every score, every model, every seed (validation and test for the network).
- `qubo_*class.json`: chosen attributes and relevance values per seed. `mtl_loss_*class.csv`: loss curves.
- `qubo_sweep.csv`, `qubo_sweep_means.csv`: the sensitivity sweep.
- `step1_summary.json`, `steps.json`.
