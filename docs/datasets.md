# Datasets

| Dataset | What it contains | Labels | Size | Source and licence |
|---|---|---|---|---|
| **LAT** (Kuang et al., PLOS ONE 2022, doi 10.1371/journal.pone.0271714, file S1) | Pre-paid subscribers of a Latin-American operator: payment and recharge, devices, data use (MB and sessions) of social network, Instagram, WhatsApp, bank, Uber and internet; made and received calls | age in years, gender | 11,034 subscribers, 29 attributes | Direct download from PLOS, CC BY 4.0 |

The data is not stored in this repository. Step 0 downloads it and records its SHA-256.

## Iranian Churn (external check, Step 3)

- Source: UCI Machine Learning Repository, dataset 563, doi 10.24432/C5JW3Z; licence CC BY 4.0.
  Described in Jafari-Marandi et al., Neural Computing and Applications 32 (2020) 14929-14962.
- 3,150 customers of an Iranian operator, 13 usage and account attributes aggregated over nine months, five
  age groups, no gender, no missing values.
- `Age` (one value per group) and `Customer Value` (a formula that depends on the age group) reveal the label
  and are dropped; 300 exact copies are removed; 2,850 customers and 11 attributes remain. See
  `docs/step3_churn.md`.
