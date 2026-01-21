# Customer Lifetime Value (LTV) Prediction Case Study

**Repository layout**
```
.
├── notebook/
│   └── LTV_Case_Study_Notebook.ipynb
├── data/
│   ├── case_study_base.csv
│   └── case_study_xs.csv
├── README.md          
└── LICENSE            # MIT License
```

## Project overview

This repository contains an end to end case study that predicts **12 month Customer Lifetime Value (LTV)** at the time of customer signup, using only features available at purchase. The LTV definition used in the project is:

```
LTV_12m = Base subscription revenue (sum of monthly commissions while active, up to 12 months)
        + One-time cross-sell revenue (purchased within first 12 months)
```

The notebook implements two modeling strategies:
- **Approach A — Direct Baseline Regression:** predict total `LTV_12m` directly.
- **Approach B — Decomposed LTV Modeling:** model base subscription value via discrete time survival (person month hazard) and cross sell using a hurdle model (probability of purchase + conditional amount). Then combine components to obtain expected LTV.

## Files of interest

- `notebook/LTV_Case_Study_Notebook.ipynb` — main notebook implementing data processing, EDA, modeling, and evaluation.
- `data/case_study_base.csv` — customer level base table (signup date, churn date, product, monthly commission, etc.).
- `data/case_study_xs.csv` — cross sell transactions (user_id, date, one-time commission).
- `LICENSE` — MIT License.

## Quick start

1. Clone the repository:
```bash
git clone https://github.com/Moiz101-ch/ltv-prediction-case-study.git
```

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate         # Windows
```

1. Install dependencies (the notebook also installs core packages on demand):
```bash
pip install numpy pandas matplotlib scikit-learn jupyterlab
```

1. Launch the notebook:
```bash
jupyter lab
# or
jupyter notebook
```
Open `notebook/LTV_Case_Study_Notebook.ipynb`.

## Reproducibility notes

- The notebook includes a deterministic random seed (`SEED = 42`) to make results reproducible.
- The notebook uses a time based train/test split (by `customer_started_at`) to avoid leakage.
- The pipeline uses scikit-learn `Pipeline` and `ColumnTransformer` for preprocessing and modeling to make it easier to serialize and reproduce training steps.

## Reported metrics

**Approach A — Direct Regression**
- RMSE ≈ 112
- MAE ≈ 51.8
- R² ≈ 0.10

**Approach B — Decomposed LTV**
- Hazard (monthly churn) AUC ≈ 0.786
- Base LTV (from hazard) — MAE ≈ 51.3, RMSE ≈ 78.1, R² ≈ -0.14
- Cross-sell classifier AUC ≈ 0.605
- Cross-sell conditional regression — MAE ≈ 13.0, RMSE ≈ 91.1, R² ≈ ~0
- Final combined LTV values depend on correct recombination of components (see notebook for diagnostic checks).

> Note: These metrics were computed in the notebook as a first iteration baseline. The notebook also contains diagnostics, plots, and a suggested next step checklist.

## Suggested improvements (high level)

- Add behavioral and engagement features (session counts, recent activity).
- Use tree based models (LightGBM / XGBoost / CatBoost) or survival specific models for better non-linear fit.
- Calibrate predicted probabilities (important when multiplying probability × conditional amount).
- Use decile lift / top-k capture as business oriented evaluation metrics (useful for marketing targeting).
- Add monitoring and reproducible model serialization for production use.

## License

This repository is licensed under the **MIT License** — see `LICENSE` for details.

## Author

Abdul Moiz Nazir
