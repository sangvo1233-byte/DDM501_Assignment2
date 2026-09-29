# DDM501 — Individual Assignment 2: ML Pipeline Design & MLOps Analysis

Võ Minh Sang · 25MS13286

- **Report:** [`DDM501_Assignment2_25MS13286_VoMinhSang.pdf`](DDM501_Assignment2_25MS13286_VoMinhSang.pdf)
  (source: `report.html`)
- **Experiment code:** [`code/run_experiments.py`](code/run_experiments.py), which runs 10 configurations, each one a
  tracked MLflow run
- **Results quoted in the report:** [`code/results/results.json`](code/results/results.json)

## Reproduce the results table

```bash
cd code
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_experiments.py          # writes mlruns/ and results/results.json
mlflow ui --backend-store-uri ./mlruns
```

`code/pipeline/` is a copy of the Lab 2 pipeline modules (ingestion, three-level validation, feature engineering,
evaluation). It is included so this repository runs on its own. `code/data/credit_default.csv` has the UCI
"Default of Credit Card Clients" schema: 30,000 rows and a 23.3% default rate. With `RANDOM_STATE=501` the metrics
reproduce exactly.
