# Causal Portfolio Recommendation Engine

A recommendation system for asset management that reasons **causally**
about recommendations (uplift / double machine learning) rather than
optimizing for engagement or similarity like standard collaborative
filtering — modeling the effect of recommending one action vs. another,
personalized to a user's risk profile and current market conditions.

Full proposal, architecture, and phased plan: `docs/AWS_DEPLOYMENT.md`
(deployment) and `docs/Causal_Recsys_Project_Proposal.docx` (original spec).

## Status: MVP complete (7-day build)

End-to-end: synthetic data → cleaned/joined dataset → CF baseline +
causal uplift model, both trained → MLflow-tracked → offline A/B (IPS)
evaluated → served via FastAPI → displayed in a React dashboard with
causal rationale.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline (data gen -> ETL -> both models -> tracking -> eval)
./run_pipeline.sh

# Terminal 1: API
uvicorn src.serving.app:app --reload --port 8000
# -> http://localhost:8000/docs

# Terminal 2: dashboard
cd client && npm install && npm run dev
# -> open the printed Vite URL

# Tests
pytest tests/ -v

# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

## Repo layout
```
data/
  raw/                  synthetic source datasets + DATA_DICTIONARY.md
  processed/            ETL output, model predictions, A/B report
src/
  ingestion/            synthetic data generator, ETL, multimodal embeddings
  models/               CF baseline (S-learner), causal uplift model (DML)
  evaluation/           MLflow tracking, offline A/B (IPS) framework
  serving/              FastAPI recommendation endpoint
client/                 React dashboard (Vite)
tests/                  pytest suite (14 tests: data integrity, ETL/leakage, API)
docs/                   original proposal + AWS deployment guide
.github/workflows/      CI: tests, retrain check, frontend build
models/                 trained model artifacts (.joblib, gitignored)
run_pipeline.sh         end-to-end pipeline runner
```

## Key results

| Metric | CF baseline (non-causal) | Causal uplift model (DML) |
|---|---|---|
| MAE vs. true ITE | 0.604 | **0.596** |
| ATE recovery gap | 0.208 | **0.181** |
| Offline IPS policy value | **1.6207** | 1.5739 |

The uplift model recovers the true causal effect more accurately (lower
MAE, smaller ATE gap). The CF baseline scores marginally higher on the
IPS offline metric — bootstrapped and statistically significant (90% CI
excludes zero) — because IPS measures agreement with the logged
(confounded) policy, a different question from causal accuracy. Both
results are reported rather than picking whichever looks better; see
`data/processed/offline_ab_report.csv` and the Day 3/4 notes below.

## Known substitutions (things that need your credentials, not mine)

- **Databricks ETL** → implemented in plain pandas (`src/ingestion/etl.py`),
  structured to port 1:1 to PySpark if you have a real Databricks workspace.
- **AWS deployment** → not deployed (needs your AWS account). Full command
  sequence in `docs/AWS_DEPLOYMENT.md`.
- **GPT-4o/Gemini multimodal embeddings** → falls back to a local
  deterministic pseudo-embedding. Set `OPENAI_API_KEY` and rerun
  `src/ingestion/multimodal_embeddings.py` for real embeddings — no other
  code changes needed.

## Open items carried over from the original proposal

- Time-series forecaster (Prophet/statsmodels) is in the tech stack and
  architecture diagram but was never placed in a specific day — still
  unscheduled. Natural next step: feed `market_timeseries.csv` regime/
  volatility forecasts in as an additional feature to both models.
- AWS billing ownership — still open, flagged in `docs/AWS_DEPLOYMENT.md`.
