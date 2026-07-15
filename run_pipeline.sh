#!/usr/bin/env bash
# End-to-end pipeline run (Day 7 integration).
# Runs every phase in order: data generation -> ETL -> both models ->
# tracking -> offline eval -> (serving/client are run separately, see below).
#
# Usage: ./run_pipeline.sh   (from repo root)
set -euo pipefail

echo "=== 1/7: Generating synthetic data ==="
python3 src/ingestion/generate_synthetic_data.py

echo ""
echo "=== 2/7: Running ETL (clean + join) ==="
python3 src/ingestion/etl.py

echo ""
echo "=== 3/7: Training CF baseline (non-causal) ==="
python3 src/models/cf_baseline.py

echo ""
echo "=== 4/7: Training causal uplift model (DML) ==="
python3 src/models/uplift_model.py

echo ""
echo "=== 5/7: Logging MLflow runs ==="
python3 src/evaluation/mlflow_tracking.py

echo ""
echo "=== 6/7: Offline A/B evaluation (IPS) ==="
python3 src/evaluation/offline_ab.py

echo ""
echo "=== 7/7: Multimodal report embeddings ==="
python3 src/ingestion/multimodal_embeddings.py

echo ""
echo "=== Pipeline complete ==="
echo "Next steps to see it end-to-end:"
echo "  Terminal 1: uvicorn src.serving.app:app --reload --port 8000"
echo "  Terminal 2: cd client && npm install && npm run dev"
echo "  Then open the printed Vite dev URL, and http://localhost:8000/docs for the API."
