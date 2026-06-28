# PageRank Feasibility Audit of Wikipedia RfA Support Votes

This project analyzes the SNAP Wikipedia Requests for Adminship network as a directed support-vote graph. It computes PageRank, labels support edges as PageRank-uphill or non-uphill, engineers graph features, trains machine-learning models to predict uphill edges, and interprets the findings as an empirical PageRank domination-feasibility audit.

## Dataset

Place the raw SNAP file here:

```text
data/raw/wiki-RfA.txt
```

## Local Reproducibility

Install requirements:

```powershell
pip install -r requirements.txt
```

Run full pipeline:

```powershell
python src\run_pipeline.py
```

## Docker Reproducibility

Build image:

```powershell
docker build -t pagerank-rfa-audit .
```

Run image:

```powershell
docker run --rm -it -v "${PWD}:/app" -w /app pagerank-rfa-audit
```
