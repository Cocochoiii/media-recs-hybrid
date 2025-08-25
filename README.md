# Hybrid Personalized Media Recommender

A production-style, open-source project implementing a **hybrid recommendation system** that combines:
- Collaborative filtering (PyTorch Matrix Factorization + sequence/LSTM)
- Content-based semantic retrieval (BERT embeddings with a TF-IDF fallback)
- A simple **contrastive learning** module to align user and item spaces

**Built for:** Northeastern University project (Oct 2024 – Apr 2025).

---

## Highlights

- **Scale**: Designed to handle 50k+ user interactions and 5k+ items (synthetic generators included).
- **Hybrid**: Combines collaborative filtering scores with BERT/TFiDF content similarities.
- **Models**: Matrix Factorization (implicit feedback), LSTM session model, and a lightweight contrastive setup (InfoNCE).
- **API**: FastAPI microservice with `/recommendations` and `/health`.
- **Monitoring**: Prometheus metrics + Grafana dashboard scaffolding.
- **Tracing**: OpenTelemetry (OTLP) -> Collector -> (export to your APM of choice).
- **Logging**: Structured JSON logs to Logstash -> Elasticsearch -> Kibana.
- **Deploy**: `docker-compose up -d` brings up API + monitoring + logging stack (see notes).
- **Distributed Training**: Example SageMaker entrypoint with Estimator scaffolding.

> Note: This repository favors clarity and completeness. It’s designed to run locally on a laptop (CPU) or on a small VM, while providing pathways for production scaling.

---

## Quickstart

```bash
# 1) Clone & enter
git clone https://github.com/you/media-recs-hybrid.git
cd media-recs-hybrid

# 2) (Optional) create venv
python3 -m venv .venv && source .venv/bin/activate

# 3) Install (API + training)
pip install -r requirements.txt

# 4) Generate small synthetic data
python data/generate_synthetic_data.py --users 500 --items 800 --interactions 8000

# 5) Train a quick MF model (CPU)
python recs/training/train_mf.py --epochs 3 --out_dir artifacts

# 6) Precompute content embeddings (TF-IDF fallback if transformers not available)
python recs/training/train_content.py --out_dir artifacts

# 7) Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
# try: http://localhost:8000/recommendations?user_id=u_001
```

### Docker Compose (API + Monitoring + ELK + OTel)
```bash
docker compose up -d
# API on :8000, Prometheus on :9090, Grafana on :3000, Kibana on :5601
```
> First run of Elasticsearch can take ~1–2 minutes. Kibana will follow.

---

## Project Layout

```
media-recs-hybrid/
├── api/                    # FastAPI service + telemetry + logging
├── artifacts/              # Saved models/encoders (created after training)
├── configs/                # Prometheus, Grafana, Logstash, OTel collector
├── data/                   # CSVs and synthetic data generator
├── recs/                   # Core library: models, training, metrics
├── scripts/                # Helper scripts (profiling, load test, bootstraps)
├── tests/                  # Unit tests (pytest)
├── .github/workflows/      # CI: lint + tests
├── docker-compose.yml
├── Dockerfile.api
├── LICENSE (MIT)
└── README.md
```

---

## API

- `GET /health` – liveness probe
- `GET /recommendations?user_id=<id>&k=10` – hybrid ranked list
- `GET /metrics` – Prometheus metrics (via instrumentator)

**Hybrid Score**: `alpha * CF + (1 - alpha) * ContentSim`, with per-request or config override (default `alpha=0.6`).

---

## Models

**Matrix Factorization (MF)** – implicit feedback; learns user and item factor matrices with simple negative sampling.

**Content Encoder** – tries to use `sentence-transformers` to embed titles/desc. If unavailable, falls back to TF-IDF + cosine.

**Sequence LSTM** – optional session model; predicts next-item based on past sequence.

**Contrastive (InfoNCE)** – optional alignment of user and item embeddings for better hybrid mixing.

---

## Monitoring / Logging / Tracing

- **Prometheus**: scrapes API `/metrics`.
- **Grafana**: pre-provisioned Prometheus datasource (basic dashboard stub in `configs/grafana`).
- **ELK**: Python logs -> Logstash (TCP) -> Elasticsearch -> Kibana.
- **OpenTelemetry**: FastAPI instrumentation -> OTel Collector (OTLP gRPC 4317).

> Credentials for Elasticsearch are set via environment vars in `docker-compose.yml`. For PoC the defaults are used; tweak for production.

---

## Profiling

While the original note mentions `pprof` (common in Go), this Python repo includes:

- **cProfile** helper (`scripts/profile_api.sh`) to profile endpoints
- **py-spy** instructions to capture a live flamegraph:
  ```bash
  pip install py-spy
  py-spy record -o flame.svg --pid $(pgrep -f "uvicorn api.main") --duration 20
  ```

---

## AWS SageMaker (optional)

An example entrypoint is provided at `recs/training/sagemaker_entrypoint.py`. You can package this repo as a training image and launch a distributed training job with a PyTorch Estimator. See the inline comments for guidance.

---

## Tests & CI

```bash
pytest -q
```
GitHub Actions workflow runs lint + tests on pushes and PRs.

---

## License

[MIT](LICENSE)

---

## Acknowledgements

This codebase is provided as an educational, end-to-end template for hybrid recommenders, observability, and deployment. Adjust hyperparameters, data schema, and infra for your needs.
