# Crypto Content Recommender

A reproducible recommendation system for crypto education, news, and analysis. The project traces nine model variants from simple retrieval baselines to a PyTorch two-tower model, a neural ranker, and a Transformer reranker. It includes a Streamlit demo, a CPU inference API, and the artifacts needed to reproduce the offline evaluation.

[Live Streamlit demo](https://rkatmcugsybuh4d5cejg6s.streamlit.app/) · [End-to-end notebook](notebooks/01_end_to_end_walkthrough.ipynb) · [Project overview in Traditional Chinese](README_REPOSITORY_OVERVIEW.md)

## Evaluation

The synthetic dataset contains **240 content items, 80 users, and 8,564 interactions**, including impressions, clicks, likes, saves, and hides. At the shared temporal cutoff, 199 content items and 6,852 interactions are available for training. Each of the 30 eligible test users contributes one later positive target that was unseen during training and has no hide feedback. Hit Rate@10 and Recall@10 are therefore equal in this evaluation.

| Version | Recommendation method | Hit Rate@10 | NDCG@10 |
| --- | --- | ---: | ---: |
| V1 | Time-decayed popularity | 0.1333 | 0.0525 |
| V2 | TF-IDF content retrieval | 0.1333 | 0.0711 |
| V3 | Item-item collaborative filtering | 0.0333 | 0.0105 |
| V4 | Hybrid candidate retrieval | 0.1333 | 0.0617 |
| V5 | Logistic regression ranker | 0.6667 | 0.3981 |
| V6 | Reranking with topic and source diversity | 0.6667 | **0.3991** |
| V7 | PyTorch two-tower retrieval | 0.1333 | 0.0452 |
| V8 | Neural ranking over frozen two-tower candidates | 0.2333 | 0.1013 |
| V9 | Transformer reranking over two-tower candidates | 0.2000 | 0.1225 |

V6 achieved the highest NDCG@10 on this synthetic holdout. V1–V6 follow their respective candidate rules, while V7–V9 share a pool capped at 50 two-tower candidates. Within that shared pool, V9 has higher NDCG@10 than V8, with 6 of 30 targets in the top ten compared with 7 for V8. The evaluation exports include per-user results, candidate counts, and paired bootstrap intervals relative to V6. [Full evaluation artifacts](reports/repository_evaluation/)

## Run the demo locally

From the repository root, use Python 3.11, the version used by GitHub Actions.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**User Feed** runs the six baseline methods and supports session feedback. **Saved Model Results** shows the nine-model evaluation and lets you browse saved V7–V9 recommendations for the 80 synthetic users and a new-user case. The demo reads the saved neural outputs without loading PyTorch weights.

## Serve and reproduce the models

Install the service dependencies in the same environment and start the local API.

```bash
python -m pip install -r requirements-service.txt
python -m src.serve_neural
```

In another terminal, send a recommendation request.

```bash
curl http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"U0001","model":"v9_transformer_content_reranker","k":10}'
```

The FastAPI service serves V7–V9 on CPU. It checks the checkpoint manifests at startup, shares one loaded two-tower model across the three variants, and includes checkpoint hashes in the response. V9 also returns its reranker checkpoint hash. Use `/health` to check the process and `/ready` to inspect the loaded models. [API and integration details](README_SYSTEM_INTEGRATION.md)

The following commands rebuild the nine-model evaluation in a new output directory and verify the artifacts checked into the repository.

```bash
python -m src.evaluate_repository --output reports/repository_evaluation_new
python -m src.evaluate_repository --verify-only
python -m src.run_transformer --verify-only
python -m pytest -q
```

The rebuild trains the six baselines from the fixed synthetic dataset and loads the saved V7–V9 checkpoints. It exports model metrics, per-user results, recommendations, candidate counts, and paired quality intervals. [GitHub Actions workflow](.github/workflows/verify.yml)

## Performance and training evidence

| Experiment | What was measured | Evidence |
| --- | --- | --- |
| CPU HTTP inference | 1,200 local requests across V7–V9 at concurrency 1 and 4, with zero recorded errors. V9 p95 latency was 6.05 ms at concurrency 1 and 17.74 ms at concurrency 4. | [HTTP benchmark results](reports/service_v12_benchmark/summary.csv) |
| Transformer inference | 800 in-process CPU requests comparing V8 and V9 across four rounds, with latency, throughput, CPU time, and sampled memory recorded. | [Transformer benchmark](README_TRANSFORMER_RERANKING.md) |
| Apple GPU profiling | Six rounds of V7 and V8 batch inference measurements on Apple M5 Pro using PyTorch MPS. | [MPS profiling and audit](README_GPU_MEASUREMENT_AUDIT.md) |
| Distributed training | One-process and two-process CPU/Gloo runs cover global batching, gradient weighting, checkpoint reloads, and communication sampling. The implementation also includes a CUDA/NCCL path. | [DDP implementation and CPU results](README_DDP_TRAINING.md) |

The HTTP and in-process benchmarks use different timing boundaries. HTTP measurements include client and server work such as transport, queuing, and response validation. Both benchmarks describe the recorded local workloads.

## Repository map

| Path | Contents |
| --- | --- |
| [`app.py`](app.py) | Streamlit feed, evaluation dashboard, and saved recommendations |
| [`src/`](src/) | Data generation, retrieval and ranking, evaluation, API serving, profiling, and distributed training |
| [`tests/`](tests/) | Model, API, dashboard, and artifact regression checks |
| [`data/processed/`](data/processed/) | Synthetic content, user profiles, and interactions |
| [`artifacts/neural/`](artifacts/neural/) | Shared neural checkpoint and training artifacts |
| [`reports/`](reports/) | Saved evaluations, benchmark logs, checkpoints, and verification records |

For model training and design details, see the [PyTorch two-tower and neural ranker notes](README_PYTORCH.md), the [Transformer reranker notes](README_TRANSFORMER_RERANKING.md), and the [data schema](reports/DATA_SCHEMA.md).
