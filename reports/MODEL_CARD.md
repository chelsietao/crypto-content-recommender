# Model Card

## System name

Personalized & Diverse Crypto Content Recommender

## Intended use

Educational portfolio prototype for studying content recommendation, cold start, ranking,
diversity, product metrics, and offline experimentation. It is intended for local demonstration
and technical discussion.

## Out-of-scope use

- Cryptocurrency price or return prediction
- Trading, portfolio, or asset-allocation advice
- Financial suitability decisions
- Production content moderation or misinformation detection
- Claims about Binance users, creators, products, or private systems

## Data

The default run deterministically generates 240 synthetic content records, 80 synthetic users, and
their impression/click/like/save/hide events. Preference propensities are controlled by persona,
topic, asset, difficulty, and content quality. Small raw CSV fixtures are included only for teaching.

## Models

Six strategies are compared: time-decayed popularity, TF-IDF content-based retrieval, item-item
collaborative filtering, hybrid recall, logistic regression ranking, and diversity reranking.

## Evaluation

Each user's latest positive item is held out. All events for that pair are removed from training.
Metrics include Hit/Recall@10, NDCG@10, catalog coverage, source/topic coverage, and intra-list
diversity.

## Known limitations

- Synthetic preferences make results useful for implementation testing, not impact estimation.
- Leave-one-out evaluation has one relevant item per user and therefore Hit Rate equals Recall.
- The learned ranker uses heuristic labels and lacks production impression-position features.
- TF-IDF does not fully represent semantic meaning, multilingual content, or rapidly shifting events.
- Quality is a synthetic proxy; financial safety and factuality need independent systems and review.

## Responsible deployment requirements

A real deployment would require consented event data, privacy and compliance review, bot and abuse
filtering, policy-aware candidate eligibility, source-quality controls, online A/B testing, latency
and drift monitoring, rollback procedures, and subgroup audits across language and creator cohorts.
