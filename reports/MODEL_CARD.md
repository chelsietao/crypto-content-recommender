# Recommendation Model Specification

The project compares six recommendation strategies on a fixed synthetic dataset, using Python, scikit-learn, SciPy, pandas, and DuckDB.

## Data and labels

The dataset contains 240 content records and 80 users, with 8,564 impression, click, like, save, and hide events.
A user-content pair receives its strongest positive event weight when no hide is present, and receives the negative hide weight when a hide is present.

## Evaluation

A global timestamp separates training from evaluation, and the content catalog contains only items published by that timestamp.
Each evaluated user contributes the first later positive item absent from their training history, with hidden pairs excluded from positive targets.
The ranker uses an earlier feature snapshot and a later label window within the training period, with candidate membership determined by the earlier snapshot.

## Model flow

Popularity, TF-IDF interests, sparse item neighbors, and exploration each retrieve up to 20 unseen items, with their union forming the hybrid candidate pool.
Logistic regression scores that pool, and diversity reranking applies source and primary-topic caps.
The item-neighbor matrix retains up to 40 positive similarities per item, with similarity computation performed in blocks.

## Recorded outputs

The pipeline writes recommendation metrics and per-user details, including candidate recall, coverage, diversity, model coefficients, and all missed targets.
The Streamlit app displays synthetic-user feeds and evaluation dashboards, with feedback retained in browser session state.
