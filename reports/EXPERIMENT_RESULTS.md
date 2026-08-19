# Offline Experiment Results

## Evaluation design

For each synthetic user, the latest positive user-content interaction is held out.
All events for that held-out pair are removed from training. Models must rank the
held-out item in the top 10 while excluding items observed during training.
This chronological leave-one-out design is more realistic than a random split, but
results remain synthetic and do not demonstrate production impact.

## Results

| model                      |   hit_rate@10 |   recall@10 |   ndcg@10 |   catalog_coverage |   intra_list_diversity |   mean_topic_coverage |   mean_source_coverage |   evaluated_users |
|:---------------------------|--------------:|------------:|----------:|-------------------:|-----------------------:|----------------------:|-----------------------:|------------------:|
| v1_popularity              |        0.1875 |      0.1875 |    0.0921 |             0.1    |                 0.858  |                9.7375 |                 6.4    |                80 |
| v2_content_based           |        0.2375 |      0.2375 |    0.1037 |             0.8458 |                 0.6743 |                7.0875 |                 3.975  |                80 |
| v3_collaborative_filtering |        0.0625 |      0.0625 |    0.0313 |             0.8667 |                 0.8758 |                9.9125 |                 6.025  |                80 |
| v4_hybrid                  |        0.225  |      0.225  |    0.0934 |             0.775  |                 0.7216 |                7.7125 |                 4.525  |                80 |
| v5_logistic_ranker         |        0.1    |      0.1    |    0.0447 |             0.8375 |                 0.7297 |                7.8375 |                 4.4375 |                80 |
| v6_diversity_reranking     |        0.1    |      0.1    |    0.0435 |             0.875  |                 0.8559 |                9.475  |                 6.5    |                80 |

## Findings

- Best NDCG@10: **v2_content_based** (0.1037).
- Best intra-list diversity: **v3_collaborative_filtering** (0.8758).
- Popularity is a necessary baseline, not a personalized solution.
- Content-based retrieval supports new-item and low-history scenarios.
- Collaborative filtering depends on sufficient co-interaction density.
- Hybrid and ranking stages combine complementary signals.
- Diversity reranking explicitly trades a small amount of relevance for healthier exposure.

## Interpretation limits

- Users and interactions are deterministic synthetic data.
- Offline ranking metrics cannot be interpreted as real CTR or retention lift.
- A real deployment requires online experimentation, safety review, and data-quality monitoring.
