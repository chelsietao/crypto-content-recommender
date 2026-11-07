# Offline Recommendation Evaluation

Training interactions and the content catalog are frozen at one global cutoff, and each evaluated user contributes the first eligible positive item after that cutoff.
Targets were available at the cutoff and absent from that user's training history, and hidden pairs are excluded from positive labels.

The ranker uses a separate historical feature snapshot within the training period, and labels come from the subsequent window.

## Results

| model                      |   hit_rate@10 |   recall@10 |   ndcg@10 |   candidate_recall |   catalog_coverage |   intra_list_diversity |   mean_topic_coverage |   mean_source_coverage |   evaluated_users |
|:---------------------------|--------------:|------------:|----------:|-------------------:|-------------------:|-----------------------:|----------------------:|-----------------------:|------------------:|
| v1_popularity              |        0.1333 |      0.1333 |    0.0525 |                  1 |             0.1005 |                 0.8657 |                9.9    |                 6.2333 |                30 |
| v2_content_based           |        0.1333 |      0.1333 |    0.0711 |                  1 |             0.6884 |                 0.6988 |                7.3667 |                 4.2    |                30 |
| v3_collaborative_filtering |        0.0333 |      0.0333 |    0.0105 |                  1 |             0.7638 |                 0.8753 |                9.5667 |                 5.9333 |                30 |
| v4_hybrid                  |        0.1333 |      0.1333 |    0.0617 |                  1 |             0.6281 |                 0.7501 |                7.6667 |                 4.4333 |                30 |
| v5_logistic_ranker         |        0.6667 |      0.6667 |    0.3981 |                  1 |             0.4322 |                 0.8163 |                8.6    |                 5.5    |                30 |
| v6_diversity_reranking     |        0.6667 |      0.6667 |    0.3991 |                  1 |             0.4372 |                 0.8978 |               10.8333 |                 6.4    |                30 |

Hit Rate and Recall match because evaluation uses one target per user, and candidate recall records whether that target entered the model's candidate pool.
Catalog coverage uses the frozen content catalog as its denominator, and diversity uses the average pairwise topic-tag distance within each result list.
