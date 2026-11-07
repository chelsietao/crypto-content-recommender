# PyTorch Recommendation Experiment

The run trains two towers on impression outcomes, then freezes their weights before training a neural ranker.
Labels are observed within 15 minutes of each impression, and user features use events strictly before the request.

## Training

|               |   selected_epoch |   validation_loss | parameters_changed   |   training_examples |   validation_examples |
|:--------------|-----------------:|------------------:|:---------------------|--------------------:|----------------------:|
| two_tower     |                4 |          0.945872 | True                 |                2345 |                   689 |
| neural_ranker |               18 |          0.810582 | True                 |                 533 |                   394 |

## Offline evaluation

| model                      |   hit_rate@10 |   recall@10 |   ndcg@10 |   candidate_recall |   catalog_coverage |   intra_list_diversity |   mean_topic_coverage |   mean_source_coverage |   evaluated_users |
|:---------------------------|--------------:|------------:|----------:|-------------------:|-------------------:|-----------------------:|----------------------:|-----------------------:|------------------:|
| v1_popularity              |        0.1333 |      0.1333 |    0.0525 |             1      |             0.1005 |                 0.8657 |                9.9    |                 6.2333 |                30 |
| v2_content_based           |        0.1333 |      0.1333 |    0.0711 |             1      |             0.6884 |                 0.6988 |                7.3667 |                 4.2    |                30 |
| v3_collaborative_filtering |        0.0333 |      0.0333 |    0.0105 |             1      |             0.7638 |                 0.8753 |                9.5667 |                 5.9333 |                30 |
| v4_hybrid                  |        0.1333 |      0.1333 |    0.0617 |             1      |             0.6281 |                 0.7501 |                7.6667 |                 4.4333 |                30 |
| v5_logistic_ranker         |        0.6667 |      0.6667 |    0.3981 |             1      |             0.4322 |                 0.8163 |                8.6    |                 5.5    |                30 |
| v6_diversity_reranking     |        0.6667 |      0.6667 |    0.3991 |             1      |             0.4372 |                 0.8978 |               10.8333 |                 6.4    |                30 |
| v7_pytorch_two_tower       |        0.1333 |      0.1333 |    0.0452 |             0.7333 |             0.4673 |                 0.7462 |                7.1667 |                 4.7    |                30 |
| v8_pytorch_neural_ranker   |        0.2333 |      0.2333 |    0.1013 |             0.7333 |             0.4724 |                 0.7094 |                7.1    |                 4.4333 |                30 |

The checkpoint reload was checked for 81 users, with maximum retrieval-score error 0.0.
The summary records CPU execution, configuration, chronological boundaries, label counts, and file hashes.
