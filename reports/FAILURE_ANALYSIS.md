# Recommendation Miss Analysis

Every missed target is retained in the CSV, with its candidate-pool membership recorded.
A value of 0 identifies a recall-stage miss, and a value of 1 identifies a target that entered the pool but did not appear in the final result list.

## Counts

| model                      |   target_in_candidates |   missed_targets |
|:---------------------------|-----------------------:|-----------------:|
| v1_popularity              |                      1 |               26 |
| v2_content_based           |                      1 |               26 |
| v3_collaborative_filtering |                      1 |               29 |
| v4_hybrid                  |                      1 |               26 |
| v5_logistic_ranker         |                      1 |               10 |
| v6_diversity_reranking     |                      1 |               10 |

## Sample records

| model                      | user_id   | persona                        | target_content_id   | title                                              | topic_tags            | coin_tags   |   target_in_candidates |
|:---------------------------|:----------|:-------------------------------|:--------------------|:---------------------------------------------------|:----------------------|:------------|-----------------------:|
| v1_popularity              | U0056     | Bitcoin and Macro Reader       | C0193               | A Practical Guide to Bitcoin                       | bitcoin|development   | BTC         |                      1 |
| v1_popularity              | U0030     | Multi-topic Discovery User     | C0192               | How Education Is Changing Digital Assets           | education|defi        | BNB         |                      1 |
| v1_popularity              | U0045     | Ethereum Developer             | C0198               | How Regulation Is Changing Digital Assets          | regulation|security   | BNB         |                      1 |
| v1_popularity              | U0004     | DeFi Explorer                  | C0194               | Understanding Ethereum in Today's Crypto Ecosystem | ethereum|macro        | ETH         |                      1 |
| v1_popularity              | U0043     | Beginner Education Explorer    | C0197               | Risks and Opportunities in Security                | security|regulation   |             |                      1 |
| v2_content_based           | U0056     | Bitcoin and Macro Reader       | C0193               | A Practical Guide to Bitcoin                       | bitcoin|development   | BTC         |                      1 |
| v2_content_based           | U0030     | Multi-topic Discovery User     | C0192               | How Education Is Changing Digital Assets           | education|defi        | BNB         |                      1 |
| v2_content_based           | U0045     | Ethereum Developer             | C0198               | How Regulation Is Changing Digital Assets          | regulation|security   | BNB         |                      1 |
| v2_content_based           | U0004     | DeFi Explorer                  | C0194               | Understanding Ethereum in Today's Crypto Ecosystem | ethereum|macro        | ETH         |                      1 |
| v2_content_based           | U0010     | DeFi Explorer                  | C0195               | What New Users Should Know About Defi              | defi|security         | ETH         |                      1 |
| v3_collaborative_filtering | U0056     | Bitcoin and Macro Reader       | C0193               | A Practical Guide to Bitcoin                       | bitcoin|development   | BTC         |                      1 |
| v3_collaborative_filtering | U0030     | Multi-topic Discovery User     | C0192               | How Education Is Changing Digital Assets           | education|defi        | BNB         |                      1 |
| v3_collaborative_filtering | U0045     | Ethereum Developer             | C0198               | How Regulation Is Changing Digital Assets          | regulation|security   | BNB         |                      1 |
| v3_collaborative_filtering | U0004     | DeFi Explorer                  | C0194               | Understanding Ethereum in Today's Crypto Ecosystem | ethereum|macro        | ETH         |                      1 |
| v3_collaborative_filtering | U0010     | DeFi Explorer                  | C0195               | What New Users Should Know About Defi              | defi|security         | ETH         |                      1 |
| v4_hybrid                  | U0030     | Multi-topic Discovery User     | C0192               | How Education Is Changing Digital Assets           | education|defi        | BNB         |                      1 |
| v4_hybrid                  | U0045     | Ethereum Developer             | C0198               | How Regulation Is Changing Digital Assets          | regulation|security   | BNB         |                      1 |
| v4_hybrid                  | U0004     | DeFi Explorer                  | C0194               | Understanding Ethereum in Today's Crypto Ecosystem | ethereum|macro        | ETH         |                      1 |
| v4_hybrid                  | U0010     | DeFi Explorer                  | C0195               | What New Users Should Know About Defi              | defi|security         | ETH         |                      1 |
| v4_hybrid                  | U0043     | Beginner Education Explorer    | C0197               | Risks and Opportunities in Security                | security|regulation   |             |                      1 |
| v5_logistic_ranker         | U0004     | DeFi Explorer                  | C0194               | Understanding Ethereum in Today's Crypto Ecosystem | ethereum|macro        | ETH         |                      1 |
| v5_logistic_ranker         | U0053     | NFT and Creator Economy Reader | C0196               | The Latest Developments in Stablecoin              | stablecoin|regulation | USDT|USDC   |                      1 |
| v5_logistic_ranker         | U0052     | DeFi Explorer                  | C0193               | A Practical Guide to Bitcoin                       | bitcoin|development   | BTC         |                      1 |
| v5_logistic_ranker         | U0054     | Multi-topic Discovery User     | C0196               | The Latest Developments in Stablecoin              | stablecoin|regulation | USDT|USDC   |                      1 |
| v5_logistic_ranker         | U0019     | Beginner Education Explorer    | C0194               | Understanding Ethereum in Today's Crypto Ecosystem | ethereum|macro        | ETH         |                      1 |
| v6_diversity_reranking     | U0045     | Ethereum Developer             | C0198               | How Regulation Is Changing Digital Assets          | regulation|security   | BNB         |                      1 |
| v6_diversity_reranking     | U0053     | NFT and Creator Economy Reader | C0196               | The Latest Developments in Stablecoin              | stablecoin|regulation | USDT|USDC   |                      1 |
| v6_diversity_reranking     | U0052     | DeFi Explorer                  | C0193               | A Practical Guide to Bitcoin                       | bitcoin|development   | BTC         |                      1 |
| v6_diversity_reranking     | U0063     | Ethereum Developer             | C0198               | How Regulation Is Changing Digital Assets          | regulation|security   | BNB         |                      1 |
| v6_diversity_reranking     | U0019     | Beginner Education Explorer    | C0194               | Understanding Ethereum in Today's Crypto Ecosystem | ethereum|macro        | ETH         |                      1 |
