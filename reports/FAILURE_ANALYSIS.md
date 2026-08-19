# Recommendation Failure Analysis

This report samples missed held-out items and assigns a testable failure hypothesis.
The labels are diagnostic heuristics, not verified root causes.

## Failure categories

| model            | failure_hypothesis                                                             |   examples |
|:-----------------|:-------------------------------------------------------------------------------|-----------:|
| v1_popularity    | Global popularity did not capture the user's niche interest.                   |         65 |
| v2_content_based | Recommended pool may not match the user's preferred difficulty.                |          9 |
| v2_content_based | The held-out interest was not represented strongly enough in candidate scores. |          6 |

## Sample cases

| model         | user_id   | persona                        | target_content_id   | title                                                     | topic_tags              | coin_tags   | failure_hypothesis                                           |
|:--------------|:----------|:-------------------------------|:--------------------|:----------------------------------------------------------|:------------------------|:------------|:-------------------------------------------------------------|
| v1_popularity | U0053     | NFT and Creator Economy Reader | C0219               | What New Users Should Know About Defi                     | defi|on-chain           | ETH         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0022     | DeFi Explorer                  | C0219               | What New Users Should Know About Defi                     | defi|on-chain           | ETH         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0072     | Multi-topic Discovery User     | C0214               | The Latest Developments in Macro                          | macro|ethereum          | ETH         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0030     | Multi-topic Discovery User     | C0214               | The Latest Developments in Macro                          | macro|ethereum          | ETH         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0018     | Multi-topic Discovery User     | C0219               | What New Users Should Know About Defi                     | defi|on-chain           | ETH         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0044     | Bitcoin and Macro Reader       | C0222               | How Regulation Is Changing Digital Assets                 | regulation|education    |             | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0058     | DeFi Explorer                  | C0226               | The Latest Developments in Macro                          | macro|defi              |             | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0062     | Bitcoin and Macro Reader       | C0224               | Understanding Creator Economy in Today's Crypto Ecosystem | creator-economy|macro   | SOL         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0004     | DeFi Explorer                  | C0223               | A Practical Guide to Nft                                  | nft|on-chain            | ETH         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0012     | Multi-topic Discovery User     | C0228               | How Education Is Changing Digital Assets                  | education|regulation    |             | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0017     | NFT and Creator Economy Reader | C0224               | Understanding Creator Economy in Today's Crypto Ecosystem | creator-economy|macro   | SOL         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0032     | Bitcoin and Macro Reader       | C0229               | A Practical Guide to Bitcoin                              | bitcoin|creator-economy | BTC         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0049     | Beginner Education Explorer    | C0226               | The Latest Developments in Macro                          | macro|defi              |             | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0002     | Bitcoin and Macro Reader       | C0229               | A Practical Guide to Bitcoin                              | bitcoin|creator-economy | BTC         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0028     | DeFi Explorer                  | C0228               | How Education Is Changing Digital Assets                  | education|regulation    |             | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0050     | Bitcoin and Macro Reader       | C0229               | A Practical Guide to Bitcoin                              | bitcoin|creator-economy | BTC         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0037     | Beginner Education Explorer    | C0229               | A Practical Guide to Bitcoin                              | bitcoin|creator-economy | BTC         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0020     | Bitcoin and Macro Reader       | C0224               | Understanding Creator Economy in Today's Crypto Ecosystem | creator-economy|macro   | SOL         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0026     | Bitcoin and Macro Reader       | C0227               | Risks and Opportunities in On Chain                       | on-chain|stablecoin     | XRP         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0056     | Bitcoin and Macro Reader       | C0231               | What New Users Should Know About Defi                     | defi|on-chain           | ETH         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0074     | Bitcoin and Macro Reader       | C0229               | A Practical Guide to Bitcoin                              | bitcoin|creator-economy | BTC         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0075     | Ethereum Developer             | C0233               | Risks and Opportunities in Security                       | security|ethereum       |             | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0009     | Ethereum Developer             | C0233               | Risks and Opportunities in Security                       | security|ethereum       |             | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0064     | DeFi Explorer                  | C0234               | How Regulation Is Changing Digital Assets                 | regulation|stablecoin   | ETH         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0063     | Ethereum Developer             | C0234               | How Regulation Is Changing Digital Assets                 | regulation|stablecoin   | ETH         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0001     | Beginner Education Explorer    | C0229               | A Practical Guide to Bitcoin                              | bitcoin|creator-economy | BTC         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0021     | Ethereum Developer             | C0227               | Risks and Opportunities in On Chain                       | on-chain|stablecoin     | XRP         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0036     | Multi-topic Discovery User     | C0232               | The Latest Developments in Stablecoin                     | stablecoin|development  | USDT|USDC   | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0079     | Beginner Education Explorer    | C0229               | A Practical Guide to Bitcoin                              | bitcoin|creator-economy | BTC         | Global popularity did not capture the user's niche interest. |
| v1_popularity | U0055     | Beginner Education Explorer    | C0234               | How Regulation Is Changing Digital Assets                 | regulation|stablecoin   | ETH         | Global popularity did not capture the user's niche interest. |

## Follow-up experiments

1. Add position-aware negative sampling from eligible impressions.
2. Compare short-term and long-term interest profiles.
3. Calibrate ranker probabilities by persona and activity cohort.
4. Add semantic embeddings and multilingual content representations.
5. Test source-quality and safety constraints as explicit guardrails.
