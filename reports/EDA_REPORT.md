# Exploratory Data Analysis

## Dataset overview

- Content items: **240**
- Synthetic users: **80**
- Interaction events: **8,564**
- Impression-to-click proxy: **31.22%**

All behavior is synthetic and exists to exercise the system and evaluation pipeline.

## Event funnel

| event_type   |   events |   users |
|:-------------|---------:|--------:|
| impression   |     5760 |      80 |
| click        |     1798 |      80 |
| hide         |      568 |      80 |
| like         |      251 |      76 |
| save         |      187 |      67 |

## Engagement by persona

| persona                        |   impressions |   clicks |   meaningful_actions |   ctr_proxy |
|:-------------------------------|--------------:|---------:|---------------------:|------------:|
| Ethereum Developer             |           936 |      343 |                   71 |      0.3665 |
| Beginner Education Explorer    |          1008 |      341 |                   77 |      0.3383 |
| DeFi Explorer                  |           936 |      280 |                   69 |      0.2991 |
| NFT and Creator Economy Reader |           936 |      276 |                   70 |      0.2949 |
| Bitcoin and Macro Reader       |          1008 |      296 |                   82 |      0.2937 |
| Multi-topic Discovery User     |           936 |      262 |                   69 |      0.2799 |

## Source performance

| source               |   impressions |   clicks |   hides |   mean_quality |
|:---------------------|--------------:|---------:|--------:|---------------:|
| Block Learning Lab   |           753 |      256 |      66 |         0.7726 |
| Open Crypto Desk     |           741 |      244 |      73 |         0.7626 |
| Onchain Observer     |           744 |      235 |      70 |         0.7912 |
| Digital Asset Brief  |           761 |      231 |      70 |         0.7632 |
| Protocol Academy     |           727 |      226 |      74 |         0.7968 |
| Chain Research Daily |           689 |      224 |      67 |         0.7355 |
| Web3 Builder Journal |           684 |      200 |      75 |         0.737  |
| Creator Economy Wire |           661 |      182 |      73 |         0.7519 |

## Data-quality observations

- IDs are unique and all interaction foreign keys resolve.
- All timestamps are timezone-aware UTC values.
- Topic and coin tags use normalized pipe-separated values.
- Exposure contains both preference-biased and exploratory impressions.
- Synthetic propensities create controlled signal; they are not estimates of real users.

## SQL skills demonstrated

The analysis uses DuckDB joins, grouped aggregation, filtered counts, distinct counts,
safe division with `NULLIF`, date truncation, and ordered product metrics.
