# Personalized and Diverse Crypto Content Recommendation

## 1. Project Overview

This project develops and evaluates a personalized crypto-content
recommendation system for users with limited interaction history.

The system is inspired by content-community platforms such as Binance
Square, but it does not use private Binance data and is not affiliated
with Binance.

## 2. Problem Statement

Crypto users face a large volume of news, educational articles, market
commentary, and creator content. New users may struggle to find content
that is relevant to their interests and appropriate for their experience
level.

A popularity-only feed may repeatedly expose users to the same topics,
large creators, and highly popular assets. It may also provide limited
exposure to new content and smaller creators.

The research problem is:

> How can a recommendation system provide relevant, timely, and diverse
> crypto content when user interaction history is limited?

## 3. Target Users

### Primary User

Beginner retail users who:

- Have limited knowledge of crypto and Web3
- Have little or no interaction history
- Want to discover educational and market-related content
- Have limited time to browse content
- Prefer relevant information without receiving investment advice

### Secondary User

Product and content teams who want to:

- Evaluate recommendation quality
- Understand user interests
- Monitor content diversity
- Identify recommendation failures
- Improve content and creator distribution

## 4. User Value

The system aims to help users:

- Reduce information overload
- Discover relevant content more efficiently
- Receive content appropriate to their interests
- Explore a more diverse set of topics and sources
- Understand why an item was recommended

## 5. Product Value

The recommendation system may help a content platform:

- Improve meaningful user engagement
- Increase content discovery
- Support new-content and new-creator exposure
- Reduce repetitive recommendations
- Build a healthier content ecosystem

## 6. Research Questions

1. Does content-based recommendation outperform popularity ranking?
2. Can explicit topic selection improve recommendations for new users?
3. Does hybrid recommendation outperform individual recommendation methods?
4. Can diversity reranking improve topic and source coverage without
   materially reducing relevance?
5. Which recommendation method performs best for different user groups?

## 7. Initial Hypotheses

- Content-based recommendation will improve cold-start relevance compared
  with global popularity ranking.
- Collaborative filtering will perform better for users with sufficient
  interaction history.
- Hybrid recommendation will achieve the best overall Recall@10.
- Diversity reranking will improve topic and source coverage with a small
  trade-off in NDCG@10.

## 8. Proposed System

The first version will compare:

1. Popularity-based recommendation
2. Content-based recommendation
3. Collaborative filtering
4. Hybrid recommendation
5. Hybrid recommendation with diversity reranking

The system will include:

- Crypto-content collection and preprocessing
- Synthetic user-interaction generation
- User-interest profiling
- Candidate retrieval
- Content ranking
- Cold-start handling
- Diversity reranking
- Offline evaluation
- An interactive demonstration interface

## 9. Evaluation Metrics

The initial offline metrics will include:

- Recall@10
- NDCG@10
- Hit Rate@10
- Catalog Coverage
- Topic Coverage
- Source Coverage
- Intra-list Diversity

A future online experiment proposal may include:

- Click-through rate
- Save rate
- Dwell time
- Hide rate
- User retention
- Creator exposure concentration

## 10. Data Plan

The project will use publicly accessible crypto-content metadata, such as:

- Content ID
- Title
- Summary
- Publication time
- Source
- URL
- Topic tags
- Asset or coin tags

Because private platform interaction data is unavailable, user events will
initially be synthetic.

Synthetic events may include:

- Impression
- Click
- Like
- Save
- Hide

## 11. Deliverables

- Reproducible Python project
- Exploratory data analysis
- Popularity baseline
- Content-based recommender
- Collaborative-filtering recommender
- Hybrid recommendation pipeline
- Diversity reranking experiment
- Offline evaluation report
- Failure-case analysis
- Streamlit demonstration
- English GitHub README
- Short demonstration video

## 12. Non-Goals

This project will not:

- Predict cryptocurrency prices
- Generate trading signals
- Recommend assets to buy or sell
- Provide financial or investment advice
- Reproduce Binance's private recommendation system
- Claim that synthetic behavior represents actual Binance users

## 13. Limitations

- User interactions are synthetic.
- Public content may not represent all crypto communities.
- Offline metrics do not prove improvement in real-world user engagement.
- Content-quality and source-quality labels may be incomplete.
- The project is an educational portfolio project and is not affiliated
  with Binance.

## 14. Expected Timeline

The initial project is expected to take approximately 8 to 10 weeks at
8 to 12 hours per week.

## 15. Disclaimer

This project is for educational and research purposes only. It does not
provide financial or investment advice.