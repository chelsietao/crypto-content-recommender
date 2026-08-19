# A/B Test Proposal

## Objective

Determine whether the personalized, diversity-aware feed improves meaningful content discovery
relative to a time-decayed popularity feed.

## Variants

- **Control:** time-decayed popularity recommendation (V1).
- **Treatment:** logistic ranking followed by diversity reranking (V6).

## Hypothesis

Treatment increases meaningful engagement and content coverage without materially increasing
negative feedback.

## Unit and allocation

- Randomization unit: authenticated user ID.
- Allocation: 50% control / 50% treatment.
- Sticky assignment for the full experiment.
- Exclude employees, bots, and users with invalid instrumentation.

## Metrics

- Primary: meaningful engagement rate (like or save per eligible impression).
- Secondary: CTR, dwell time, session depth, day-7 return rate, creator coverage.
- Guardrails: hide/report rate, latency p95, source concentration, policy-violating exposure.

## Analysis

- Pre-register sample-size assumptions before launch.
- Compare intention-to-treat groups with 95% confidence intervals.
- Segment by new vs. established users, language, and activity level.
- Do not stop early solely because a metric becomes temporarily significant.
- Ship only if the primary metric improves and no guardrail crosses its agreed threshold.

## Instrumentation requirements

Every recommendation impression must record model version, rank, request ID, user assignment,
content ID, timestamp in UTC, latency, and subsequent interaction events.

## Caveat

This document is a product experiment proposal. No online experiment was performed in this
portfolio project.
