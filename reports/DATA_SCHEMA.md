# Data Schema

## 1. Purpose

This document defines the structure, meaning, constraints, and relationships of the datasets used in the Crypto Content Recommender project.

The recommendation system uses three main entities:

1. **Users** — who receives recommendations.
2. **Contents** — what can be recommended.
3. **Interactions** — how users respond to content.

The schema is designed for an initial prototype using public content metadata and synthetic user behavior.

---

## 2. Dataset Overview

| Dataset | Processed File | One Row Represents |
|---|---|---|
| Contents | `data/processed/contents.csv` | One recommendable content item |
| Users | `data/processed/users.csv` | One synthetic user profile |
| Interactions | `data/processed/interactions.csv` | One user-content interaction event |

Raw or manually collected data should be stored under:

```text
data/raw/
```

Cleaned and validated datasets should be stored under:

```text
data/processed/
```

---

## 3. Content Dataset

### File

```text
data/processed/contents.csv
```

### Row Definition

Each row represents one unique crypto-related content item that may be recommended to a user.

### Schema

| Column | Type | Required | Key | Description |
|---|---|---:|---|---|
| `content_id` | string | Yes | Primary Key | Unique identifier for one content item |
| `title` | string | Yes | — | Original content title |
| `summary` | string | Yes | — | Short plain-text summary |
| `published_at` | datetime | Yes | — | UTC publication timestamp |
| `source` | string | Yes | — | Publisher or content provider |
| `url` | string | Yes | Unique | Original content URL |
| `language` | string | Yes | — | ISO 639-1 language code |
| `topic_tags` | string | Yes | — | Pipe-separated topic labels |
| `coin_tags` | string | No | — | Pipe-separated crypto asset symbols |
| `difficulty` | string | No | — | Estimated knowledge difficulty |
| `content_type` | string | Yes | — | General category of the content |

### Example

```csv
content_id,title,summary,published_at,source,url,language,topic_tags,coin_tags,difficulty,content_type
C0001,What Is Bitcoin?,A beginner introduction to Bitcoin and blockchain technology.,2026-07-20T08:00:00Z,Example Source,https://example.com/bitcoin,en,bitcoin|education,BTC,beginner,education
```

### Allowed Values

#### `language`

Initial version:

```text
en
```

Future versions may support additional ISO 639-1 codes such as:

```text
zh
ja
ko
```

#### `difficulty`

```text
beginner
intermediate
advanced
```

The value may be empty if difficulty has not been assigned.

#### `content_type`

```text
news
education
analysis
opinion
research
announcement
```

### Validation Rules

- `content_id` must be unique and non-empty.
- `content_id` should match the pattern `C` followed by four or more digits, such as `C0001`.
- `url` must be unique after URL normalization.
- `title`, `summary`, `published_at`, `source`, `url`, `language`, and `content_type` cannot be empty.
- `published_at` must be convertible to a valid UTC datetime.
- `language` must contain a valid supported language code.
- `difficulty` must be one of the allowed values when present.
- `content_type` must be one of the allowed values.
- `topic_tags` must contain at least one topic.
- Tags should use lowercase normalized values.
- Crypto asset symbols in `coin_tags` should use uppercase values such as `BTC` and `ETH`.
- Duplicate content items should be removed.
- Full copyrighted articles should not be stored unless permission is available.
- Only titles, metadata, short summaries, and original URLs should normally be stored.

---

## 4. User Dataset

### File

```text
data/processed/users.csv
```

### Row Definition

Each row represents one unique synthetic user profile.

These users are created for system testing and do not represent real Binance users.

### Schema

| Column | Type | Required | Key | Description |
|---|---|---:|---|---|
| `user_id` | string | Yes | Primary Key | Unique synthetic user identifier |
| `persona` | string | Yes | — | User behavior archetype |
| `experience_level` | string | Yes | — | Estimated crypto knowledge level |
| `preferred_topics` | string | Yes | — | Pipe-separated topic preferences |
| `preferred_coins` | string | No | — | Pipe-separated asset preferences |
| `preferred_language` | string | Yes | — | Preferred ISO 639-1 language code |
| `created_at` | datetime | Yes | — | Synthetic account creation timestamp |

### Example

```csv
user_id,persona,experience_level,preferred_topics,preferred_coins,preferred_language,created_at
U0001,Beginner Education Explorer,beginner,education|security|bitcoin,BTC|ETH,en,2026-07-01T09:00:00Z
```

### Allowed Values

#### `experience_level`

```text
beginner
intermediate
advanced
```

#### `preferred_language`

Initial version:

```text
en
```

### Initial Personas

```text
Beginner Education Explorer
Bitcoin and Macro Reader
Ethereum Developer
DeFi Explorer
NFT and Creator Economy Reader
Multi-topic Discovery User
```

These personas are initial assumptions used to generate controlled synthetic behavior.

### Validation Rules

- `user_id` must be unique and non-empty.
- `user_id` should match the pattern `U` followed by four or more digits, such as `U0001`.
- `persona`, `experience_level`, `preferred_topics`, `preferred_language`, and `created_at` cannot be empty.
- `experience_level` must be one of the allowed values.
- `preferred_topics` must contain at least one topic.
- Topic values should use lowercase normalized labels.
- Crypto asset symbols in `preferred_coins` should use uppercase values.
- `created_at` must be convertible to a valid UTC datetime.
- Synthetic users must not contain real names, email addresses, phone numbers, wallet addresses, or other personal data.

---

## 5. Interaction Dataset

### File

```text
data/processed/interactions.csv
```

### Row Definition

Each row represents one event generated when one user interacts with one content item.

A user may interact with the same content item more than once, but each event must have a different `event_id`.

### Schema

| Column | Type | Required | Key | Description |
|---|---|---:|---|---|
| `event_id` | string | Yes | Primary Key | Unique identifier for one interaction event |
| `user_id` | string | Yes | Foreign Key | References `users.user_id` |
| `content_id` | string | Yes | Foreign Key | References `contents.content_id` |
| `event_type` | string | Yes | — | Type of interaction |
| `timestamp` | datetime | Yes | — | UTC timestamp of the event |
| `session_id` | string | No | — | Identifier for one browsing session |

### Example

```csv
event_id,user_id,content_id,event_type,timestamp,session_id
E0001,U0001,C0001,impression,2026-07-27T10:00:00Z,S0001
E0002,U0001,C0001,click,2026-07-27T10:00:12Z,S0001
E0003,U0001,C0001,save,2026-07-27T10:01:05Z,S0001
```

### Allowed Event Types

| Event Type | Meaning | Initial Weight |
|---|---|---:|
| `impression` | Content was displayed to the user | 0 |
| `click` | User opened the content | 1 |
| `like` | User expressed positive feedback | 2 |
| `save` | User saved the content for later | 3 |
| `hide` | User hid or rejected the content | -2 |

These weights are initial modeling assumptions, not fixed truths. They may be modified during later experiments.

### Validation Rules

- `event_id` must be unique and non-empty.
- `event_id` should match the pattern `E` followed by four or more digits, such as `E0001`.
- Every `user_id` must exist in the User Dataset.
- Every `content_id` must exist in the Content Dataset.
- `event_type` must be one of the allowed event types.
- `timestamp` must be convertible to a valid UTC datetime.
- An interaction timestamp should not occur before the corresponding user's `created_at`.
- `session_id`, when present, should match the project session ID format.
- Duplicate events with identical user, content, event type, timestamp, and session should be reviewed.
- Multiple valid events between the same user and content are allowed.
- Synthetic event sequences should follow reasonable behavior when possible.

For example:

```text
impression → click → like
```

is more realistic than:

```text
save → impression
```

when timestamps occur in that order.

---

## 6. Table Relationships

The Interaction Dataset acts as a bridge between users and content items.

```text
Users
  user_id
     |
     | one user can produce many interactions
     v
Interactions
  event_id
  user_id
  content_id
     ^
     | one content item can receive many interactions
     |
Contents
  content_id
```

### Key Relationships

```text
users.user_id
    ↓
interactions.user_id
```

`interactions.user_id` must reference an existing `users.user_id`.

```text
contents.content_id
    ↓
interactions.content_id
```

`interactions.content_id` must reference an existing `contents.content_id`.

### Relationship Types

- One user can generate many interactions.
- One content item can receive many interactions.
- Each interaction belongs to exactly one user.
- Each interaction refers to exactly one content item.
- Users and contents have a many-to-many relationship through the Interaction Dataset.

---

## 7. Key Definitions

### Primary Key

A primary key uniquely identifies one row within a table.

Primary keys in this project are:

```text
contents.content_id
users.user_id
interactions.event_id
```

Primary keys must:

- be present,
- be unique,
- remain stable,
- not be reused for different entities.

### Foreign Key

A foreign key is a column that references the primary key of another table.

Foreign keys in this project are:

```text
interactions.user_id → users.user_id
interactions.content_id → contents.content_id
```

Foreign keys prevent interactions from referring to users or contents that do not exist.

---

## 8. Missing-Value Policy

Required fields must not contain:

```text
empty strings
null
None
NaN
N/A
unknown
```

unless the schema explicitly permits the value.

Optional fields may be empty.

Examples of optional fields:

```text
contents.coin_tags
contents.difficulty
users.preferred_coins
interactions.session_id
```

Missing optional values should be stored consistently as empty CSV fields rather than mixed placeholder strings.

---

## 9. Time Standard

All datetime fields should use UTC and ISO 8601 format.

Recommended format:

```text
YYYY-MM-DDTHH:MM:SSZ
```

Example:

```text
2026-07-27T10:00:00Z
```

Relevant columns:

```text
contents.published_at
users.created_at
interactions.timestamp
```

Using one time standard prevents inconsistencies between users, content sources, and event logs.

---

## 10. Multi-Value Fields

The prototype stores multiple tags in one CSV field using the pipe character:

```text
|
```

Example:

```text
bitcoin|security|education
```

This approach is acceptable for the initial CSV prototype.

When processing the data in Python, the value may be split using:

```python
tags = value.split("|")
```

For a larger production database, topics and assets would normally be normalized into separate mapping tables.

---

## 11. Data Processing Stages

### Raw Data

Location:

```text
data/raw/
```

Raw data may contain:

- original public metadata,
- inconsistent capitalization,
- missing values,
- duplicate records,
- unvalidated timestamps,
- manually generated synthetic records.

Raw files should not be edited destructively after collection whenever reproducibility is important.

### Processed Data

Location:

```text
data/processed/
```

Processed data should contain:

- standardized column names,
- normalized tags,
- validated timestamps,
- removed duplicates,
- verified primary keys,
- verified foreign keys,
- accepted category values.

The recommendation models should normally read from the processed datasets.

---

## 12. Data Quality Checks

The validation pipeline should check at least the following:

### Contents

- Required columns exist.
- `content_id` values are unique.
- URLs are not duplicated.
- Required values are not missing.
- Datetimes are valid.
- Categories use allowed values.

### Users

- Required columns exist.
- `user_id` values are unique.
- Required values are not missing.
- Experience levels use allowed values.
- Datetimes are valid.

### Interactions

- Required columns exist.
- `event_id` values are unique.
- Event types use allowed values.
- All `user_id` values exist in Users.
- All `content_id` values exist in Contents.
- Datetimes are valid.
- Event timestamps are logically consistent.

---

## 13. Data Limitations

- User profiles and interaction events are synthetic.
- Synthetic behavior does not represent actual Binance users.
- Persona definitions are simplified assumptions.
- Synthetic event weights may not reflect real engagement value.
- Public content does not represent the entire crypto ecosystem.
- Some public metadata may contain source-specific biases.
- Topic and difficulty labels may involve subjective judgment.
- Offline evaluation cannot demonstrate real-world engagement improvement.
- Good offline metrics do not guarantee good online user experience.
- The dataset must not be used to provide personalized financial advice.
- The recommendation system does not generate trading signals or price predictions.

---

## 14. Schema Version

Current schema version:

```text
1.0
```

When columns, allowed values, or validation rules change substantially, the schema version should be updated.

Example:

```text
1.0 → Initial prototype schema
1.1 → Added new optional field
2.0 → Breaking change to table structure
```

Last updated:

```text
2026-07-27
```