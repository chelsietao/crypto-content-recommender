"""Deterministic synthetic data generation, loading, and validation.

The project intentionally uses synthetic users and interactions because real
platform behavior is private. The generator creates controlled preference
patterns so recommender behavior can be tested without claiming real-world
impact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    COINS,
    CONTENT_TYPES,
    DIFFICULTIES,
    EVENT_WEIGHTS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TOPICS,
)


@dataclass(frozen=True)
class DatasetBundle:
    contents: pd.DataFrame
    users: pd.DataFrame
    interactions: pd.DataFrame


PERSONAS = {
    "Beginner Education Explorer": {
        "topics": ["education", "security", "bitcoin"],
        "coins": ["BTC", "ETH"],
        "difficulty": "beginner",
    },
    "Bitcoin and Macro Reader": {
        "topics": ["bitcoin", "macro", "regulation"],
        "coins": ["BTC"],
        "difficulty": "intermediate",
    },
    "Ethereum Developer": {
        "topics": ["ethereum", "development", "security"],
        "coins": ["ETH"],
        "difficulty": "advanced",
    },
    "DeFi Explorer": {
        "topics": ["defi", "stablecoin", "on-chain"],
        "coins": ["ETH", "USDT", "USDC"],
        "difficulty": "intermediate",
    },
    "NFT and Creator Economy Reader": {
        "topics": ["nft", "creator-economy", "ethereum"],
        "coins": ["ETH", "SOL"],
        "difficulty": "beginner",
    },
    "Multi-topic Discovery User": {
        "topics": ["education", "defi", "macro", "on-chain"],
        "coins": ["BTC", "ETH", "BNB", "SOL"],
        "difficulty": "intermediate",
    },
}

SOURCE_NAMES = (
    "Block Learning Lab",
    "Chain Research Daily",
    "Open Crypto Desk",
    "Web3 Builder Journal",
    "Digital Asset Brief",
    "Onchain Observer",
    "Protocol Academy",
    "Creator Economy Wire",
)

TITLE_TEMPLATES = (
    "A Practical Guide to {topic}",
    "Understanding {topic} in Today's Crypto Ecosystem",
    "What New Users Should Know About {topic}",
    "The Latest Developments in {topic}",
    "Risks and Opportunities in {topic}",
    "How {topic} Is Changing Digital Assets",
)


def _join(values: list[str]) -> str:
    return "|".join(values)


def _split(value: object) -> set[str]:
    if pd.isna(value) or value == "":
        return set()
    return {part for part in str(value).split("|") if part}


def generate_contents(n_contents: int = 240, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows: list[dict[str, object]] = []

    for index in range(n_contents):
        primary_topic = TOPICS[index % len(TOPICS)]
        remaining = [topic for topic in TOPICS if topic != primary_topic]
        secondary = str(rng.choice(remaining))
        topic_tags = [primary_topic, secondary]

        if primary_topic == "bitcoin":
            coin_tags = ["BTC"]
        elif primary_topic in {"ethereum", "development", "defi", "nft"}:
            coin_tags = ["ETH"]
        elif primary_topic == "stablecoin":
            coin_tags = ["USDT", "USDC"]
        else:
            coin_tags = [str(rng.choice(COINS))] if rng.random() < 0.72 else []

        difficulty = str(rng.choice(DIFFICULTIES, p=[0.46, 0.38, 0.16]))
        content_type = str(rng.choice(CONTENT_TYPES, p=[0.24, 0.28, 0.23, 0.09, 0.10, 0.06]))
        source = SOURCE_NAMES[index % len(SOURCE_NAMES)]
        published_at = start + pd.Timedelta(hours=int(index * 8 + rng.integers(0, 6)))
        display_topic = primary_topic.replace("-", " ").title()
        title = TITLE_TEMPLATES[index % len(TITLE_TEMPLATES)].format(topic=display_topic)
        summary = (
            f"A {difficulty} {content_type} item explaining {display_topic}, "
            f"with context on {secondary.replace('-', ' ')} and responsible risk awareness."
        )
        rows.append(
            {
                "content_id": f"C{index + 1:04d}",
                "title": title,
                "summary": summary,
                "published_at": published_at,
                "source": source,
                "url": f"https://example.com/research-content/{index + 1:04d}",
                "language": "en",
                "topic_tags": _join(topic_tags),
                "coin_tags": _join(coin_tags),
                "difficulty": difficulty,
                "content_type": content_type,
                "quality_score": round(float(rng.beta(7, 2)), 4),
            }
        )
    return pd.DataFrame(rows)


def generate_users(n_users: int = 80, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    names = list(PERSONAS)
    rows: list[dict[str, object]] = []
    for index in range(n_users):
        persona = names[index % len(names)]
        profile = PERSONAS[persona]
        rows.append(
            {
                "user_id": f"U{index + 1:04d}",
                "persona": persona,
                "experience_level": profile["difficulty"],
                "preferred_topics": _join(profile["topics"]),
                "preferred_coins": _join(profile["coins"]),
                "preferred_language": "en",
                "created_at": pd.Timestamp("2025-12-01T00:00:00Z")
                + pd.Timedelta(days=int(rng.integers(0, 25))),
            }
        )
    return pd.DataFrame(rows)


def _affinity(user: pd.Series, content: pd.Series) -> float:
    user_topics = _split(user["preferred_topics"])
    user_coins = _split(user["preferred_coins"])
    content_topics = _split(content["topic_tags"])
    content_coins = _split(content["coin_tags"])
    topic_match = len(user_topics & content_topics) / max(len(user_topics), 1)
    coin_match = len(user_coins & content_coins) / max(len(user_coins), 1)
    difficulty_match = float(user["experience_level"] == content["difficulty"])
    return 0.58 * topic_match + 0.24 * coin_match + 0.10 * difficulty_match + 0.08 * float(
        content["quality_score"]
    )


def generate_interactions(
    users: pd.DataFrame,
    contents: pd.DataFrame,
    impressions_per_user: int = 72,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    rows: list[dict[str, object]] = []
    event_number = 1
    content_count = len(contents)

    for user_number, (_, user) in enumerate(users.iterrows(), start=1):
        # Mix preference-biased exposure with exploration to prevent a trivial dataset.
        affinities = np.array([_affinity(user, row) for _, row in contents.iterrows()])
        probabilities = np.exp(2.7 * affinities)
        probabilities /= probabilities.sum()
        selected = rng.choice(
            content_count,
            size=min(impressions_per_user, content_count),
            replace=False,
            p=probabilities,
        )
        selected = sorted(selected, key=lambda idx: contents.iloc[idx]["published_at"])

        for session_position, content_index in enumerate(selected):
            content = contents.iloc[int(content_index)]
            event_time = pd.Timestamp(content["published_at"]) + pd.Timedelta(
                hours=int(rng.integers(4, 72)), minutes=int(rng.integers(0, 60))
            )
            session_id = f"S{user_number:04d}-{session_position // 6 + 1:03d}"
            common = {
                "user_id": user["user_id"],
                "content_id": content["content_id"],
                "session_id": session_id,
            }
            rows.append(
                {
                    "event_id": f"E{event_number:07d}",
                    **common,
                    "event_type": "impression",
                    "timestamp": event_time,
                }
            )
            event_number += 1

            affinity = _affinity(user, content)
            click_probability = np.clip(0.08 + 0.78 * affinity, 0.03, 0.92)
            if rng.random() < click_probability:
                rows.append(
                    {
                        "event_id": f"E{event_number:07d}",
                        **common,
                        "event_type": "click",
                        "timestamp": event_time + pd.Timedelta(seconds=int(rng.integers(5, 90))),
                    }
                )
                event_number += 1
                if rng.random() < np.clip(0.08 + 0.42 * affinity, 0.04, 0.65):
                    event_type = "save" if rng.random() < 0.42 else "like"
                    rows.append(
                        {
                            "event_id": f"E{event_number:07d}",
                            **common,
                            "event_type": event_type,
                            "timestamp": event_time + pd.Timedelta(minutes=int(rng.integers(1, 8))),
                        }
                    )
                    event_number += 1
            elif rng.random() < np.clip(0.18 - 0.15 * affinity, 0.01, 0.18):
                rows.append(
                    {
                        "event_id": f"E{event_number:07d}",
                        **common,
                        "event_type": "hide",
                        "timestamp": event_time + pd.Timedelta(seconds=int(rng.integers(2, 45))),
                    }
                )
                event_number += 1

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def validate_bundle(bundle: DatasetBundle) -> None:
    contents, users, interactions = bundle.contents, bundle.users, bundle.interactions
    required_content = {
        "content_id",
        "title",
        "summary",
        "published_at",
        "source",
        "url",
        "language",
        "topic_tags",
        "coin_tags",
        "difficulty",
        "content_type",
        "quality_score",
    }
    required_users = {
        "user_id",
        "persona",
        "experience_level",
        "preferred_topics",
        "preferred_coins",
        "preferred_language",
        "created_at",
    }
    required_events = {"event_id", "user_id", "content_id", "event_type", "timestamp", "session_id"}

    assert required_content.issubset(contents.columns)
    assert required_users.issubset(users.columns)
    assert required_events.issubset(interactions.columns)
    assert contents["content_id"].is_unique
    assert contents["url"].is_unique
    assert users["user_id"].is_unique
    assert interactions["event_id"].is_unique
    assert set(interactions["event_type"]).issubset(EVENT_WEIGHTS)
    assert set(interactions["user_id"]).issubset(set(users["user_id"]))
    assert set(interactions["content_id"]).issubset(set(contents["content_id"]))
    assert contents["published_at"].notna().all()
    assert interactions["timestamp"].notna().all()
    assert contents["title"].str.len().gt(0).all()
    assert contents["topic_tags"].str.len().gt(0).all()


def save_bundle(bundle: DatasetBundle, directory: Path = PROCESSED_DATA_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    bundle.contents.to_csv(directory / "contents.csv", index=False)
    bundle.users.to_csv(directory / "users.csv", index=False)
    bundle.interactions.to_csv(directory / "interactions.csv", index=False)


def load_bundle(directory: Path = PROCESSED_DATA_DIR) -> DatasetBundle:
    bundle = DatasetBundle(
        contents=pd.read_csv(directory / "contents.csv", parse_dates=["published_at"]),
        users=pd.read_csv(directory / "users.csv", parse_dates=["created_at"]),
        interactions=pd.read_csv(directory / "interactions.csv", parse_dates=["timestamp"]),
    )
    validate_bundle(bundle)
    return bundle


def build_dataset(
    n_contents: int = 240,
    n_users: int = 80,
    impressions_per_user: int = 72,
    save: bool = True,
) -> DatasetBundle:
    contents = generate_contents(n_contents=n_contents)
    users = generate_users(n_users=n_users)
    interactions = generate_interactions(
        users=users,
        contents=contents,
        impressions_per_user=impressions_per_user,
    )
    bundle = DatasetBundle(contents=contents, users=users, interactions=interactions)
    validate_bundle(bundle)
    if save:
        save_bundle(bundle)
    return bundle


if __name__ == "__main__":
    generated = build_dataset()
    print(
        f"Generated {len(generated.contents)} contents, {len(generated.users)} users, "
        f"and {len(generated.interactions)} events."
    )
