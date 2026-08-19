from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42
TOP_K = 10

EVENT_WEIGHTS = {
    "impression": 0.0,
    "click": 1.0,
    "like": 2.0,
    "save": 3.0,
    "hide": -2.0,
}

TOPICS = (
    "bitcoin",
    "ethereum",
    "defi",
    "stablecoin",
    "security",
    "regulation",
    "nft",
    "creator-economy",
    "development",
    "macro",
    "on-chain",
    "education",
)

COINS = ("BTC", "ETH", "BNB", "SOL", "USDT", "USDC", "ADA", "XRP")
DIFFICULTIES = ("beginner", "intermediate", "advanced")
CONTENT_TYPES = ("news", "education", "analysis", "opinion", "research", "announcement")
