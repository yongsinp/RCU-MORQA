from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RCU_EN_PATH = PROJECT_ROOT / "data" / "rcu-en"
OUT_PATH = PROJECT_ROOT / "out"
MODELS_PATH = PROJECT_ROOT / "models"
EXTERNAL_PATH = PROJECT_ROOT / "external"
QWEN_TOKENIZER_PATH = EXTERNAL_PATH / "qwen_tokenizer"
DEEPSEEK_TOKENIZER_PATH = EXTERNAL_PATH / "deepseek_v3_tokenizer"
