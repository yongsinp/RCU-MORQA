from src.util.paths import (
    DATA_RCU_EN_PATH,
    DEEPSEEK_TOKENIZER_PATH,
    OUT_PATH,
    PROJECT_ROOT,
    QWEN_TOKENIZER_PATH,
)


def test_project_paths_are_root_relative() -> None:
    assert DATA_RCU_EN_PATH == PROJECT_ROOT / "data" / "rcu-en"
    assert OUT_PATH == PROJECT_ROOT / "out"
    assert QWEN_TOKENIZER_PATH == PROJECT_ROOT / "external" / "qwen_tokenizer"
    assert DEEPSEEK_TOKENIZER_PATH == PROJECT_ROOT / "external" / "deepseek_v3_tokenizer"
