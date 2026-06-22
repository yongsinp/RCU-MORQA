from src.util.paths import (
    DATA_RCU_EN_PATH,
    EXTERNAL_PATH,
    OUT_PATH,
    PROJECT_ROOT,
)


def test_project_paths_are_root_relative() -> None:
    assert DATA_RCU_EN_PATH == PROJECT_ROOT / "data" / "rcu-en"
    assert OUT_PATH == PROJECT_ROOT / "out"
    assert EXTERNAL_PATH == PROJECT_ROOT / "external"
