import os

from src.util.env import load_env


def test_load_env_reads_dotenv_file(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_MORQA_ENV=loaded\n", encoding="utf-8")

    monkeypatch.delenv("TEST_MORQA_ENV", raising=False)
    load_env(env_file)

    assert os.getenv("TEST_MORQA_ENV") == "loaded"
