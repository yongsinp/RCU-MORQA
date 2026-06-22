from pathlib import Path

from dotenv import load_dotenv

from src.util.paths import PROJECT_ROOT


def load_env(dotenv_path: Path | None = None) -> None:
    path = dotenv_path or (PROJECT_ROOT / ".env")
    load_dotenv(dotenv_path=path, override=False)
