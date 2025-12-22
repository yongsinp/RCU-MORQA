import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def read_json(file_path: str) -> Any:
    logger.debug(f"Reading JSON file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as r:
        return json.load(r)


def write_json(file_path, json_data):
    logger.debug(f"Writing JSON file: {file_path}")

    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as w:
        json.dump(json_data, w, indent=4, ensure_ascii=False)
