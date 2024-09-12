# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/webui/common.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

import os
from typing import Any, Dict

from yaml import safe_dump, safe_load

DEFAULT_DATA_DIR = "data"
DEFAULT_SCRIPT_DIR = "script"
DEFAULT_CACHE_DIR = "cache"
USER_CONFIG = "user_config.yaml"


def get_config_path() -> os.PathLike:
    r"""
    Gets the path to user config.
    """
    return os.path.join(DEFAULT_CACHE_DIR, USER_CONFIG)


def load_config() -> Dict[str, Any]:
    r"""
    Loads user config if exists.
    """
    try:
        with open(get_config_path(), "r", encoding="utf-8") as f:
            return safe_load(f)
    except Exception:
        return {"lang": None}


def save_config(lang: str) -> None:
    r"""
    Saves user config.
    """
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    user_config = load_config()
    user_config["lang"] = lang or user_config["lang"]

    with open(get_config_path(), "w", encoding="utf-8") as f:
        safe_dump(user_config, f)
