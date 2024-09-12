# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/extras/packages.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

import importlib.util


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def is_gradio_available():
    return _is_package_available("gradio")


def is_matplotlib_available():
    return _is_package_available("matplotlib")
