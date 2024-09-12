# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/webui/components/top.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

from typing import TYPE_CHECKING, Dict

from ...extras.packages import is_gradio_available

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component


def create_top() -> Dict[str, "Component"]:
    with gr.Row():
        lang = gr.Dropdown(choices=["en", "zh"], scale=1)

    return dict(
        lang=lang
    )
