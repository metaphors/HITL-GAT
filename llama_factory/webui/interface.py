# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/webui/interface.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

from ..extras.packages import is_gradio_available
from .common import save_config
from .components import (
    create_top,
    create_construct_victim_models_tab,
    create_generate_adversarial_examples_tab,
    create_conduct_human_annotation_tab,
    create_evaluate_adversarial_robustness_tab
)
from .css import CSS
from .engine import Engine

if is_gradio_available():
    import gradio as gr


def create_ui() -> "gr.Blocks":
    engine = Engine()

    with gr.Blocks(title="TSAttack", css=CSS) as demo:
        engine.manager.add_elems("top", create_top())
        lang: "gr.Dropdown" = engine.manager.get_elem_by_id("top.lang")

        with gr.Tab("1. Construct Victim Models"):
            engine.manager.add_elems(
                "construct_victim_models",
                create_construct_victim_models_tab(engine)
            )

        with gr.Tab("2. Generate Adversarial Examples"):
            engine.manager.add_elems(
                "generate_adversarial_examples",
                create_generate_adversarial_examples_tab(engine)
            )

        with gr.Tab("3. Conduct Human Annotation"):
            engine.manager.add_elems(
                "conduct_human_annotation",
                create_conduct_human_annotation_tab(engine)
            )

        with gr.Tab("4. Evaluate Adversarial Robustness"):
            engine.manager.add_elems(
                "evaluate_adversarial_robustness",
                create_evaluate_adversarial_robustness_tab(engine)
            )

        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo
