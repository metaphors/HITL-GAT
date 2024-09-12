# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/webui/components/train.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

from typing import TYPE_CHECKING, Dict

from ...extras.constants import PRETRAINED_LANGUAGE_MODEL_LIST, TEST_DATASET_LIST, ATTACK_METHOD_LIST, \
    ADV_OUTPUT_DIR_LIST
from ...extras.packages import is_gradio_available
from ..utils import change_victim_language_model, change_test_dataset, change_attack_method
from .data import create_preview_box

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_generate_adversarial_examples_tab(engine: "Engine") -> Dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        pretrained_language_model = gr.Dropdown(choices=PRETRAINED_LANGUAGE_MODEL_LIST, interactive=True, scale=2)
        dataset = gr.Dropdown(choices=TEST_DATASET_LIST, interactive=True, scale=2)
        preview_elems = create_preview_box(dataset)

    with gr.Row():
        victim_language_model = gr.Dropdown(interactive=False, scale=1)
        attack_method = gr.Dropdown(choices=ATTACK_METHOD_LIST, interactive=True, scale=1)
        output_dir = gr.Dropdown(choices=ADV_OUTPUT_DIR_LIST, value=ADV_OUTPUT_DIR_LIST[0], interactive=False, scale=1)

    with gr.Row():
        start_btn = gr.Button(variant="primary", interactive=True, scale=1)
        stop_btn = gr.Button(variant="stop", interactive=True, scale=1)

    with gr.Row():
        output_box = gr.Markdown()

    input_elems.update({
        pretrained_language_model,
        dataset,
        victim_language_model,
        attack_method,
        output_dir
    })
    elem_dict.update(
        dict(
            pretrained_language_model=pretrained_language_model,
            dataset=dataset,
            **preview_elems,
            victim_language_model=victim_language_model,
            attack_method=attack_method,
            output_dir=output_dir,
            start_btn=start_btn,
            stop_btn=stop_btn,
            output_box=output_box
        )
    )
    output_elems = [
        output_box
    ]

    pretrained_language_model.change(change_victim_language_model,
                                     [pretrained_language_model, dataset, attack_method],
                                     [victim_language_model, output_dir])
    dataset.change(change_test_dataset, [pretrained_language_model, dataset, attack_method],
                   [victim_language_model, output_dir])
    attack_method.change(change_attack_method, [victim_language_model, attack_method], output_dir)

    start_btn.click(engine.generate_adversarial_examples_runner.run_generation, input_elems, output_elems)
    stop_btn.click(engine.generate_adversarial_examples_runner.set_abort)

    return elem_dict
