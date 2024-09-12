# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/webui/components/train.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

from typing import TYPE_CHECKING, Dict

from ...extras.constants import ATTACKED_MODEL_LIST, EVALUATION_DATASET_LIST, EVALUATION_MODEL_LIST
from ...extras.packages import is_gradio_available
from .data import create_preview_box

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_evaluate_adversarial_robustness_tab(engine: "Engine") -> Dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        model_for_generating_adversarial_examples = gr.Dropdown(choices=ATTACKED_MODEL_LIST,
                                                                value=ATTACKED_MODEL_LIST[0],
                                                                interactive=True, scale=1)
        dataset_for_evaluating_adversarial_robustness = gr.Dropdown(choices=EVALUATION_DATASET_LIST,
                                                                    value=EVALUATION_DATASET_LIST[0],
                                                                    interactive=True, scale=1)
        model_for_evaluating_adversarial_robustness = gr.Dropdown(choices=EVALUATION_MODEL_LIST,
                                                                  value=EVALUATION_MODEL_LIST[0],
                                                                  interactive=True, scale=1)

    with gr.Row():
        evaluate_btn = gr.Button(variant="primary", interactive=True, scale=1)
        stop_btn = gr.Button(variant="stop", interactive=True, scale=1)

    with gr.Row():
        dataset = gr.Dropdown(choices=["Adv.TNCC-title", "Adv.TU_SA"], visible=True, interactive=True, scale=1)
        preview_elems = create_preview_box(dataset)

    with gr.Row():
        evaluation_results = gr.DataFrame(
            headers=["Model",
                     "Adversarial Robustness on Adv.TNCC-title",
                     "Adversarial Robustness on Adv.TU_SA",
                     "Average Adversarial Robustness"],
            datatype=["str", "number", "number", "number"], type="pandas",
            column_widths=["16%", "28%", "28%", "28%"], wrap=True, visible=True, interactive=False, scale=1)

    with gr.Row():
        output_box = gr.Markdown()

    input_elems.update({
        model_for_generating_adversarial_examples,
        dataset_for_evaluating_adversarial_robustness,
        model_for_evaluating_adversarial_robustness
    })
    elem_dict.update(
        dict(
            model_for_generating_adversarial_examples=model_for_generating_adversarial_examples,
            dataset_for_evaluating_adversarial_robustness=dataset_for_evaluating_adversarial_robustness,
            model_for_evaluating_adversarial_robustness=model_for_evaluating_adversarial_robustness,
            evaluate_btn=evaluate_btn,
            stop_btn=stop_btn,
            dataset=dataset,
            **preview_elems,
            evaluation_results=evaluation_results,
            output_box=output_box
        )
    )
    output_elems = [
        evaluation_results,
        output_box
    ]

    evaluate_btn.click(engine.evaluate_adversarial_robustness_runner.run_evaluation, input_elems, output_elems)
    stop_btn.click(engine.evaluate_adversarial_robustness_runner.set_abort)

    return elem_dict
