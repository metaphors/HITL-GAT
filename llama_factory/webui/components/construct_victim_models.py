# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/webui/components/train.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

from typing import TYPE_CHECKING, Dict

from ...extras.constants import (PRETRAINED_LANGUAGE_MODEL_LIST, TRAIN_DATASET_LIST,
                                 VICTIM_OUTPUT_DIR_LIST, METRIC_FOR_BEST_MODEL_LIST)
from ...extras.packages import is_gradio_available
from ..utils import change_pretrained_language_model, change_train_dataset
from .data import create_preview_box

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_construct_victim_models_tab(engine: "Engine") -> Dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        pretrained_language_model = gr.Dropdown(choices=PRETRAINED_LANGUAGE_MODEL_LIST, interactive=True, scale=2)
        dataset = gr.Dropdown(choices=TRAIN_DATASET_LIST, interactive=True, scale=2)
        preview_elems = create_preview_box(dataset)

    with gr.Row():
        output_dir = gr.Dropdown(choices=VICTIM_OUTPUT_DIR_LIST, value=VICTIM_OUTPUT_DIR_LIST[0], interactive=False,
                                 scale=2)
        metric_for_best_model = gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                            interactive=False, scale=1)
        batch_size = gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1)
        num_train_epochs = gr.Textbox(value="20", interactive=False, scale=1)
        learning_rate = gr.Textbox(value="5e-5", interactive=False, scale=1)
        warmup_ratio = gr.Textbox(value="0.0", interactive=False, scale=1)

    with gr.Row():
        start_btn = gr.Button(variant="primary", interactive=True, scale=1)
        stop_btn = gr.Button(variant="stop", interactive=True, scale=1)

    with gr.Row():
        f1_viewer = gr.Plot(scale=1)
        accuracy_viewer = gr.Plot(scale=1)
        loss_viewer = gr.Plot(scale=1)

    with gr.Row():
        progress_bar = gr.Slider(label="Running...", minimum=0, maximum=100, step=1, value=0,
                                 visible=False, interactive=False, scale=1)

    with gr.Row():
        output_box = gr.Markdown()

    input_elems.update({
        pretrained_language_model,
        dataset,
        output_dir,
        metric_for_best_model,
        batch_size,
        num_train_epochs,
        learning_rate,
        warmup_ratio
    })
    elem_dict.update(
        dict(
            pretrained_language_model=pretrained_language_model,
            dataset=dataset,
            **preview_elems,
            output_dir=output_dir,
            metric_for_best_model=metric_for_best_model,
            batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            start_btn=start_btn,
            stop_btn=stop_btn,
            f1_viewer=f1_viewer,
            accuracy_viewer=accuracy_viewer,
            loss_viewer=loss_viewer,
            progress_bar=progress_bar,
            output_box=output_box
        )
    )
    output_elems = [
        progress_bar,
        output_box,
        f1_viewer,
        accuracy_viewer,
        loss_viewer
    ]

    pretrained_language_model.change(change_pretrained_language_model, [pretrained_language_model, dataset],
                                     [output_dir,
                                      metric_for_best_model, batch_size, num_train_epochs, learning_rate, warmup_ratio])
    dataset.change(change_train_dataset, [pretrained_language_model, dataset],
                   [output_dir, metric_for_best_model, batch_size, num_train_epochs, learning_rate, warmup_ratio])

    start_btn.click(engine.construct_victim_models_runner.run_train, input_elems, output_elems)
    stop_btn.click(engine.construct_victim_models_runner.set_abort)

    return elem_dict
