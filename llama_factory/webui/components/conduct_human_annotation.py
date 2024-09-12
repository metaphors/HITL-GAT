# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/webui/components/train.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

from typing import TYPE_CHECKING, Dict

from ...extras.constants import ADV_INPUT_DIR_LIST
from ...extras.packages import is_gradio_available
from ..utils import change_input_dir, start_annotation, start_submission

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_conduct_human_annotation_tab(engine: "Engine") -> Dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        input_dir = gr.Dropdown(choices=ADV_INPUT_DIR_LIST, interactive=True, scale=1)
        filter_btn = gr.Button(variant="secondary", interactive=False, scale=1)

    with gr.Row():
        annotate_btn = gr.Button(variant="primary", interactive=False, scale=1)

    with gr.Row():
        data_frame = gr.Dataframe(
            headers=["label", "orig", "adv", "score"], datatype=["str", "str", "str", "number"], type="pandas",
            column_widths=["10%", "40%", "40%", "10%"], wrap=True, visible=False, interactive=False, scale=1
        )

    with gr.Row():
        submit_btn = gr.Button(variant="primary", visible=False, interactive=False, scale=1)

    with gr.Row():
        output_box = gr.Markdown()

    input_elems.update({
        input_dir
    })
    elem_dict.update(
        dict(
            input_dir=input_dir,
            filter_btn=filter_btn,
            annotate_btn=annotate_btn,
            data_frame=data_frame,
            submit_btn=submit_btn,
            output_box=output_box
        )
    )
    output_elems = [
        filter_btn,
        annotate_btn,
        output_box
    ]

    lang = engine.manager.get_elem_by_id("top.lang")

    input_dir.change(change_input_dir, lang, [filter_btn, annotate_btn, data_frame, submit_btn, output_box])

    filter_btn.click(engine.conduct_human_annotation_runner.run_filter, input_elems, output_elems)

    annotate_btn.click(start_annotation, [lang, input_dir],
                       [annotate_btn, data_frame, submit_btn, output_box])

    submit_btn.click(start_submission, [lang, input_dir, data_frame], [submit_btn, output_box])

    return elem_dict
