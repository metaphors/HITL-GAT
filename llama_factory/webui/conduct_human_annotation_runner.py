# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/webui/runner.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

import os
from copy import deepcopy
from subprocess import Popen, PIPE, TimeoutExpired
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional

from ..extras.packages import is_gradio_available
from .common import DEFAULT_DATA_DIR, DEFAULT_SCRIPT_DIR
from .locales import ALERTS

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component

    from .manager import Manager


class ConductHumanAnnotationRunner:
    def __init__(self, manager: "Manager") -> None:
        self.manager = manager
        self.filter: Optional["Popen"] = None
        self.aborted = False
        self.running = False
        self.running_data: Dict["Component", Any] = None

    def _finalize(self, finish_info: str) -> str:
        finish_info = finish_info
        self.filter = None
        self.aborted = False
        self.running = False
        self.running_data = None
        return finish_info

    def _launch(self, data: Dict["Component", Any]) -> Generator[Dict["Component", Any], None, None]:
        self.running_data = data

        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        input_dir = get("conduct_human_annotation.input_dir")

        script_name = "filter+{}+{}.py".format(input_dir.split(".")[1], input_dir.split(".")[2])
        script_path = os.path.join(DEFAULT_SCRIPT_DIR, script_name)
        env = deepcopy(os.environ)
        self.filter = Popen(["python", script_path], env=env, stdout=PIPE, stderr=PIPE,
                            text=True, universal_newlines=True)
        yield from self.monitor()

    def run_filter(self, data):
        yield from self._launch(data)

    def monitor(self):
        self.aborted = False
        self.running = True

        get = lambda elem_id: self.running_data[self.manager.get_elem_by_id(elem_id)]
        lang = get("top.lang")
        input_dir = get("conduct_human_annotation.input_dir")

        filter_btn = self.manager.get_elem_by_id("conduct_human_annotation.filter_btn")
        annotate_btn = self.manager.get_elem_by_id("conduct_human_annotation.annotate_btn")
        output_box = self.manager.get_elem_by_id("conduct_human_annotation.output_box")

        while self.filter is not None:
            return_dict = {
                filter_btn: gr.Button(variant="secondary", interactive=False, scale=1),
                annotate_btn: gr.Button(variant="primary", interactive=False, scale=1),
                output_box: ALERTS["info_filtering"][lang]
            }
            yield return_dict

            try:
                self.filter.wait(2)
                self.filter = None
            except TimeoutExpired:
                continue

        if os.path.exists(os.path.join(DEFAULT_DATA_DIR, input_dir, "data_filtered.txt")):
            finish_info = ALERTS["info_filtered"][lang]
        else:
            finish_info = ALERTS["err_filter"][lang]

        return_dict = {
            filter_btn: gr.Button(variant="secondary", interactive=False, scale=1),
            annotate_btn: gr.Button(variant="primary", interactive=True, scale=1),
            output_box: self._finalize(finish_info)
        }
        yield return_dict
