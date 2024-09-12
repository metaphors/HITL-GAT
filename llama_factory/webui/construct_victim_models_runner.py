# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/webui/runner.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

import os
import json
from copy import deepcopy
from subprocess import Popen, PIPE, TimeoutExpired
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional

from transformers.trainer import TRAINING_ARGS_NAME

from ..extras.misc import torch_gc
from ..extras.packages import is_gradio_available
from ..extras.plotting import generate_epoch_value_plot
from .common import DEFAULT_DATA_DIR, DEFAULT_SCRIPT_DIR
from .locales import ALERTS
from .utils import abort_process

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component

    from .manager import Manager


class ConstructVictimModelsRunner:
    def __init__(self, manager: "Manager") -> None:
        self.manager = manager
        self.trainer: Optional["Popen"] = None
        self.aborted = False
        self.running = False
        self.running_data: Dict["Component", Any] = None

    def set_abort(self) -> None:
        self.aborted = True
        if self.trainer is not None:
            abort_process(self.trainer.pid)

    def _initialize(self, data: Dict["Component", Any]) -> str:
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]

        lang = get("top.lang")
        pretrained_language_model = get("construct_victim_models.pretrained_language_model")
        dataset = get("construct_victim_models.dataset")

        if self.running:
            return ALERTS["err_conflict"][lang]

        if not pretrained_language_model:
            return ALERTS["err_no_model"][lang]

        if not dataset:
            return ALERTS["err_no_dataset"][lang]

        return ""

    def _finalize(self, lang: str, finish_info: str) -> str:
        finish_info = ALERTS["info_aborted"][lang] if self.aborted else finish_info
        self.trainer = None
        self.aborted = False
        self.running = False
        self.running_data = None
        torch_gc()
        return finish_info

    def _launch(self, data: Dict["Component", Any]) -> Generator[Dict["Component", Any], None, None]:
        output_box = self.manager.get_elem_by_id("construct_victim_models.output_box")
        error = self._initialize(data)
        if error:
            gr.Warning(error)
            yield {output_box: error}
        else:
            self.running_data = data

            get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
            output_dir = get("construct_victim_models.output_dir")
            os.makedirs(os.path.join(DEFAULT_DATA_DIR, output_dir), exist_ok=True)

            pretrained_language_model = get("construct_victim_models.pretrained_language_model")
            dataset = get("construct_victim_models.dataset")
            script_name = "fine-tune+{}+{}.py".format(pretrained_language_model, dataset.split(".")[0])
            script_path = os.path.join(DEFAULT_SCRIPT_DIR, script_name)
            env = deepcopy(os.environ)
            self.trainer = Popen(["python", script_path], env=env, stdout=PIPE, stderr=PIPE,
                                 text=True, universal_newlines=True)
            yield from self.monitor()

    def run_train(self, data):
        yield from self._launch(data)

    def monitor(self):
        self.aborted = False
        self.running = True

        get = lambda elem_id: self.running_data[self.manager.get_elem_by_id(elem_id)]
        lang = get("top.lang")
        output_dir = get("construct_victim_models.output_dir")
        num_train_epochs = get("construct_victim_models.num_train_epochs")

        progress_bar = self.manager.get_elem_by_id("construct_victim_models.progress_bar")
        output_box = self.manager.get_elem_by_id("construct_victim_models.output_box")
        f1_viewer = self.manager.get_elem_by_id("construct_victim_models.f1_viewer")
        accuracy_viewer = self.manager.get_elem_by_id("construct_victim_models.accuracy_viewer")
        loss_viewer = self.manager.get_elem_by_id("construct_victim_models.loss_viewer")

        running_progress = gr.Slider(label="Running...", minimum=0, maximum=100, step=1, value=0,
                                     visible=True, interactive=False, scale=1)
        running_f1 = None
        running_accuracy = None
        running_loss = None

        epoch_list = []
        f1_list = []
        accuracy_list = []
        loss_list = []

        while self.trainer is not None:
            if self.aborted:
                yield {
                    progress_bar: gr.Slider(label="Running...", minimum=0, maximum=100, step=1, value=0,
                                            visible=False, interactive=False, scale=1),
                    output_box: ALERTS["info_aborting"][lang]
                }
            else:
                running_log = self.trainer.stdout.readline().strip()
                print(running_log)
                if running_log.startswith("{'eval_loss':"):
                    eval_dict = json.loads(running_log.replace("'", '"'))
                    running_progress = gr.Slider(label="Running...", minimum=0, maximum=100, step=1,
                                                 value=round((eval_dict["epoch"] / float(num_train_epochs)) * 100, 2),
                                                 visible=True, interactive=False, scale=1)
                    epoch_list.append(eval_dict["epoch"])
                    if "eval_f1" in eval_dict:
                        f1_list.append(eval_dict["eval_f1"])
                    elif "eval_macro-f1" in eval_dict:
                        f1_list.append(eval_dict["eval_macro-f1"])
                    accuracy_list.append(eval_dict["eval_accuracy"])
                    loss_list.append(eval_dict["eval_loss"])
                    running_f1 = gr.Plot(generate_epoch_value_plot(epoch_list, f1_list))
                    running_accuracy = gr.Plot(generate_epoch_value_plot(epoch_list, accuracy_list))
                    running_loss = gr.Plot(generate_epoch_value_plot(epoch_list, loss_list))
                return_dict = {
                    progress_bar: running_progress,
                    output_box: running_log
                }
                if running_f1 is not None:
                    return_dict[f1_viewer] = running_f1
                if running_accuracy is not None:
                    return_dict[accuracy_viewer] = running_accuracy
                if running_loss is not None:
                    return_dict[loss_viewer] = running_loss

                yield return_dict

            try:
                self.trainer.wait(2)
                self.trainer = None
            except TimeoutExpired:
                continue

        if os.path.exists(os.path.join(DEFAULT_DATA_DIR, output_dir, TRAINING_ARGS_NAME)):
            finish_info = ALERTS["info_finished"][lang]
        else:
            finish_info = ALERTS["err_failed"][lang]

        return_dict = {
            progress_bar: gr.Slider(label="Running...", minimum=0, maximum=100, step=1, value=0,
                                    visible=False, interactive=False, scale=1),
            output_box: self._finalize(lang, finish_info),
            f1_viewer: gr.Plot(scale=1),
            accuracy_viewer: gr.Plot(scale=1),
            loss_viewer: gr.Plot(scale=1)
        }
        yield return_dict
