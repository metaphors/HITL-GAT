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
from .utils import abort_process

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component

    from .manager import Manager


class GenerateAdversarialExamplesRunner:
    def __init__(self, manager: "Manager") -> None:
        self.manager = manager
        self.generator: Optional["Popen"] = None
        self.aborted = False
        self.running = False
        self.running_data: Dict["Component", Any] = None

    def set_abort(self) -> None:
        self.aborted = True
        if self.generator is not None:
            abort_process(self.generator.pid)

    def _initialize(self, data: Dict["Component", Any]) -> str:
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]

        lang = get("top.lang")
        pretrained_language_model = get("generate_adversarial_examples.pretrained_language_model")
        dataset = get("generate_adversarial_examples.dataset")
        attack_method = get("generate_adversarial_examples.attack_method")

        if self.running:
            return ALERTS["err_conflict"][lang]

        if not pretrained_language_model:
            return ALERTS["err_no_model"][lang]

        if not dataset:
            return ALERTS["err_no_dataset"][lang]

        if not attack_method:
            return ALERTS["err_no_attack_method"][lang]

        return ""

    def _finalize(self, lang: str, finish_info: str) -> str:
        finish_info = ALERTS["info_aborted"][lang] if self.aborted else finish_info
        self.generator = None
        self.aborted = False
        self.running = False
        self.running_data = None
        return finish_info

    def _launch(self, data: Dict["Component", Any]) -> Generator[Dict["Component", Any], None, None]:
        output_box = self.manager.get_elem_by_id("generate_adversarial_examples.output_box")
        error = self._initialize(data)
        if error:
            gr.Warning(error)
            yield {output_box: error}
        else:
            self.running_data = data

            get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
            output_dir = get("generate_adversarial_examples.output_dir")
            os.makedirs(os.path.join(DEFAULT_DATA_DIR, output_dir), exist_ok=True)

            attack_method = get("generate_adversarial_examples.attack_method")
            victim_language_model = get("generate_adversarial_examples.victim_language_model")
            script_name = "attack+{}+{}.py".format(attack_method, victim_language_model)
            script_path = os.path.join(DEFAULT_SCRIPT_DIR, script_name)
            env = deepcopy(os.environ)
            self.generator = Popen(["python", script_path], env=env, stdout=PIPE, stderr=PIPE,
                                   text=True, universal_newlines=True)
            yield from self.monitor()

    def run_generation(self, data):
        yield from self._launch(data)

    def monitor(self):
        self.aborted = False
        self.running = True

        get = lambda elem_id: self.running_data[self.manager.get_elem_by_id(elem_id)]
        lang = get("top.lang")
        output_dir = get("generate_adversarial_examples.output_dir")

        output_box = self.manager.get_elem_by_id("generate_adversarial_examples.output_box")

        running_log = ""

        while self.generator is not None:
            if self.aborted:
                yield {
                    output_box: ALERTS["info_aborting"][lang]
                }
            else:
                running_log_path = os.path.join(DEFAULT_DATA_DIR, output_dir, "log.txt")
                if os.path.isfile(running_log_path):
                    with open(running_log_path, "r", encoding="utf-8") as f:
                        running_log = f.read()

                return_dict = {
                    output_box: running_log
                }
                yield return_dict

            try:
                self.generator.wait(2)
                self.generator = None
            except TimeoutExpired:
                continue

        running_log_path = os.path.join(DEFAULT_DATA_DIR, output_dir, "log.txt")
        sample_num = 0
        if output_dir.endswith("TNCC-title"):
            sample_total = 927
        elif output_dir.endswith("TNCC-document"):
            sample_total = 920
        elif output_dir.endswith("TU_SA"):
            sample_total = 1000
        if os.path.isfile(running_log_path):
            with open(running_log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Sample"):
                        sample_num += 1
        if sample_num == sample_total:
            finish_info = ALERTS["info_finished"][lang]
        else:
            finish_info = ALERTS["err_failed"][lang]

        return_dict = {
            output_box: self._finalize(lang, finish_info)
        }
        yield return_dict
