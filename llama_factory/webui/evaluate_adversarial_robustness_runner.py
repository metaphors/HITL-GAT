# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/webui/runner.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

import os
import pandas as pd
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


class EvaluateAdversarialRobustnessRunner:
    def __init__(self, manager: "Manager") -> None:
        self.manager = manager
        self.evaluator: Optional["Popen"] = None
        self.aborted = False
        self.running = False
        self.running_data: Dict["Component", Any] = None

    def set_abort(self) -> None:
        self.aborted = True
        if self.evaluator is not None:
            abort_process(self.evaluator.pid)

    def _initialize(self, data: Dict["Component", Any]) -> str:
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]

        lang = get("top.lang")

        if self.running:
            return ALERTS["err_conflict"][lang]

        return ""

    def _finalize(self, lang: str, finish_info: str) -> str:
        finish_info = ALERTS["info_aborted"][lang] if self.aborted else finish_info
        self.evaluator = None
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

            script_path = os.path.join(DEFAULT_SCRIPT_DIR, "evaluate+all.py")
            env = deepcopy(os.environ)
            self.evaluator = Popen(["python", script_path], env=env, stdout=PIPE, stderr=PIPE,
                                   text=True, universal_newlines=True)
            yield from self.monitor()

    def run_evaluation(self, data):
        yield from self._launch(data)

    def monitor(self):
        self.aborted = False
        self.running = True

        get = lambda elem_id: self.running_data[self.manager.get_elem_by_id(elem_id)]
        lang = get("top.lang")

        evaluation_results = self.manager.get_elem_by_id("evaluate_adversarial_robustness.evaluation_results")
        output_box = self.manager.get_elem_by_id("evaluate_adversarial_robustness.output_box")

        while self.evaluator is not None:
            if self.aborted:
                yield {
                    evaluation_results: gr.DataFrame(
                        headers=["Model",
                                 "Adversarial Robustness on Adv.TNCC-title",
                                 "Adversarial Robustness on Adv.TU_SA",
                                 "Average Adversarial Robustness"],
                        datatype=["str", "number", "number", "number"], type="pandas",
                        column_widths=["16%", "28%", "28%", "28%"], wrap=True, visible=True, interactive=False,
                        scale=1),
                    output_box: ALERTS["info_aborting"][lang]
                }
            else:
                return_dict = {
                    evaluation_results: gr.DataFrame(
                        headers=["Model",
                                 "Adversarial Robustness on Adv.TNCC-title",
                                 "Adversarial Robustness on Adv.TU_SA",
                                 "Average Adversarial Robustness"],
                        datatype=["str", "number", "number", "number"], type="pandas",
                        column_widths=["16%", "28%", "28%", "28%"], wrap=True, visible=True, interactive=False,
                        scale=1),
                    output_box: ALERTS["info_evaluating"][lang]
                }
                yield return_dict

            try:
                self.evaluator.wait(2)
                self.evaluator = None
            except TimeoutExpired:
                continue

        results_file = os.path.join(DEFAULT_DATA_DIR, "Dataset.AdvTS", "results.txt")

        if os.path.exists(results_file):
            finish_info = ALERTS["info_evaluated"][lang]

            results = pd.read_csv(results_file, sep="\t", lineterminator="\n", header=None,
                                  names=["model+dataset", "right_num", "total_num"]).to_dict(orient="records")
            new_results = {"Model": ["CINO-small-v2", "CINO-base-v2", "CINO-large-v2"],
                           "Adversarial Robustness on Adv.TNCC-title": [0, 0, 0],
                           "Adversarial Robustness on Adv.TU_SA": [0, 0, 0],
                           "Average Adversarial Robustness": [0, 0, 0]}
            for result in results:
                model = result["model+dataset"].split("+")[0]
                dataset = result["model+dataset"].split("+")[1]
                if model == "CINO-small-v2" and dataset == "TNCC-title":
                    new_results["Adversarial Robustness on Adv.TNCC-title"][0] = round(
                        result["right_num"] / result["total_num"], 4)
                elif model == "CINO-base-v2" and dataset == "TNCC-title":
                    new_results["Adversarial Robustness on Adv.TNCC-title"][1] = round(
                        result["right_num"] / result["total_num"], 4)
                elif model == "CINO-large-v2" and dataset == "TNCC-title":
                    new_results["Adversarial Robustness on Adv.TNCC-title"][2] = round(
                        result["right_num"] / result["total_num"], 4)
                elif model == "CINO-small-v2" and dataset == "TU_SA":
                    new_results["Adversarial Robustness on Adv.TU_SA"][0] = round(
                        result["right_num"] / result["total_num"], 4)
                elif model == "CINO-base-v2" and dataset == "TU_SA":
                    new_results["Adversarial Robustness on Adv.TU_SA"][1] = round(
                        result["right_num"] / result["total_num"], 4)
                elif model == "CINO-large-v2" and dataset == "TU_SA":
                    new_results["Adversarial Robustness on Adv.TU_SA"][2] = round(
                        result["right_num"] / result["total_num"], 4)
            for idx, _ in enumerate(new_results["Model"]):
                new_results["Average Adversarial Robustness"][idx] = round(
                    (new_results["Adversarial Robustness on Adv.TNCC-title"][idx] +
                     new_results["Adversarial Robustness on Adv.TU_SA"][idx]) / 2, 4)
            new_results_df = pd.DataFrame.from_dict(new_results)

            return_dict = {
                evaluation_results: new_results_df,
                output_box: self._finalize(lang, finish_info)
            }
        else:
            finish_info = ALERTS["err_evaluator"][lang]

            return_dict = {
                evaluation_results: gr.DataFrame(
                    headers=["Model",
                             "Adversarial Robustness on Adv.TNCC-title",
                             "Adversarial Robustness on Adv.TU_SA",
                             "Average Adversarial Robustness"],
                    datatype=["str", "number", "number", "number"], type="pandas",
                    column_widths=["16%", "28%", "28%", "28%"], wrap=True, visible=True, interactive=False,
                    scale=1),
                output_box: self._finalize(lang, finish_info)
            }

        yield return_dict
