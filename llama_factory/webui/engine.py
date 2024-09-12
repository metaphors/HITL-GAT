# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/webui/engine.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

from typing import TYPE_CHECKING, Any, Dict

from .common import load_config
from .locales import LOCALES
from .manager import Manager
from .construct_victim_models_runner import ConstructVictimModelsRunner
from .generate_adversarial_examples_runner import GenerateAdversarialExamplesRunner
from .conduct_human_annotation_runner import ConductHumanAnnotationRunner
from .evaluate_adversarial_robustness_runner import EvaluateAdversarialRobustnessRunner

if TYPE_CHECKING:
    from gradio.components import Component


class Engine:
    def __init__(self) -> None:
        self.manager = Manager()
        self.construct_victim_models_runner = ConstructVictimModelsRunner(self.manager)
        self.generate_adversarial_examples_runner = GenerateAdversarialExamplesRunner(self.manager)
        self.conduct_human_annotation_runner = ConductHumanAnnotationRunner(self.manager)
        self.evaluate_adversarial_robustness_runner = EvaluateAdversarialRobustnessRunner(self.manager)

    def _update_component(self, input_dict: Dict[str, Dict[str, Any]]) -> Dict["Component", "Component"]:
        r"""
        Gets the dict to update the components.
        """
        output_dict: Dict["Component", "Component"] = {}
        for elem_id, elem_attr in input_dict.items():
            elem = self.manager.get_elem_by_id(elem_id)
            output_dict[elem] = elem.__class__(**elem_attr)

        return output_dict

    def resume(self):
        user_config = load_config()
        lang = user_config.get("lang", None) or "en"

        init_dict = {"top.lang": {"value": lang}}
        yield self._update_component(init_dict)

        if self.construct_victim_models_runner.running:
            yield {elem: elem.__class__(value=value) for elem, value in
                   self.construct_victim_models_runner.running_data.items()}
        if self.generate_adversarial_examples_runner.running:
            yield {elem: elem.__class__(value=value) for elem, value in
                   self.generate_adversarial_examples_runner.running_data.items()}
        if self.conduct_human_annotation_runner.running:
            yield {elem: elem.__class__(value=value) for elem, value in
                   self.conduct_human_annotation_runner.running_data.items()}
        if self.evaluate_adversarial_robustness_runner.running:
            yield {elem: elem.__class__(value=value) for elem, value in
                   self.evaluate_adversarial_robustness_runner.running_data.items()}

    def change_lang(self, lang: str):
        return {
            elem: elem.__class__(**LOCALES[elem_name][lang])
            for elem_name, elem in self.manager.get_elem_iter()
            if elem_name in LOCALES
        }
