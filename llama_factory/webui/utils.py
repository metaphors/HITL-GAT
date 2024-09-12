# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/webui/utils.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

import os
import signal

import psutil
import pandas as pd

from ..extras.constants import VICTIM_OUTPUT_DIR_LIST, ADV_OUTPUT_DIR_LIST, METRIC_FOR_BEST_MODEL_LIST
from ..extras.packages import is_gradio_available
from .common import DEFAULT_DATA_DIR
from .locales import ALERTS

if is_gradio_available():
    import gradio as gr


def abort_process(pid: int) -> None:
    r"""
    Aborts the processes recursively in a bottom-up way.
    """
    try:
        children = psutil.Process(pid).children()
        if children:
            for child in children:
                abort_process(child.pid)
        os.kill(pid, signal.SIGABRT)
    except Exception:
        pass


def change_pretrained_language_model(pretrained_language_model: str, dataset: str) -> ["gr.Dropdown", "gr.Dropdown",
                                                                                       "gr.Slider", "gr.Textbox",
                                                                                       "gr.Textbox", "gr.Textbox"]:
    if dataset:
        if pretrained_language_model == "Tibetan-BERT" and dataset == "TU_SA.train":
            return [gr.Dropdown(choices=["Victim.BERT.Tibetan-BERT+TU_SA"], value="Victim.BERT.Tibetan-BERT+TU_SA",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="20", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.0", interactive=False, scale=1)]
        elif pretrained_language_model == "Tibetan-BERT" and dataset == "TNCC-title.train":
            return [gr.Dropdown(choices=["Victim.BERT.Tibetan-BERT+TNCC-title"],
                                value="Victim.BERT.Tibetan-BERT+TNCC-title",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="20", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.0", interactive=False, scale=1)]
        elif pretrained_language_model == "Tibetan-BERT" and dataset == "TNCC-document.train":
            return [gr.Dropdown(choices=["Victim.BERT.Tibetan-BERT+TNCC-document"],
                                value="Victim.BERT.Tibetan-BERT+TNCC-document",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="20", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.0", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-small-v2" and dataset == "TU_SA.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-small-v2+TU_SA"],
                                value="Victim.XLM-RoBERTa.CINO-small-v2+TU_SA",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-small-v2" and dataset == "TNCC-title.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-small-v2+TNCC-title"],
                                value="Victim.XLM-RoBERTa.CINO-small-v2+TNCC-title",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-small-v2" and dataset == "TNCC-document.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-small-v2+TNCC-document"],
                                value="Victim.XLM-RoBERTa.CINO-small-v2+TNCC-document",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-base-v2" and dataset == "TU_SA.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-base-v2+TU_SA"],
                                value="Victim.XLM-RoBERTa.CINO-base-v2+TU_SA",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-base-v2" and dataset == "TNCC-title.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-base-v2+TNCC-title"],
                                value="Victim.XLM-RoBERTa.CINO-base-v2+TNCC-title",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-base-v2" and dataset == "TNCC-document.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-base-v2+TNCC-document"],
                                value="Victim.XLM-RoBERTa.CINO-base-v2+TNCC-document",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-large-v2" and dataset == "TU_SA.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-large-v2+TU_SA"],
                                value="Victim.XLM-RoBERTa.CINO-large-v2+TU_SA",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="3e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-large-v2" and dataset == "TNCC-title.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-large-v2+TNCC-title"],
                                value="Victim.XLM-RoBERTa.CINO-large-v2+TNCC-title",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="3e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-large-v2" and dataset == "TNCC-document.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-large-v2+TNCC-document"],
                                value="Victim.XLM-RoBERTa.CINO-large-v2+TNCC-document",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="3e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
    else:
        return [
            gr.Dropdown(choices=VICTIM_OUTPUT_DIR_LIST, value=VICTIM_OUTPUT_DIR_LIST[0], interactive=False, scale=2),
            gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                        interactive=False, scale=1),
            gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
            gr.Textbox(value="20", interactive=False, scale=1),
            gr.Textbox(value="5e-5", interactive=False, scale=1),
            gr.Textbox(value="0.0", interactive=False, scale=1)
        ]


def change_train_dataset(pretrained_language_model: str, dataset: str) -> ["gr.Dropdown", "gr.Dropdown", "gr.Slider",
                                                                           "gr.Textbox", "gr.Textbox", "gr.Textbox"]:
    if pretrained_language_model:
        if pretrained_language_model == "Tibetan-BERT" and dataset == "TU_SA.train":
            return [gr.Dropdown(choices=["Victim.BERT.Tibetan-BERT+TU_SA"], value="Victim.BERT.Tibetan-BERT+TU_SA",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="20", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.0", interactive=False, scale=1)]
        elif pretrained_language_model == "Tibetan-BERT" and dataset == "TNCC-title.train":
            return [gr.Dropdown(choices=["Victim.BERT.Tibetan-BERT+TNCC-title"],
                                value="Victim.BERT.Tibetan-BERT+TNCC-title",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="20", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.0", interactive=False, scale=1)]
        elif pretrained_language_model == "Tibetan-BERT" and dataset == "TNCC-document.train":
            return [gr.Dropdown(choices=["Victim.BERT.Tibetan-BERT+TNCC-document"],
                                value="Victim.BERT.Tibetan-BERT+TNCC-document",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="20", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.0", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-small-v2" and dataset == "TU_SA.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-small-v2+TU_SA"],
                                value="Victim.XLM-RoBERTa.CINO-small-v2+TU_SA",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-small-v2" and dataset == "TNCC-title.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-small-v2+TNCC-title"],
                                value="Victim.XLM-RoBERTa.CINO-small-v2+TNCC-title",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-small-v2" and dataset == "TNCC-document.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-small-v2+TNCC-document"],
                                value="Victim.XLM-RoBERTa.CINO-small-v2+TNCC-document",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-base-v2" and dataset == "TU_SA.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-base-v2+TU_SA"],
                                value="Victim.XLM-RoBERTa.CINO-base-v2+TU_SA",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-base-v2" and dataset == "TNCC-title.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-base-v2+TNCC-title"],
                                value="Victim.XLM-RoBERTa.CINO-base-v2+TNCC-title",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-base-v2" and dataset == "TNCC-document.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-base-v2+TNCC-document"],
                                value="Victim.XLM-RoBERTa.CINO-base-v2+TNCC-document",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="5e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-large-v2" and dataset == "TU_SA.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-large-v2+TU_SA"],
                                value="Victim.XLM-RoBERTa.CINO-large-v2+TU_SA",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="3e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-large-v2" and dataset == "TNCC-title.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-large-v2+TNCC-title"],
                                value="Victim.XLM-RoBERTa.CINO-large-v2+TNCC-title",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="3e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
        elif pretrained_language_model == "CINO-large-v2" and dataset == "TNCC-document.train":
            return [gr.Dropdown(choices=["Victim.XLM-RoBERTa.CINO-large-v2+TNCC-document"],
                                value="Victim.XLM-RoBERTa.CINO-large-v2+TNCC-document",
                                interactive=False, scale=2),
                    gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                                interactive=False, scale=1),
                    gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
                    gr.Textbox(value="40", interactive=False, scale=1),
                    gr.Textbox(value="3e-5", interactive=False, scale=1),
                    gr.Textbox(value="0.1", interactive=False, scale=1)]
    else:
        return [
            gr.Dropdown(choices=VICTIM_OUTPUT_DIR_LIST, value=VICTIM_OUTPUT_DIR_LIST[0], interactive=False, scale=2),
            gr.Dropdown(choices=METRIC_FOR_BEST_MODEL_LIST, value=METRIC_FOR_BEST_MODEL_LIST[0],
                        interactive=False, scale=1),
            gr.Slider(minimum=1, maximum=1024, value=32, step=1, interactive=False, scale=1),
            gr.Textbox(value="20", interactive=False, scale=1),
            gr.Textbox(value="5e-5", interactive=False, scale=1),
            gr.Textbox(value="0.0", interactive=False, scale=1)
        ]


def change_victim_language_model(pretrained_language_model: str, dataset: str, attack_method: str) -> ["gr.Dropdown",
                                                                                                       "gr.Dropdown"]:
    if dataset:
        if attack_method:
            victim_language_model_value = "{}+{}".format(pretrained_language_model, dataset.split(".")[0])
            output_dir_value = "Adv.{}.{}".format(attack_method, victim_language_model_value)
            return [gr.Dropdown(choices=[victim_language_model_value], value=victim_language_model_value,
                                interactive=False, scale=1),
                    gr.Dropdown(choices=[output_dir_value], value=output_dir_value, interactive=False, scale=1)]
        else:
            victim_language_model_value = "{}+{}".format(pretrained_language_model, dataset.split(".")[0])
            return [gr.Dropdown(choices=[victim_language_model_value], value=victim_language_model_value,
                                interactive=False, scale=1),
                    gr.Dropdown(choices=ADV_OUTPUT_DIR_LIST, value=ADV_OUTPUT_DIR_LIST[0], interactive=False, scale=1)]
    else:
        return [gr.Dropdown(interactive=False, scale=1),
                gr.Dropdown(choices=ADV_OUTPUT_DIR_LIST, value=ADV_OUTPUT_DIR_LIST[0], interactive=False, scale=1)]


def change_test_dataset(pretrained_language_model: str, dataset: str, attack_method: str) -> ["gr.Dropdown",
                                                                                              "gr.Dropdown"]:
    if pretrained_language_model:
        if attack_method:
            victim_language_model_value = "{}+{}".format(pretrained_language_model, dataset.split(".")[0])
            output_dir_value = "Adv.{}.{}".format(attack_method, victim_language_model_value)
            return [gr.Dropdown(choices=[victim_language_model_value], value=victim_language_model_value,
                                interactive=False, scale=1),
                    gr.Dropdown(choices=[output_dir_value], value=output_dir_value, interactive=False, scale=1)]
        else:
            victim_language_model_value = "{}+{}".format(pretrained_language_model, dataset.split(".")[0])
            return [gr.Dropdown(choices=[victim_language_model_value], value=victim_language_model_value,
                                interactive=False, scale=1),
                    gr.Dropdown(choices=ADV_OUTPUT_DIR_LIST, value=ADV_OUTPUT_DIR_LIST[0], interactive=False, scale=1)]
    else:
        return [gr.Dropdown(interactive=False, scale=1),
                gr.Dropdown(choices=ADV_OUTPUT_DIR_LIST, value=ADV_OUTPUT_DIR_LIST[0], interactive=False, scale=1)]


def change_attack_method(victim_language_model: str, attack_method: str) -> "gr.Dropdown":
    if victim_language_model:
        output_dir_value = "Adv.{}.{}".format(attack_method, victim_language_model)
        return gr.Dropdown(choices=[output_dir_value], value=output_dir_value, interactive=False, scale=1)
    else:
        return gr.Dropdown(choices=ADV_OUTPUT_DIR_LIST, value=ADV_OUTPUT_DIR_LIST[0], interactive=False, scale=1)


def change_input_dir(lang: str) -> ["gr.Button", "gr.Button", "gr.DataFrame", "gr.Button", "gr.Markdown"]:
    return [gr.Button(variant="secondary", interactive=True, scale=1),
            gr.Button(variant="primary", interactive=False, scale=1),
            gr.Dataframe(
                headers=["label", "orig", "adv", "score"], datatype=["str", "str", "str", "number"], type="pandas",
                column_widths=["10%", "40%", "40%", "10%"], wrap=True, visible=False, interactive=False, scale=1
            ),
            gr.Button(variant="primary", visible=False, interactive=False, scale=1),
            gr.Markdown(ALERTS["info_ready"][lang])]


def start_annotation(lang: str, input_dir: str) -> ["gr.Button", "gr.DataFrame", "gr.Button", "gr.Markdown"]:
    input_data_path = os.path.join(DEFAULT_DATA_DIR, input_dir, "data_filtered.txt")
    input_data = pd.read_csv(input_data_path, sep="\t", lineterminator="\n",
                             header=None, names=["label", "orig", "adv"])
    input_data["score"] = [0] * input_data.shape[0]
    print(input_data)
    return [gr.Button(variant="primary", interactive=False, scale=1),
            gr.Dataframe(
                value=input_data, headers=["label", "orig", "adv", "score"], datatype=["str", "str", "str", "number"],
                row_count=(input_data.shape[0], "fixed"), col_count=(input_data.shape[1], "fixed"), type="pandas",
                column_widths=["10%", "40%", "40%", "10%"], wrap=True, visible=True, interactive=True, scale=1
            ),
            gr.Button(variant="primary", visible=True, interactive=True, scale=1),
            gr.Markdown(ALERTS["info_annotating"][lang])]


def start_submission(lang: str, input_dir: str, data_frame: pd.DataFrame) -> ["gr.Button", "gr.Markdown"]:
    output_data_path = os.path.join(DEFAULT_DATA_DIR, input_dir, "data_annotated.txt")
    data_frame.to_csv(output_data_path, sep="\t", lineterminator="\n", header=False, index=False)
    return [gr.Button(variant="primary", visible=True, interactive=False, scale=1),
            gr.Markdown(ALERTS["info_submitted"][lang])]
