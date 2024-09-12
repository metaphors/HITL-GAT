# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/webui/locales.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

LOCALES = {
    "lang": {
        "en": {
            "label": "Language"
        },
        "zh": {
            "label": "语言"
        }
    },
    "pretrained_language_model": {
        "en": {
            "label": "Pretrained Language Model"
        },
        "zh": {
            "label": "预训练语言模型"
        }
    },
    "victim_language_model": {
        "en": {
            "label": "Victim Language Model"
        },
        "zh": {
            "label": "受害者语言模型"
        }
    },
    "model_for_generating_adversarial_examples": {
        "en": {
            "label": "Model for Generating Adversarial Examples"
        },
        "zh": {
            "label": "用于生成对抗性样本的模型"
        }
    },
    "model_for_evaluating_adversarial_robustness": {
        "en": {
            "label": "Model for Evaluating Adversarial Robustness"
        },
        "zh": {
            "label": "用于评估对抗鲁棒性的模型"
        }
    },
    "dataset": {
        "en": {
            "label": "Dataset"
        },
        "zh": {
            "label": "数据集"
        }
    },
    "dataset_for_evaluating_adversarial_robustness": {
        "en": {
            "label": "Dataset for Evaluating Adversarial Robustness"
        },
        "zh": {
            "label": "用于评估对抗鲁棒性的数据集"
        }
    },
    "data_preview_btn": {
        "en": {
            "value": "Preview Dataset"
        },
        "zh": {
            "value": "预览数据集"
        },
    },
    "preview_count": {
        "en": {
            "label": "Count"
        },
        "zh": {
            "label": "总数"
        },
    },
    "page_index": {
        "en": {
            "label": "Page"
        },
        "zh": {
            "label": "页数"
        }
    },
    "prev_btn": {
        "en": {
            "value": "Prev"
        },
        "zh": {
            "value": "上一页"
        }
    },
    "next_btn": {
        "en": {
            "value": "Next"
        },
        "zh": {
            "value": "下一页"
        }
    },
    "close_btn": {
        "en": {
            "value": "Close"
        },
        "zh": {
            "value": "关闭"
        }
    },
    "preview_samples": {
        "en": {
            "label": "Samples"
        },
        "zh": {
            "label": "样本"
        }
    },
    "attack_method": {
        "en": {
            "label": "Attack Method"
        },
        "zh": {
            "label": "攻击方法"
        }
    },
    "output_dir": {
        "en": {
            "label": "Output Directory"
        },
        "zh": {
            "label": "输出目录"
        }
    },
    "input_dir": {
        "en": {
            "label": "Input Directory"
        },
        "zh": {
            "label": "输入目录"
        }
    },
    "metric_for_best_model": {
        "en": {
            "label": "Metric for Best Model"
        },
        "zh": {
            "label": "最优模型衡量指标"
        }
    },
    "batch_size": {
        "en": {
            "label": "Batch Size"
        },
        "zh": {
            "label": "批量大小"
        }
    },
    "num_train_epochs": {
        "en": {
            "label": "Train Epochs"
        },
        "zh": {
            "label": "训练轮数"
        }
    },
    "learning_rate": {
        "en": {
            "label": "Learning Rate"
        },
        "zh": {
            "label": "学习率"
        }
    },
    "warmup_ratio": {
        "en": {
            "label": "Warmup Ratio"
        },
        "zh": {
            "label": "预热比"
        }
    },
    "arg_save_btn": {
        "en": {
            "value": "Save Arguments"
        },
        "zh": {
            "value": "保存训练参数"
        }
    },
    "arg_load_btn": {
        "en": {
            "value": "Load Arguments"
        },
        "zh": {
            "value": "载入训练参数"
        }
    },
    "start_btn": {
        "en": {
            "value": "Start"
        },
        "zh": {
            "value": "开始"
        }
    },
    "stop_btn": {
        "en": {
            "value": "Abort"
        },
        "zh": {
            "value": "中断"
        }
    },
    "filter_btn": {
        "en": {
            "value": "Filter (levenshtein_distance/text_length<=0.1)"
        },
        "zh": {
            "value": "过滤 (levenshtein_distance/text_length<=0.1)"
        }
    },
    "annotate_btn": {
        "en": {
            "value": "Annotate (1~5, 1↓ 5↑)"
        },
        "zh": {
            "value": "标注 (1~5, 1↓ 5↑)"
        }
    },
    "submit_btn": {
        "en": {
            "value": "Submit"
        },
        "zh": {
            "value": "提交"
        }
    },
    "evaluate_btn": {
        "en": {
            "value": "Evaluate"
        },
        "zh": {
            "value": "评估"
        }
    },
    "f1_viewer": {
        "en": {
            "label": "F1/Macro-F1"
        },
        "zh": {
            "label": "F1/Macro-F1"
        }
    },
    "accuracy_viewer": {
        "en": {
            "label": "Accuracy"
        },
        "zh": {
            "label": "准确率"
        }
    },
    "loss_viewer": {
        "en": {
            "label": "Loss"
        },
        "zh": {
            "label": "损失值"
        }
    },
    "output_box": {
        "en": {
            "value": "Ready."
        },
        "zh": {
            "value": "准备就绪。"
        }
    }
}

ALERTS = {
    "err_conflict": {
        "en": "A process is in running, please abort it first.",
        "zh": "任务已存在，请先中断它。"
    },
    "err_no_model": {
        "en": "Please select a pretrained language model.",
        "zh": "请选择预训练语言模型。"
    },
    "err_no_dataset": {
        "en": "Please select a dataset.",
        "zh": "请选择数据集。"
    },
    "err_no_attack_method": {
        "en": "Please select a attack method.",
        "zh": "请选择攻击方法。"
    },
    "err_failed": {
        "en": "Failed.",
        "zh": "训练出错。"
    },
    "err_filter": {
        "en": "Failed.",
        "zh": "过滤出错。"
    },
    "err_evaluator": {
        "en": "Failed.",
        "zh": "评估出错。"
    },
    "info_aborting": {
        "en": "Aborted, wait for terminating...",
        "zh": "已中断，正在等待进程结束……"
    },
    "info_finished": {
        "en": "Finished.",
        "zh": "训练完毕。"
    },
    "info_aborted": {
        "en": "Ready.",
        "zh": "准备就绪。"
    },
    "info_filtering": {
        "en": "Filtering...",
        "zh": "过滤中……"
    },
    "info_filtered": {
        "en": "Filtered.",
        "zh": "过滤完毕。"
    },
    "info_annotating": {
        "en": "Annotating...",
        "zh": "标注中……"
    },
    "info_submitted": {
        "en": "Submitted.",
        "zh": "提交完毕。"
    },
    "info_ready": {
        "en": "Ready.",
        "zh": "准备就绪。"
    },
    "info_evaluating": {
        "en": "Evaluating...",
        "zh": "评估中……"
    },
    "info_evaluated": {
        "en": "Evaluated.",
        "zh": "评估完毕。"
    }
}
