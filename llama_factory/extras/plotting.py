# Source Attribution:
# The majority of the code is derived from the following source:
# - LLaMA-Factory GitHub Repository: https://github.com/hiyouga/LLaMA-Factory
# - Tag: v0.9.0
# - File: /src/llamafactory/extras/ploting.py
# - Reference Paper: LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models (Zheng et al., ACL 2024)

import math
from typing import List

from .packages import is_matplotlib_available

if is_matplotlib_available():
    import matplotlib.figure
    import matplotlib.pyplot as plt


def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def generate_epoch_value_plot(epoch_list: List[float], value_list: List[float]) -> "matplotlib.figure.Figure":
    r"""
    Plots curves.
    """
    plt.close("all")
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epoch_list, value_list, color="#1f77b4", alpha=0.4, label="original")
    ax.plot(epoch_list, smooth(value_list), color="#1f77b4", label="smoothed")
    ax.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    return fig
