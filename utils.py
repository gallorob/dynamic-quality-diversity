import logging
import os
import shutil
from typing import Dict, Any, List, Optional

import numpy as np
import numpy.typing as npt
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from ribs.archives import ArchiveBase, GridArchive
from ribs.visualize import grid_archive_heatmap

from configs import configs


vectorized_beta = np.vectorize(lambda a, b, rng: rng.beta(a, b))

plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['FreeSerif']
plt.rcParams['font.size'] = 16

colors = [
    (1.0, 0.0, 0.1607843137254902),
    (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
    (0.4, 0.6509803921568628, 0.11764705882352941),
    (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
    (0.0, 0.8235294117647058, 0.8352941176470589),
    (1.0, 0.4980392156862745, 0.0),
    (0.6862745098039216, 0.5529411764705883, 0.0),
    (0.4980392156862745, 0.5019607843137255, 0.803921568627451),
    (0.7019607843137254, 0.9137254901960784, 0.0),
    (0.7686274509803922, 0.1803921568627451, 0.3764705882352941),
    (0.6509803921568628, 0.33725490196078434, 0.1568627450980392),
    (0.9686274509803922, 0.5058823529411764, 0.7490196078431373),
    (0.5529411764705883, 0.8274509803921568, 0.7803921568627451),
    (0.7450980392156863, 0.7294117647058823, 0.8549019607843137),
    (0.984313725490196, 0.5019607843137255, 0.4470588235294118),
    (0.5019607843137255, 0.6941176470588235, 0.8274509803921568)
]

markers = [
    '*', 'o', '1', '2', '3', '4', '8', 's', 'p', 'x', 'P', 'D', '|', '_', '<', '>', 'v', '^'
]
markersize = 10


def make_folders(exp_name: str):
    # create results directory if it doesn't exist
    os.makedirs(configs.plots_dir, exist_ok=True)
    # create the directory for this experiment
    os.makedirs(os.path.join(configs.plots_dir, exp_name), exist_ok=True)
    # clear temporary images (for now, containing just the archive frames)
    if os.path.exists(os.path.join(configs.plots_dir, './tmp')):
        shutil.rmtree(os.path.join(configs.plots_dir, './tmp'))
    # recreate the temporary folder, making sure it has been removed before
    os.makedirs(os.path.join(configs.plots_dir, './tmp'), exist_ok=False)


def plot_metric(metric: str,
                metrics: Dict[str, Any],
                title: str,
                ylabel: str,
                fname: str,
                markevery: int = configs.total_iters // 10,
                policies: List[str] = configs.policies,
                rolling_average_window: Optional[int] = 1,
                savefig: bool = True
                ):

    plt.figure(figsize=(8, 6))
    for policy, marker, color in zip(policies, markers, colors):
        plotting_metrics = metrics[policy]
        if rolling_average_window is not None:
            to_plot = np.convolve(plotting_metrics[metric], np.ones(rolling_average_window), 'same') / rolling_average_window
        else:
            to_plot = plotting_metrics[metric]
        plt.plot(np.cumsum(plotting_metrics['Evaluations']) if configs.early_stopping else np.arange(len(to_plot)),
                 to_plot,
                 marker=marker, markersize=10,
                 color=color,
                 markevery=markevery, linestyle='solid', linewidth=2)

    plt.title(title if rolling_average_window is None else f'{title} (w={rolling_average_window})')
    plt.xlabel("Evaluations")
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend(policies, title="Policy", bbox_to_anchor=(1.04, 1), loc='upper left')
    if savefig: plt.savefig(f'{fname}.png', bbox_inches='tight', transparent=True)
    plt.close()
