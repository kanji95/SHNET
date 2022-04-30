import os
import math
import random
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image

from tqdm import tqdm

import torch
import torch.optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import nvidia_smi

import wandb


def print_(statement, default_gpu=True):
    if default_gpu:
        print(statement, flush=True)

def log_gpu_usage():
    nvidia_smi.nvmlInit()
    print_(f"Driver Version: {nvidia_smi.nvmlSystemGetDriverVersion()}")
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print_(f"mem: {mem_res.used / (1024**2)} (GiB)")  # usage in GiB
        print_(f"mem: {100 * (mem_res.used / mem_res.total):.3f}%")
    nvidia_smi.nvmlShutdown()

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.5 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

@torch.no_grad()
def grad_check(named_parameters, experiment):
    thresh = 0.001

    layers = []
    max_grads = []
    mean_grads = []
    max_colors = []
    mean_colors = []

    for n, p in named_parameters:
        # import pdb; pdb.set_trace()
        # print(n)
        if p.requires_grad and "bias" not in n:
            max_grad = p.grad.abs().max()
            mean_grad = p.grad.abs().mean()
            layers.append(n)
            max_grads.append(max_grad)
            mean_grads.append(mean_grad)

    for i, (val_mx, val_mn) in enumerate(zip(max_grads, mean_grads)):
        if val_mx > thresh:
            max_colors.append("r")
        else:
            max_colors.append("g")
        if val_mn > thresh:
            mean_colors.append("b")
        else:
            mean_colors.append("y")
    ax = plt.subplot(111)
    x = np.arange(len(layers))
    w = 0.3

    ax.bar(x - w, max_grads, width=w, color=max_colors, align="center", hatch="////")
    ax.bar(x, mean_grads, width=w, color=mean_colors, align="center", hatch="----")

    plt.xticks(x - w / 2, layers, rotation="vertical")
    plt.xlim(left=-1, right=len(layers))
    plt.ylim(bottom=0.0, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Gradient Values")
    plt.title("Model Gradients")

    hatch_dict = {0: "////", 1: "----"}
    legends = []
    for i in range(len(hatch_dict)):
        p = patches.Patch(facecolor="#DCDCDC", hatch=hatch_dict[i])
        legends.append(p)

    ax.legend(legends, ["Max", "Mean"])

    plt.grid(True)
    plt.tight_layout()
    experiment.log({"Gradients": wandb.Image(plt)})
    plt.close()

@torch.no_grad()
def log_predicitons(
    orig_image, orig_phrase, output_mask, orig_mask, experiment, title="train", k=4
):
    indices = random.choices(range(output_mask.shape[0]), k=k)

    figure, axes = plt.subplots(nrows=k, ncols=3)
    for i, index in enumerate(indices):
        index = indices[i]

        pred_mask = np.uint8(output_mask[i] * 255)

        axes[i, 0].imshow(orig_mask[i])
        axes[i, 0].set_title("ground truth")
        axes[i, 0].set_axis_off()

        axes[i, 1].imshow(orig_image[i])
        axes[i, 1].set_title(orig_phrase[i])
        axes[i, 1].set_axis_off()

        axes[i, 2].imshow(pred_mask)
        axes[i, 2].set_title("predicted mask")
        axes[i, 2].set_axis_off()

    figure.tight_layout()
    experiment.log({f"{title}_segmentation": wandb.Image(figure)}, commit=True)
    plt.close(figure)
