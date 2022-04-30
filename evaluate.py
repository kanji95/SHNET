import os
import psutil
import gc
from time import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.position_encoding import *

from utilities.utils import print_
from utilities.metrics import compute_mask_IOU


@torch.no_grad()
def evaluate(
    val_loader,
    joint_model,
    image_encoder,
    epochId,
    args,
):

    image_encoder.eval()
    joint_model.eval()

    pid = os.getpid()
    py = psutil.Process(pid)

    total_loss = 0
    total_inter, total_union = 0, 0
    total_accuracy = 0

    feature_dim = args.feature_dim

    bce_loss = nn.BCELoss()

    data_len = len(val_loader)

    print_(
        "\n================================================= Evaluating only Grounding Network ======================================================="
    )

    for step, batch in enumerate(val_loader):

        img = batch["image"].cuda(non_blocking=True)
        phrase = batch["phrase"].cuda(non_blocking=True)
        phrase = phrase.squeeze(dim=1)
        phrase_mask = batch["phrase_mask"].cuda(non_blocking=True)

        gt_mask = batch["seg_mask"].cuda(non_blocking=True)
        gt_mask = gt_mask.squeeze(dim=1)

        batch_size = img.shape[0]
        img_mask = torch.ones(
            batch_size, feature_dim * feature_dim, dtype=torch.int64
        ).cuda(non_blocking=True)

        start_time = time()

        img = image_encoder(img)
        mask = joint_model(img, phrase, img_mask, phrase_mask)

        end_time = time()
        elapsed_time = end_time - start_time

        loss = bce_loss(mask, gt_mask)

        inter, union = compute_mask_IOU(
            mask, gt_mask, args.threshold
        )

        total_inter += inter.item()
        total_union += union.item()
        
        accuracy = 0
        total_accuracy += accuracy

        total_loss += float(loss.item())

        if step % 200 == 0:
            gc.collect()
            memoryUse = py.memory_info()[0] / 2.0 ** 20

            timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

            curr_loss = total_loss / (step + 1)
            overall_IOU = total_inter / total_union
            curr_acc = total_accuracy / (step + 1)

            print_(
                f"{timestamp} Validation: iter [{step:3d}/{data_len}] loss {curr_loss:.4f} overall_IOU {overall_IOU:.4f} curr_acc {curr_acc:.4f} memory_use {memoryUse:.3f}MB elapsed {elapsed_time:.2f}"
            )

    val_loss = total_loss / data_len
    val_IOU = total_inter / total_union

    val_acc = total_accuracy / data_len

    timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")
    print_(
        f"{timestamp} Validation: EpochId: {epochId:2d} loss {val_loss:.4f} overall_IOU {val_IOU:.4f} val_acc {val_acc:.4f}"
    )
    print_("============================================================================================================================================\n")
    
    return val_loss, val_IOU
