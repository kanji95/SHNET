import os
import psutil
import gc
from time import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.utils import print_, grad_check
from utilities.metrics import compute_mask_IOU


def train(
    train_loader,
    joint_model,
    image_encoder,
    optimizer,
    experiment,
    epochId,
    args,
):

    pid = os.getpid()
    py = psutil.Process(pid)

    joint_model.train()
    image_encoder.eval()

    optimizer.zero_grad()

    total_loss = 0
    total_inter, total_union = 0, 0

    feature_dim = args.feature_dim

    bce_loss = nn.BCELoss()

    data_len = len(train_loader)

    epoch_start = time()

    print_("\n=========================================================== Training Grounding Network ===================================================")
    train_loader.dataset.unc_prob = 0
    for step, batch in enumerate(train_loader):
        iterId = step + (epochId * data_len) - 1
        with torch.no_grad():
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
        with torch.no_grad():
            img = image_encoder(img)
            
        mask = joint_model(img, phrase, img_mask, phrase_mask)
        loss = bce_loss(mask, gt_mask)
        loss.backward()
        if iterId % 500 == 0 and args.grad_check:
            grad_check(joint_model.named_parameters(), experiment)
        optimizer.step()
        joint_model.zero_grad()
        end_time = time()
        elapsed_time = end_time - start_time
        with torch.no_grad():
            inter, union = compute_mask_IOU(mask, gt_mask, args.threshold)
        total_inter += inter.item()
        total_union += union.item()
        total_loss += float(loss.item())
        if iterId % 200 == 0 and step != 0:
            gc.collect()
            memoryUse = py.memory_info()[0] / 2.0 ** 20
            timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")
            curr_loss = total_loss / (step + 1)
            curr_IOU = total_inter / total_union
            lr = optimizer.param_groups[0]["lr"]
            print_(
                f"{timestamp} Epoch:[{epochId:2d}/{args.epochs:2d}] iter {iterId:6d} loss {curr_loss:.4f} IOU {curr_IOU:.4f} memory_use {memoryUse:.3f}MB lr {lr:.7f} elapsed {elapsed_time:.2f}"
            )
    epoch_end = time()
    epoch_time = epoch_end - epoch_start

    timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

    train_loss = total_loss / data_len
    overall_IOU = total_inter / total_union

    experiment.log({"loss": train_loss, "IOU": overall_IOU})

    print_(
        f"{timestamp} FINISHED Epoch:{epochId:2d} loss {train_loss:.4f} overall_IOU {overall_IOU:.4f} elapsed {epoch_time:.2f}"
    )
    print_("============================================================================================================================================\n")
