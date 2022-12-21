import os
import sys
import psutil
import gc
import argparse
from time import time
from datetime import datetime

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from torchvision.models._utils import IntermediateLayerGetter

import segmentation_models_pytorch as smp
from models.modeling.deeplab import *

## from detectron2 import model_zoo
## from detectron2.engine import DefaultPredictor
## from detectron2.config import get_cfg

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_labels,
    create_pairwise_bilateral,
    create_pairwise_gaussian,
)

from dataloader.referit_loader import *

from models.model import JointModel
from losses import Loss
from utils.utils import print_
from utils import im_processing
from utils.metrics import compute_mask_IOU

def get_closest_mask(ref_ids, sent_ids, pred_mask, predictor, refer):

    new_pred_mask = torch.zeros_like(pred_mask)
    mask_dim = pred_mask.shape[-1]

    for i in range(ref_ids.shape[0]):
        ref_id = ref_ids[i].item()
        sent_id = sent_ids[i].item()
        ref = refer.Refs[ref_id]
        image_id = ref["image_id"]
        image_data = refer.Imgs[image_id]
        image = Image.open(os.path.join(
            refer.IMAGE_DIR, image_data['file_name'])).convert(mode='RGB')

        image = image.resize((448, 448))
        image = np.array(image)

        sentence = list(
            filter(lambda x: (x['sent_id'] == sent_id), ref['sentences']))[0]
        tokens = sentence['tokens']

        output = predictor(image)
        masks_ = output["instances"].pred_masks.detach().cpu().numpy()

        masks = np.zeros((masks_.shape[0], mask_dim, mask_dim))
        for k in range(masks.shape[0]):
            mask = masks_[k]
            mask = Image.fromarray(mask).resize((mask_dim, mask_dim))
            masks[k] = mask

        masks = torch.from_numpy(masks)
        p_mask = pred_mask[i]

        if masks.shape[0] > 0:
            p_mask_ = torch.stack([p_mask]*masks.shape[0])
            iou = iou_seg(p_mask_.detach().cpu(), masks, False)
            max_indx = iou.argmax()

            if iou[max_indx] != 0:
                new_pred_mask[i] = masks[max_indx]
            else:
                new_pred_mask[i] = p_mask

        else:
            new_pred_mask[i] = p_mask

    return new_pred_mask.cuda()


@torch.no_grad()
def evaluate(
    val_loader,
    joint_model,
    image_encoder,
    loss_func,
    experiment,
    args,
    predictor=None,
    dcrf_use=False,
):

    image_encoder.eval()
    joint_model.eval()

    pid = os.getpid()
    py = psutil.Process(pid)

    # import pdb; pdb.set_trace()

    total_loss = 0
    total_score = 0
    n_iter = 0

    feature_dim = args.mask_dim // 4

    data_len = len(val_loader)

    epoch_start = time()
    for step, batch in enumerate(val_loader):

        image = batch["image"].cuda(non_blocking=True)
        orig_image = batch["orig_image"].numpy()
        phrase = batch["phrase"].cuda(non_blocking=True)
        phrase_mask = batch["phrase_mask"].cuda(non_blocking=True)

        orig_image = np.uint8(orig_image * 255)

        gt_mask = batch["seg_mask"]  # .cuda(non_blocking=True)
        gt_mask = gt_mask.squeeze(dim=1)

        batch_size = image.shape[0]
        img_mask = torch.ones(
            batch_size, feature_dim * feature_dim, dtype=torch.int64
        ).cuda(non_blocking=True)

        start_time = time()
        with torch.no_grad():
            img = image_encoder(image)

        output_mask = joint_model(img, phrase, img_mask, phrase_mask)
        output_mask = output_mask.detach().cpu()
        end_time = time()
        elapsed_time = end_time - start_time

        start_dcrf = time()
        if dcrf_use and batch_size == 1:

            H, W = orig_image[0].shape[:-1]
            mask_upsample = (
                F.interpolate(
                    output_mask.unsqueeze(0),
                    scale_factor=8,
                    mode="bilinear",
                    align_corners=True,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            _, labels = np.unique(mask_upsample > 0.5, return_inverse=True)
            n_labels = 2

            d = dcrf.DenseCRF2D(H, W, n_labels)

            U = unary_from_labels(labels, n_labels, gt_prob=0.6, zero_unsure=False)
            d.setUnaryEnergy(U)

            d.addPairwiseGaussian(
                sxy=3, compat=3
            )

            d.addPairwiseBilateral(
                sxy=20, srgb=3, rgbim=orig_image[0], compat=10
            ) 

            Q = d.inference(5)

            pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)

            output_mask_ = im_processing.resize_and_crop(
                pred_raw_dcrf, args.mask_dim, args.mask_dim
            )
            output_mask[0] = torch.from_numpy(output_mask_)  # .cuda()

            ## fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 10))
            ## ax[0].imshow(orig_image[0])
            ## ax[0].set_axis_off()

            ## ax[1].imshow(gt_mask[0].numpy())
            ## ax[1].set_title("gt")
            ## ax[1].set_axis_off()

            ## ax[2].imshow(pred_raw_dcrf)
            ## ax[2].set_title("crf_out")
            ## ax[2].set_axis_off()

            ## ax[3].imshow(output_mask[0])
            ## ax[3].set_title("out mask")
            ## ax[3].set_axis_off()

            ## ax[4].imshow(output_mask_)
            ## ax[4].set_title("reshape crf out")
            ## ax[4].set_axis_off()

            ## plt.savefig("dcrf_out.png")

        end_dcrf = time()
        elapsed_dcrf = end_dcrf - start_dcrf

        loss = loss_func(output_mask, gt_mask)

        batch_score = compute_mask_IOU(output_mask, gt_mask)

        total_loss += float(loss.item())
        total_score += float(batch_score)
        
        if step % 50 == 0:
            gc.collect()
            memoryUse = py.memory_info()[0] / 2.0 ** 20

            timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")
            print_(
                f"{timestamp} Validation: iter [{step:3d}/{data_len}] loss {total_loss/(step + 1):.4f} score {total_score/(step + 1):.4f} memory_use {memoryUse:.3f}MB elapsed {elapsed_time:.2f}"
            )
            # break

    val_loss = total_loss / data_len
    val_acc = total_score / data_len
    timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")
    print_(
        f"{timestamp} [{args.task}-{args.split}] loss {val_loss:.4f} score {val_acc:.4f}"
    )
    return val_loss, val_acc


@torch.no_grad()
def evaluate_after_training(joint_model, image_encoder, loss_func, experiment, args):

    torch.cuda.empty_cache()

    joint_model.eval()
    image_encoder.eval()

    predictor = None
    if args.predictor:
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        cfg.NUM_GPUS = torch.cuda.device_count()
        predictor = DefaultPredictor(cfg)  # .cuda()

    if args.task in ["unc", "unc+"]:
        splits = ["val", "testA", "testB"]
    elif args.task in ["gref"]:
        splits = ["val"]
    else:  # refclef
        splits = ["test"]

    for split in splits:

        args.split = split

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((args.image_dim, args.image_dim))

        val_dataset = ReferDataset(
            data_root=args.dataroot,
            dataset=args.task,
            transform=transforms.Compose([resize, to_tensor, normalize]),
            annotation_transform=transforms.Compose([ResizeAnnotation(args.mask_dim)]),
            split=split,
            max_query_len=args.phrase_len,
            glove_path=args.glove_path,
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("Loaded Dataset on Dataloader")

        with torch.no_grad():
            loss, accuracy = evaluate(
                val_loader,
                joint_model,
                image_encoder,
                loss_func,
                experiment,
                args,
                predictor=predictor,
                dcrf_use=args.dcrf,
            )

        if experiment:
            experiment.log({f"{args.task}_{split}": accuracy})

        torch.cuda.empty_cache()
