import os
import argparse
from time import time
from datetime import datetime

import numpy as np
import skimage
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torchvision.models._utils import IntermediateLayerGetter

# import denseCRF
# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import (
#     unary_from_labels,
#     create_pairwise_bilateral,
#     create_pairwise_gaussian,
# )

from models.modeling.deeplab import *
from dataloader.referit_loader import *

from losses import Loss
from models.model import JointModel
from utilities import im_processing
from utilities.utils import log_gpu_usage, print_
from utilities.metrics import compute_mask_IOU

# from skimage.transform import resize
# from memory_profiler import profile

def get_args_parser():
    parser = argparse.ArgumentParser("Refering Image Segmentation", add_help=False)

    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--gamma", default=0.7, type=float)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--grad_check", default=False, action="store_true")

    ## DCRF
    parser.add_argument("--use_dcrf", default=False, action="store_true")

    # MODEL Params
    parser.add_argument(
        "--image_encoder",
        type=str,
        default="deeplabv3_plus",
        choices=[
            "vgg16",
            "vgg19",
            "resnet50",
            "resnet101",
            "deeplabv2",
            "deeplabv3_resnet101",
            "deeplabv3_plus",
            "dino",
        ],
    )
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--transformer_dim", default=256, type=int)
    parser.add_argument("--feature_dim", default=14, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--attn_type", type=str, default="normal", choices=["normal", "linear", "cross", "multimodal", "context", "linear_context"])

    ## Evalute??
    parser.add_argument("--model_path", default="model_unc.pth", type=str)

    parser.add_argument(
        "--dataroot", type=str, default="/ssd_scratch/cvit/kanishk/referit/"
    )
    parser.add_argument(
        "--glove_path", default="/ssd_scratch/cvit/kanishk/glove/", type=str
    )
    parser.add_argument(
        "--task",
        default="unc",
        type=str,
        choices=[
            "unc",
            "unc+",
            "gref",
            "referit",
        ],
    )
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--cache_type", type=str, default="full")
    parser.add_argument("--image_dim", type=int, default=448)
    parser.add_argument("--mask_dim", type=int, default=56)
    parser.add_argument("--channel_dim", type=int, default=512)
    parser.add_argument("--phrase_len", type=int, default=20)

    parser.add_argument("--threshold", type=float, default=0.40)

    return parser

@torch.no_grad()
## @profile
def evaluate(image_encoder, joint_model, val_loader, args):

    image_encoder.eval()
    joint_model.eval()

    total_inter = 0
    total_union = 0

    total_dcrf_inter, total_dcrf_union = 0, 0

    mean_IOU = 0
    mean_dcrf_IOU = 0

    feature_dim = args.feature_dim

    prec_at_x = {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0}
    prec_dcrf_at_x = {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0}

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    data_len = len(val_loader)
    total_time = 0
    for step, batch in enumerate(val_loader):

        image = batch["image"].cuda(non_blocking=True)

        orig_phrase = batch["orig_phrase"][0]

        ## if "the right half of the sandwich" not in orig_phrase:
        ##      continue
        ## print(orig_phrase)

        phrase = batch["phrase"].cuda(non_blocking=True)
        phrase_mask = batch["phrase_mask"].cuda(non_blocking=True)
        index = batch["index"]

        gt_mask = batch["seg_mask"]
        gt_mask = gt_mask.squeeze(dim=1)

        batch_size = image.shape[0]
        img_mask = torch.ones(batch_size, feature_dim * feature_dim, dtype=torch.int64).cuda(
            non_blocking=True
        )

        torch.cuda.synchronize()
        start = time()

        with torch.no_grad():
            img = image_encoder(image)
            
        output_mask = joint_model(img, phrase, img_mask, phrase_mask)

        end = time()
        torch.cuda.synchronize()

        elapsed = end - start
        total_time += elapsed

        output_mask = output_mask.detach().cpu()

        if args.use_dcrf:

            ## orig_image = batch["orig_image"].numpy()
            ### img_path = batch["img_path"][0]
            ### orig_image = Image.open(img_path).convert("RGB").resize((args.image_dim, args.image_dim))
            ### orig_image = np.array(orig_image)
            
            orig_image = image[0].cpu().permute(1, 2, 0).mul_(std).add_(mean).numpy()
            
            ## orig_image = resize(orig_image, (args.image_dim, args.image_dim), anti_aliasing=True)
            proc_im = skimage.img_as_ubyte(orig_image)
            # orig_image = np.uint8(orig_image * 255)

            ## H, W = orig_image[0].shape[:-1]
            H, W = orig_image.shape[:-1]

            ## sigma_val = output_mask[0]
            ## sigma_val = (output_mask > output_mask.mean()).float()[0]
            sigma_val = (output_mask > args.threshold).float()[0]  

            ## n_labels = 2
            ## d = dcrf.DenseCRF2D(H, W, n_labels)
            ## U = np.expand_dims(-np.log(sigma_val), axis=0)
            ## U_ = np.expand_dims(-np.log(1 - sigma_val), axis=0)
            ## unary = np.concatenate((U_, U), axis=0)
            ## unary = unary.reshape((2, -1))
            ## d.setUnaryEnergy(unary)
            ## d.addPairwiseGaussian(sxy=3, compat=5)
            ## d.addPairwiseBilateral(sxy=20, srgb=10, rgbim=proc_im, compat=10)
            ## Q = d.inference(5)
            ## pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)

            mask_pred = np.stack([1 - sigma_val, sigma_val], axis=-1)
            ## bilateral_wt = 1 # 10
            ## alpha = 5 # 20
            ## beta = 1 # 3
            ## spatial_wt = 1 # 3
            ## gamma = 2 # 3
            ## num_it = 5
            bilateral_wt = 10
            alpha = 20
            beta = 10
            spatial_wt = 5
            gamma = 3
            num_it = 5
            param = (bilateral_wt, alpha, beta, spatial_wt, gamma, num_it) 
            pred_raw_dcrf = denseCRF.densecrf(proc_im, mask_pred, param)

            dcrf_output_mask = torch.from_numpy(pred_raw_dcrf).unsqueeze(0)

        inter, union = compute_mask_IOU(output_mask, gt_mask, args.threshold)

        total_inter += inter.item()
        total_union += union.item()

        score = inter.item() / union.item()
        ## if score > 0.95:
        ##     print(orig_phrase, index.item(), score)

        mean_IOU += score

        total_score = total_inter / total_union

        for x in prec_at_x:
            if score > x:
                prec_at_x[x] += 1

        total_dcrf_score = 0
        if args.use_dcrf:
            dcrf_inter, dcrf_union = compute_mask_IOU(
                dcrf_output_mask, gt_mask, args.threshold
            )

            total_dcrf_inter += dcrf_inter.item()
            total_dcrf_union += dcrf_union.item()

            dcrf_score = dcrf_inter.item() / dcrf_union.item()

            mean_dcrf_IOU += dcrf_score

            total_dcrf_score = total_dcrf_inter / total_dcrf_union

            for x in prec_dcrf_at_x:
                if dcrf_score > x:
                    prec_dcrf_at_x[x] += 1
 
        ## if step > 2000:
        ##     break

        if step % 500 == 0:

            timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

            print_(
                f"{timestamp} Step: [{step:5d}/{data_len}] IOU {total_score:.5f} dcrf_IOU {total_dcrf_score}"
            )

    overall_IOU = total_inter / total_union
    mean_IOU = mean_IOU / data_len
	 
    overall_dcrf_IOU = 0
    if args.use_dcrf:
        overall_dcrf_IOU = total_dcrf_inter / total_dcrf_union
        mean_dcrf_IOU = mean_dcrf_IOU / data_len

    print_(
            f"Overall IOU: {overall_IOU}, Mean_IOU: {mean_IOU}, Overall_dcrf_IOU: {overall_dcrf_IOU}, Mean_dcrf_IOU: {mean_dcrf_IOU} Inference_time: {total_time/(step + 1)}"
    )

    for x in prec_at_x:
        percent = (prec_at_x[x] / data_len) * 100
        print_(f"{x}% IOU: {percent}%")

    print_("==================================")

    for x in prec_dcrf_at_x:
        percent = (prec_dcrf_at_x[x] / data_len) * 100
        print_(f"{x}% dcrf_IOU: {percent}%")


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print_(f"{device} being used with {n_gpu} GPUs!!")

    print_("Initializing dataset")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((args.image_dim, args.image_dim))

    tokenizer = None

    val_dataset = ReferDataset(
        data_root=args.dataroot,
        dataset=args.task,
        transform=transforms.Compose([resize, to_tensor, normalize]),
        annotation_transform=transforms.Compose([ResizeAnnotation(args.mask_dim)]),
        split=args.split,
        max_query_len=args.phrase_len,
        glove_path=args.glove_path,
    )

    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=1, num_workers=1, pin_memory=True
    )

    out_channels = 512
    return_layers = {"layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}

    if args.image_encoder == "resnet50" or args.image_encoder == "resnet101":
        stride = [1, 1, 1]
        model = torch.hub.load(
            "pytorch/vision:v0.6.1", args.image_encoder, pretrained=True
        )
        image_encoder = IntermediateLayerGetter(model, return_layers)
    elif args.image_encoder == "deeplabv2":
        stride = 2
        model = torch.hub.load(
            "kazuto1011/deeplab-pytorch",
            "deeplabv2_resnet101",
            pretrained="voc12",
            n_classes=21,
        )
        return_layers = {"layer3": "layer2", "layer4": "layer3", "layer5": "layer4"}
        image_encoder = IntermediateLayerGetter(model.base, return_layers)
    elif args.image_encoder == "deeplabv3_resnet101":
        stride = [1, 1, 1]
        model = torch.hub.load(
            "pytorch/vision:v0.6.1", args.image_encoder, pretrained=True
        )
        image_encoder = IntermediateLayerGetter(model.backbone, return_layers)
    elif args.image_encoder == "deeplabv3_plus":
        stride = [1, 1, 1]
        model = DeepLab(num_classes=21, backbone="resnet", output_stride=16)
        model.load_state_dict(
            torch.load("./models/deeplab-resnet.pth.tar")["state_dict"]
        )
        image_encoder = IntermediateLayerGetter(model.backbone, return_layers)
    else:
        raise NotImplemented("Model not implemented")

    for param in image_encoder.parameters():
        param.requires_grad_(False)
    image_encoder.eval()

    joint_model = JointModel(
        args,
        transformer_dim=args.transformer_dim,
        out_channels=args.channel_dim,
        stride=stride,
        num_layers=args.num_layers,
        num_encoder_layers=args.num_encoder_layers,
        dropout=args.dropout,
        mask_dim=args.mask_dim,
    )

    if n_gpu > 1:
        image_encoder = nn.DataParallel(image_encoder)
        joint_model = nn.DataParallel(joint_model)

    state_dict = torch.load(args.model_path)
    state_dict = state_dict["state_dict"]
    joint_model.load_state_dict(state_dict)

    joint_model.to(device)
    image_encoder.to(device)

    evaluate(image_encoder, joint_model, val_loader, args)


if __name__ == "__main__":
    main()
