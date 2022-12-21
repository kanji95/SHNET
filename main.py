import wandb

import argparse
import gc
import os
import random
import traceback
from datetime import datetime
from time import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import *
from torchvision.models._utils import IntermediateLayerGetter

from models.modeling.deeplab import *

from models.deeplabv2 import DeepLabV2

from models.net.deeplabv3plus import deeplabv3plus
# from models.net.config import cfg

from dataloader.referit_loader import *

from evaluate import evaluate
from losses import Loss
from models.model import JointModel
from train import train
from utilities.utils import print_

plt.rcParams["figure.figsize"] = (15, 15)

torch.manual_seed(12)
torch.cuda.manual_seed(12)
random.seed(12)
np.random.seed(12)

torch.backends.cudnn.benchmark = True

torch.cuda.empty_cache()

def get_args_parser():
    parser = argparse.ArgumentParser("Refering Image Segmentation", add_help=False)

    # HYPER Params
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--gamma", default=0.7, type=float)
    parser.add_argument("--optimizer", default="AdamW", choices=["AdamW", "Adam", "SGD"], type=str)
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--grad_check", default=False, action="store_true")

    ## DCRF
    parser.add_argument("--dcrf", default=False, action="store_true")

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
            "deeplab_coco",
            "dino",
        ],
    )
    parser.add_argument("--attn_type", type=str, default="normal", choices=["normal", "linear", "cross", "multimodal", "context", "linear_context"])
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--transformer_dim", default=256, type=int)
    parser.add_argument("--feature_dim", default=14, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)

    parser.add_argument("--model_dir", type=str, default="./saved_model")
    parser.add_argument("--save", default=False, action="store_true")

    ## Evalute??
    parser.add_argument("--model_filename", default="model_unc.pth", type=str)

    # LOSS Params
    parser.add_argument("--loss", default="bce", type=str)

    parser.add_argument("--run_name", default="", type=str)

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
    parser.add_argument("--cache_type", type=str, default="full")
    parser.add_argument("--image_dim", type=int, default=448)
    parser.add_argument("--mask_dim", type=int, default=56)
    parser.add_argument("--channel_dim", type=int, default=512)
    parser.add_argument("--phrase_len", type=int, default=20)

    parser.add_argument("--threshold", type=float, default=0.40)

    return parser


def main(args):

    experiment = wandb.init(project="referring_image_segmentation", config=args)
    if args.run_name == "":
        print_("No Name Provided, Using Default Run Name")
        args.run_name = f"{experiment.id}"
    args.run_name = f'{args.task}_{args.run_name}_{experiment.id}'
    print_(f"METHOD USED FOR CURRENT RUN {args.run_name}")
    experiment.name = args.run_name
    # wandb.run.save()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print_(f"{device} being used with {n_gpu} GPUs!!")

    ####################### Model Initialization #######################

    return_layers = {"layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}

    if args.image_encoder == "resnet50" or args.image_encoder == "resnet101":
        stride = [1, 1, 1]
        model = torch.hub.load(
            "pytorch/vision:v0.6.1", args.image_encoder, pretrained=True
        )
        image_encoder = IntermediateLayerGetter(model, return_layers)
    elif args.image_encoder == "deeplabv2":
        stride = [1, 1, 1]
        model = DeepLabV2(n_classes=181, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
        
        state_dict = torch.load("./models/deeplabv1_resnet101-coco.pth", map_location=lambda storage, loc: storage)

        model.load_state_dict(state_dict, strict=False)

        return_layers = {"layer3": "layer2", "layer4": "layer3", "layer5": "layer4"}
        image_encoder = IntermediateLayerGetter(model, return_layers)
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
    # elif args.image_encoder == "deeplab_coco":
    #     # import pdb; pdb.set_trace()
    #     stride = 2
    #     model = deeplabv3plus(cfg)
    #     checkpoint = torch.load("./models/deeplabv3_plus_res101_atrous_coco_updated.pth")
    #     checkpoint = {k.replace("module.", ""): v for k,v in checkpoint.items()}
    #     model.load_state_dict(checkpoint)
    #     image_encoder = IntermediateLayerGetter(model.backbone, return_layers)
    elif args.image_encoder == "dino":
        stride = [1, 1, 1]
        resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', force_reload=True)
        image_encoder = IntermediateLayerGetter(resnet50, return_layers)
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

    wandb.watch(joint_model, log="all")

    total_parameters = 0
    for name, child in joint_model.named_children():
        num_params = sum([p.numel() for p in child.parameters() if p.requires_grad])
        if num_params > 0:
            print_(f"No. of params in {name}: {num_params}")
            total_parameters += num_params

    print_(f"Total number of params: {total_parameters}")

    if n_gpu > 1:
        image_encoder = nn.DataParallel(image_encoder)
        joint_model = nn.DataParallel(joint_model)

    joint_model.to(device)
    image_encoder.to(device)

    params = list([p for p in joint_model.parameters() if p.requires_grad])
    
    print(f"Using {args.optimizer} optimizer!!")
    cycle_momentum = False
    if args.optimizer == "AdamW":
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        cycle_momentum = True
        optimizer = SGD(params, lr=args.lr, momentum=0.8, weight_decay=args.weight_decay)

    save_path = os.path.join(args.model_dir, args.task)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_filename = os.path.join(
        save_path,
        f'{args.image_encoder}_{args.task}_{datetime.now().strftime("%d_%b_%H-%M")}.pth',
    )

    ######################## Dataset Loading ########################
    print_("Initializing dataset")
    start = time()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((args.image_dim, args.image_dim))
    random_grayscale = transforms.RandomGrayscale(p=0.3)

    train_dataset = ReferDataset(
        data_root=args.dataroot,
        dataset=args.task,
        transform=transforms.Compose(
            [resize, random_grayscale, to_tensor, normalize]
        ),
        annotation_transform=transforms.Compose([ResizeAnnotation(args.mask_dim)]),
        split="train",
        max_query_len=args.phrase_len,
        glove_path=args.glove_path,
    )
    val_dataset = ReferDataset(
        data_root=args.dataroot,
        dataset=args.task,
        transform=transforms.Compose([resize, to_tensor, normalize]),
        annotation_transform=transforms.Compose([ResizeAnnotation(args.mask_dim)]),
        split="val",
        max_query_len=args.phrase_len,
        glove_path=args.glove_path,
    )

    end = time()
    elapsed = end - start
    print_(f"Elapsed time for loading dataset is {elapsed}sec")

    start = time()

    train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    end = time()
    elapsed = end - start
    print_(f"Elapsed time for loading dataloader is {elapsed}sec")

    num_iter = len(train_loader)
    print_(f"training iterations {num_iter}")

    # Learning Rate Scheduler
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, verbose=True)
    ### lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=args.gamma, verbose=True)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.gamma,
        patience=2,
        threshold=1e-3,
        min_lr=1e-6,
        verbose=True,
    )
    
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=num_iter, pct_start=0.2, cycle_momentum=cycle_momentum, div_factor=10, final_div_factor=10)
    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=1.2e-4, step_size_up=num_iter//2, cycle_momentum=cycle_momentum)


    print_(
        f"===================== SAVING MODEL TO FILE {model_filename}! ====================="
    )

    best_acc = 0
    epochs_without_improvement = 0

    for epochId in range(args.epochs):

        train(train_loader, joint_model, image_encoder, 
                optimizer, lr_scheduler, experiment, epochId,
                args)

        val_loss, val_acc = evaluate(val_loader, joint_model, image_encoder,
                                        epochId, args)

        wandb.log({"val_loss": val_loss, "val_IOU": val_acc})

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ExponentialLR) or isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):
            lr_scheduler.step()
        elif isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            
            print_(f"Saving Checkpoint at epoch {epochId}, best validation accuracy is {best_acc}!")
            if args.save:
                torch.save(
                    {
                        "epoch": epochId,
                        "state_dict": joint_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    model_filename,
                )
            epochs_without_improvement = 0
        elif val_acc <= best_acc and epochId != args.epochs - 1:
            epochs_without_improvement += 1
            print_(f"Epochs without Improvement: {epochs_without_improvement}")

            # if epochs_without_improvement % 2 == 0:
            #     current_max_lr = lr_scheduler.max_lrs[0]
            #     print_(f"Reducing Max_lr for scheduler from {current_max_lr:.5f} to {current_max_lr*args.gamma:.5f}")
            #     lr_scheduler.max_lrs[0] = current_max_lr*args.gamma

            if epochs_without_improvement == 6:
                print_(
                    f"{epochs_without_improvement} epochs without improvement, Stopping Training!"
                )
                break         
    
    if args.save:
        print_(f"Current Run Name {args.run_name}")
        best_acc_filename = os.path.join(
            save_path,
            # f"{args.image_encoder}_{args.task}_tl_{args.num_encoder_layers}_td_{args.transformer_dim}_fd_{args.feature_dim}_id_{args.image_dim}_md_{args.mask_dim}_sl_{args.phrase_len}_{best_acc:.5f}.pth",
            f"{args.run_name}_{best_acc:.5f}.pth"
        )
        os.rename(model_filename, best_acc_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Referring Image Segmentation", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    print_(args)

    try:
        main(args)
    except Exception as e:
        traceback.print_exc()
