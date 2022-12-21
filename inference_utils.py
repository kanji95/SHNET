import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torchvision.models._utils import IntermediateLayerGetter

from models.modeling.deeplab import *
from dataloader.referit_loader import *

from PIL import Image
from skimage.transform import resize

from models.model import JointModel

from utilities.im_processing import *

from einops import rearrange


class Args:
    def __init__(
        self,
        lr=3e-4,
        num_workers=4,
        image_encoder="deeplabv3_plus",
        num_layers=1,
        num_encoder_layers=2,
        dropout=0.25,
        skip_conn=False,
        model_path="./saved_model/talk2car/baseline_drop_0.25_bs_64_el_1_sl_40_bce_0.50473.pth",
        dataroot="/ssd_scratch/cvit/kanishk/Talk2Car-RefSeg/",
        glove_path="/ssd_scratch/cvit/kanishk/glove/",
        dataset="referit",
        task="referit",
        split="test",
        phrase_len=30,
        image_dim=320,
        mask_dim=320,
        channel_dim=512,
        transformer_dim=512,
        feature_dim=20,
        attn_type="normal",
    ):
        self.lr = lr
        self.num_workers = num_workers
        self.image_encoder = image_encoder
        self.num_layers = num_layers
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.skip_conn = skip_conn
        self.model_path = model_path
        self.dataroot = dataroot
        self.glove_path = glove_path
        self.dataset = dataset
        self.task = task
        self.split = split
        self.phrase_len = phrase_len
        self.image_dim = image_dim
        self.mask_dim = mask_dim
        self.channel_dim = channel_dim
        self.transformer_dim = transformer_dim
        self.feature_dim = feature_dim
        self.attn_type = attn_type


def compute_mask_IOU(masks, target, thresh=0.3):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = (masks > thresh) * target
    intersection = temp.sum()
    union = (((masks > thresh) + target) - temp).sum()
    return intersection, union


def meanIOU(m, gt, t):
    temp = (m > t) * gt
    inter = temp.sum()
    union = ((m > t) + gt - temp).sum()
    return inter / union


def get_random_sample(val_loader, args):
    data_len = val_loader.dataset.__len__()

    indx = random.choice(range(data_len))
    batch = val_loader.dataset.__getitem__(indx)
    
    orig_image = Image.open(batch["img_path"])
    batch["orig_image"] = np.array(orig_image.resize((args.image_dim, args.image_dim)))
    
    return batch


def display_sample(batch):
    fig = plt.figure(figsize=(10, 10))

    plt.imshow(batch["orig_image"])
    plt.title(batch["orig_phrase"])
    plt.axis("off")

    plt.show()


def prepare_dataloader(args):
    print("Initializing dataset")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        val_dataset, shuffle=True, batch_size=1, num_workers=0, pin_memory=True
    )

    return val_loader

def prepare_network(args, n_gpu, device):
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
    elif args.image_encoder == "dino":
        stride = [1, 1, 1]
        resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', force_reload=True)
        image_encoder = IntermediateLayerGetter(resnet50, return_layers)
    else:
        raise NotImplemented("Model not implemented")

    for param in image_encoder.parameters():
        param.requires_grad_(False)
        
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

    image_encoder.eval();
    joint_model.eval();
    
    return image_encoder, joint_model

def modify_language_command(batch):
    use_original_command = input("Use original command Y/N?: ")
    assert use_original_command in "YNyn"

    original_phrase = True if use_original_command.upper() == "Y" else False

    if not original_phrase:
        new_command = input("Enter New Command: ")
        batch['orig_phrase'] = new_command
    
    return original_phrase

def get_best_threshold(output_mask, gt_mask):    
    iou = []
    thr = []
    cum_sum = []

    t_ = 0.0

    best_t = t_
    best_iou = 0

    while t_ < 1:
        miou = meanIOU(output_mask, gt_mask, t_)
        cum_sum.append((output_mask > t_).sum())
        iou.append(miou)
        thr.append(t_)

        if best_iou < miou:
            best_iou = miou
            best_t = t_

        t_ += 0.05

    if best_t == 0:
        best_t += 0.0001
    return best_t

def run_inference(batch, image_encoder, joint_model, val_loader, args, original_phrase=True):
    
    img = batch["image"].cuda(non_blocking=True).unsqueeze(0)
    
    phrase, phrase_mask = val_loader.dataset.tokenize_phrase(batch["orig_phrase"])
    phrase = phrase.unsqueeze(0).cuda(non_blocking=True)
    phrase_mask = phrase_mask.unsqueeze(0).cuda(non_blocking=True)

    gt_mask = batch["seg_mask"]
    gt_mask = gt_mask.squeeze(dim=1)

    orig_image = batch["orig_image"]
    orig_phrase = batch["orig_phrase"]

    batch_size = img.shape[0]
    img_mask = torch.ones(batch_size, args.feature_dim * args.feature_dim, dtype=torch.int64).cuda(non_blocking=True)

    with torch.no_grad():
        img = image_encoder(img)  

    output_mask = joint_model(img, phrase, img_mask, phrase_mask)

    output_mask = output_mask.detach().cpu().squeeze()
    mask_out = output_mask[0]

    inter, union = compute_mask_IOU(output_mask, gt_mask)
    score = inter / union

    image = batch["orig_image"]
    phrase = batch["orig_phrase"]
    mask_gt = gt_mask
    mask_pred = output_mask

    im = image
    
    if original_phrase:
        best_t = get_best_threshold(output_mask, gt_mask)
    else:
        best_t = output_mask.mean()

    ## Prediction
    im_seg = im[:] / 2
    predicts = (mask_pred > best_t).numpy()
    im_seg[:, :, 0] += predicts.astype('uint8') * 100
    im_seg = im_seg.astype('uint8')

    ## Ground Truth
    im_gt = im[:] / 2
    gt = (mask_gt > 0).numpy()
    im_gt[:, :, 1] += gt.astype('uint8') * 100
    im_gt = im_gt.astype('uint8')

    print(f'Command: {phrase}')
    
    if original_phrase:
        figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))

        axes[0].imshow(im_gt)
        axes[0].set_title("Ground Truth Mask")
        axes[0].axis("off")

        axes[1].imshow(im_seg)
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")
        
    else:
        
        figure = plt.figure(figsize=(20, 20))
        
        plt.imshow(im_seg)
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.show()