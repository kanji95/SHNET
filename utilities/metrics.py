import torch


@torch.no_grad()
def compute_mask_IOU(masks, target, thresh=0.3):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = (masks > thresh) * target
    intersection = temp.sum()
    union = (((masks > thresh) + target) - temp).sum()
    return intersection, union


@torch.no_grad()
def compute_batch_IOU(masks, target, thresh=0.3):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = (masks > thresh) * target
    intersection = torch.sum(temp.flatten(1), dim=-1, keepdim=True)
    union = torch.sum(
        (((masks > thresh) + target) - temp).flatten(1), dim=-1, keepdim=True
    )
    return intersection, union
