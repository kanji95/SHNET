# Comprehensive Multi-Modal Interactions for Referring Image Segmentation (Findings of ACL 2022)

Official PyTorch implementation | [Paper](https://openreview.net/forum?id=mZ_MHVvOwry)

## Abstract
> We investigate Referring Image Segmentation (RIS), which outputs a segmentation map corresponding to the natural language description. Addressing RIS efficiently requires considering the interactions happening across visual and linguistic modalities and the interactions within each modality. Existing methods are limited because they either compute different forms of interactions sequentially (leading to error propagation) or ignore intra-modal interactions. We address this limitation by performing all three interactions simultaneously through a Synchronous Multi-Modal Fusion Module (SFM). Moreover, to produce refined segmentation masks, we propose a novel Hierarchical Cross-Modal Aggregation Module (HCAM), where linguistic features facilitate the exchange of contextual information across the visual hierarchy. We present thorough ablation studies and validate our approach's performance on four benchmark datasets, showing considerable performance gains over the existing state-of-the-art (SOTA) methods.

![](https://user-images.githubusercontent.com/30688360/166107570-5c941733-ba6c-4864-94f5-b41b3c5b1566.jpg)


## Dataset

Prepate [UNC, UNC+, G-Ref, Referit](https://github.com/lichengunc/refer) datasets.

## Dependencies

* Python: 3.8.5
* Pytorch: 1.7.1
* Torchvision: 0.8.2
* pip install git+https://github.com/andfoy/refer.git
* Wandb: 0.12.9 (for visualization)

## Training

    python3 -W ignore main.py --batch_size 32 --num_workers 4 --optimizer AdamW --dataroot <REFERIT_DATA> --lr 1.2e-4 --weight_decay 9e-5 --image_encoder deeplabv3_plus --loss bce --dropout 0.2 --epochs 30 --gamma 0.7 --num_encoder_layers 2 --image_dim 320 --mask_dim 160 --phrase_len 30 --glove_path <GLOVE_PATH> --threshold 0.40 --task referit --feature_dim 20 --transformer_dim 512 --run_name <WANDB_RUN_NAME> --channel_dim 512 --attn_type normal --save


## Citation

If you found our work useful to your research, please consider citing:

    @article{jain2021comprehensive,
      title={Comprehensive Multi-Modal Interactions for Referring Image Segmentation},
      author={Jain, Kanishk and Gandhi, Vineet},
      journal={arXiv preprint arXiv:2104.10412},
      year={2021}
    }
