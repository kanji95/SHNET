from time import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# from allennlp.modules.elmo import Elmo, batch_to_ids

from .position_encoding import *
from .sync_fusion import *
from .aggregation import *
from .mask_decoder import *

class JointModel(nn.Module):
    def __init__(
        self,
        args,
        sfm_dim=2048,
        out_channels=512,
        stride=1,
        num_layers=1,
        num_sfm_layers=1,
        dropout=0.2,
        normalize_before=True,
        mask_dim=112,
    ):
        super(JointModel, self).__init__()

        self.mask_dim = mask_dim

        self.text_encoder = TextEncoder(num_layers=num_layers)

        feature_dim = args.feature_dim
        self.pool = nn.AdaptiveMaxPool2d((feature_dim*2, feature_dim*2))
        
        self.conv_3x3 = nn.ModuleDict(
            {
                "layer2": nn.Sequential(
                    nn.Conv2d(
                        512, out_channels, kernel_size=3, stride=stride, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                ),
                "layer3": nn.Sequential(
                    nn.Conv2d(
                        1024, out_channels, kernel_size=3, stride=stride, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                ),
                "layer4": nn.Sequential(
                    nn.Conv2d(
                        2048, out_channels, kernel_size=3, stride=stride, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                ),
            }
        )

        ############### JRM ###################
        sfm_layer = SFMLayer(
            args,
            out_channels,
            nhead=8,
            dim_feedforward=sfm_dim,
            dropout=dropout,
            normalize_before=normalize_before,
        )
        sfm_norm = nn.LayerNorm(out_channels) if normalize_before else None
        self.sfm = SFM(
            sfm_layer, num_sfm_layers, sfm_norm
        )
        
        self.hcam = HCAM(out_channels, feature_dim*feature_dim, dropout)
        # self.gbfm = GBFM(out_channels, feature_dim*feature_dim)
        
        self.aspp_decoder = ASPP(
            in_channels=out_channels, atrous_rates=[6, 12, 24], out_channels=512
        )
        
        self.mask_decoder = MaskDecoder(channel_dim=out_channels, dropout=dropout)

        # self.conv_upsample = ConvUpsample(
        #     in_channels=512,
        #     out_channels=1,
        #     channels=[256, 256],
        #     upsample=[True, True],
        #     drop=dropout,
        # )

        self.upsample = nn.Sequential(
            nn.Upsample(
                size=(self.mask_dim, self.mask_dim), mode="bilinear", align_corners=True
            ),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

    def forward(self, image, phrase, img_mask, phrase_mask):

        image_features = []
        for key in self.conv_3x3:
            layer_output = self.activation(self.conv_3x3[key](self.pool(image[key])))
            image_features.append(layer_output)
            
        B, C, H, W = layer_output.shape

        f_text = self.text_encoder(phrase)
        
        f_text = f_text.permute(1, 0, 2)
        L, _, E = f_text.shape
        
        pos_embed_img = positionalencoding2d(B, d_model=C, height=H, width=W)
        pos_embed_img = pos_embed_img.flatten(2).permute(2, 0, 1)

        pos_embed_txt = positionalencoding1d(B, d_model=E, max_len=phrase_mask.shape[1])
        pos_embed_txt = pos_embed_txt.permute(1, 0, 2)

        joint_pos_embed = torch.cat([pos_embed_img, pos_embed_txt], dim=0)

        joint_key_padding_mask = ~torch.cat([img_mask, phrase_mask], dim=1).bool()
        
        joint_features = []
        for i in range(len(image_features)):
            f_img = image_features[i]

            f_img = f_img.flatten(2).permute(2, 0, 1) ## HW, B, C
            
            joint_feat = torch.cat([f_img, f_text], dim=0)

            sfm_out = self.sfm(
                joint_feat, pos=joint_pos_embed, src_key_padding_mask=joint_key_padding_mask
            )
            sfm_out = sfm_out.permute(1, 2, 0)

            joint_features.append(sfm_out)

        level_features = torch.stack(joint_features, dim=1)
        
        fused_feature = self.activation(self.hcam(level_features, H, W, phrase_mask))
        # fused_feature = self.activation(self.gbfm(level_features, H, W, phrase_mask))
        
        x = self.aspp_decoder(fused_feature)
        x = self.upsample(self.mask_decoder(x, image_features)).squeeze(1)
        # x = self.upsample(self.conv_upsample(x)).squeeze(1)

        return x

class TextEncoder(nn.Module):
    def __init__(
        self,
        input_size=300,
        hidden_size=512,
        num_layers=1,
        batch_first=True,
        dropout=0.0,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=batch_first,
            dropout=dropout,
        )

    def forward(self, input):
        self.lstm.flatten_parameters()
        output, (_, _) = self.lstm(input)
        return output
