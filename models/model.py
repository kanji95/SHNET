from time import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# from allennlp.modules.elmo import Elmo, batch_to_ids

from .position_encoding import *
from .transformer import TransformerEncoder, TransformerEncoderLayer, _get_clones
from .fusion import *
from .mask_decoder import *

from .attention import *


class JointModel(nn.Module):
    def __init__(
        self,
        args,
        transformer_dim=2048,
        out_channels=512,
        stride=1,
        num_layers=1,
        num_encoder_layers=1,
        dropout=0.2,
        normalize_before=True,
        mask_dim=112,
    ):
        super(JointModel, self).__init__()

        self.mask_dim = mask_dim

        self.text_encoder = TextEncoder(hidden_size=out_channels, num_layers=num_layers)

        feature_dim = args.feature_dim
        # self.pool = nn.AdaptiveMaxPool2d((feature_dim*2, feature_dim*2))
        self.pool = nn.UpsamplingBilinear2d(size=(feature_dim, feature_dim))

        self.conv_3x3 = nn.ModuleDict(
            {
                "layer2": nn.Sequential(
                    nn.Conv2d(
                        512, out_channels, kernel_size=3, stride=stride[0], padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(inplace=True),
                ),
                "layer3": nn.Sequential(
                    nn.Conv2d(
                        1024, out_channels, kernel_size=3, stride=stride[1], padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(inplace=True),
                ),
                "layer4": nn.Sequential(
                    nn.Conv2d(
                        2048, out_channels, kernel_size=3, stride=stride[2], padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(inplace=True),
                ),
            }
        )

        ############### JRM ###################
        encoder_layer = TransformerEncoderLayer(
            args,
            out_channels,
            nhead=8,
            dim_feedforward=transformer_dim,
            dropout=dropout,
            normalize_before=normalize_before,
        )
        encoder_norm = nn.LayerNorm(out_channels) if normalize_before else None
        self.transformer_encoder = _get_clones(
            TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm), 3
        )
        # self.transformer_encoder_2_gram = TransformerEncoder(
        #     encoder_layer, num_encoder_layers, encoder_norm
        # )
        # self.transformer_encoder_3_gram= TransformerEncoder(
        #     encoder_layer, num_encoder_layers, encoder_norm
        # )

        self.cmmlf = CMMLF(out_channels, feature_dim * feature_dim, dropout)
        # self.gbfm = GBFM(out_channels, feature_dim*feature_dim)

        self.aspp_decoder = ASPP(
            in_channels=out_channels,
            atrous_rates=[6, 12, 24],
            out_channels=out_channels,
        )

        self.mask_decoder = MaskDecoder(channel_dim=out_channels, dropout=dropout)

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
            # layer_output = self.conv_3x3[key](image[key])
            layer_output = self.conv_3x3[key](self.pool(image[key]))
            image_features.append(layer_output)

        B, C, H, W = layer_output.shape

        # phrase_2_gram = self.get_N_gram_feature(phrase, N=2)
        # phrase_3_gram = self.get_N_gram_feature(phrase, N=3)

        f_text = self.text_encoder(phrase)
        f_text = f_text.permute(1, 0, 2)

        # f_text_2_gram = self.text_encoder(phrase_2_gram)
        # f_text_2_gram = f_text_2_gram.permute(1, 0, 2)

        # f_text_3_gram = self.text_encoder(phrase_3_gram)
        # f_text_3_gram = f_text_3_gram.permute(1, 0, 2)

        # gram_text_features = [f_text, f_text_2_gram, f_text_3_gram]

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

            f_img = f_img.flatten(2).permute(2, 0, 1)  ## HW, B, C

            # joint_enc = torch.cat([f_img, gram_text_features[i]], dim=0)
            joint_enc = torch.cat([f_img, f_text], dim=0)

            enc_out = self.transformer_encoder[i](
                joint_enc,
                pos=joint_pos_embed,
                src_key_padding_mask=joint_key_padding_mask,
            )
            enc_out = rearrange(enc_out, "l b c -> b c l")

            joint_features.append(enc_out)

        level_features = torch.stack(joint_features, dim=1)

        fused_feature = self.cmmlf(level_features, H, W, phrase_mask)
        # fused_feature = self.activation(self.gbfm(level_features, H, W, phrase_mask))

        x = self.aspp_decoder(fused_feature)
        x = self.upsample(self.mask_decoder(x, image_features)).squeeze(1)
        # x = self.upsample(self.conv_upsample(x)).squeeze(1)

        return x

    def get_N_gram_feature(self, phrase, N=2):
        padded_N_gram_phrase = F.pad(
            phrase, pad=(0, 0, N - 1, N - 1), mode="constant", value=0
        )
        phrase_N_gram = torch.zeros_like(phrase)
        for i in range(N - 1, padded_N_gram_phrase.shape[1] - N + 1):
            window_feat = padded_N_gram_phrase[:, i - N + 1 : i + N, :]
            anchor = padded_N_gram_phrase[:, i]
            anchor = torch.stack([anchor] * window_feat.shape[1], dim=1)
            weight = F.cosine_similarity(anchor, window_feat, dim=2)
            phrase_N_gram[:, i - N + 1, :] = (weight[:, :, None] * window_feat).sum(
                dim=1
            )

        return phrase_N_gram


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
