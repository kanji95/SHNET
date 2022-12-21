import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskDecoder(nn.Module):
    def __init__(
        self,
        channel_dim=512,
        dropout=0.2,
        scale_factor=2,
    ):
        super().__init__()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        self.deconv1 = nn.Sequential(
            nn.Conv2d(
                channel_dim,
                512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            self.upsample,
        )

        self.deconv2 = nn.Sequential(
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            self.upsample,
        )
        
        self.deconv3 = nn.Sequential(
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # self.upsample,
        )

        self.mask_deconv = nn.Sequential(
            nn.Conv2d(
                512,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(
                256, 1, kernel_size=3, stride=1, padding=1, bias=True
            ),
        )

    def forward(self, x, backbone_features):
        
        # import pdb; pdb.set_trace()
        # v2, v3, v4 = backbone_features
        
        # x = torch.cat([x, v4], dim=1)
        x = self.deconv1(x)
        
        # x = torch.cat([x, F.interpolate(v3, scale_factor=2)], dim=1)
        x = self.deconv2(x)
        
        # x = torch.cat([x, F.interpolate(v2, scale_factor=4)], dim=1)
        x = self.deconv3(x)
        
        mask = self.mask_deconv(x)

        return mask

class ConvUpsample(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        out_channels=1,
        channels=[512, 256, 128, 64],
        upsample=[True, True, False, False],
        scale_factor=2,
        drop=0.2,
    ):
        super().__init__()

        linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        assert len(channels) == len(upsample)

        modules = []

        for i in range(len(channels)):

            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ),
                    nn.BatchNorm2d(channels[i]),
                    nn.ReLU(),
                    nn.Dropout2d(drop),
                )
            )

            if upsample[i]:
                modules.append(linear_upsampling)

            in_channels = channels[i]

        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
            )
        )

        self.deconv = nn.Sequential(*modules)

    def forward(self, x):
        return self.deconv(x)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
