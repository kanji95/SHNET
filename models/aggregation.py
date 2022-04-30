import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HCAM(nn.Module):
    def __init__(self, channel_dim, num_regions, dropout):
        super().__init__()

        self.num_regions = num_regions

        self.conv_1 = nn.Sequential(
            nn.Conv2d(channel_dim * 2, channel_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_dim),
            nn.Dropout2d(dropout),
        )
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channel_dim * 2, channel_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_dim),
            nn.Dropout2d(dropout),
        )
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(channel_dim * 2, channel_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_dim),
            nn.Dropout2d(dropout),
        )

        self.conv_3d = nn.Sequential(
            nn.Conv3d(channel_dim, 512, 3, padding=(0, 1, 1)), nn.BatchNorm3d(512)
        )

    def forward(self, level_features, H, W, phrase_mask):

        B, N, C, _ = level_features.shape

        visual_features = level_features[:, :, :, : self.num_regions]
        textual_features = level_features[:, :, :, self.num_regions :]

        masked_sum = textual_features * phrase_mask[:, None, None, :]
        textual_features = masked_sum.sum(dim=-1) / phrase_mask[:, None, None, :].sum(
            dim=-1
        )

        v1, v2, v3 = visual_features.view(B, N, C, H, W).unbind(dim=1)
        l1, l2, l3 = textual_features.unbind(dim=1)

        v12 = self.conv_1(
            torch.cat([v1, l2[:, :, None, None].expand(B, C, H, W)], dim=1)
        ).sigmoid()
        v13 = self.conv_1(
            torch.cat([v1, l3[:, :, None, None].expand(B, C, H, W)], dim=1)
        ).sigmoid()

        v21 = self.conv_2(
            torch.cat([v2, l1[:, :, None, None].expand(B, C, H, W)], dim=1)
        ).sigmoid()
        v23 = self.conv_2(
            torch.cat([v2, l3[:, :, None, None].expand(B, C, H, W)], dim=1)
        ).sigmoid()

        v31 = self.conv_3(
            torch.cat([v3, l1[:, :, None, None].expand(B, C, H, W)], dim=1)
        ).sigmoid()
        v32 = self.conv_3(
            torch.cat([v3, l2[:, :, None, None].expand(B, C, H, W)], dim=1)
        ).sigmoid()

        bv1, bv2, bv3 = v1.clone(), v2.clone(), v3.clone()
        v1 = v1 + v12 * bv2 + v13 * bv3
        v2 = v2 + v21 * bv1 + v23 * bv3
        v3 = v3 + v31 * bv1 + v32 * bv2

        v = torch.stack([v1, v2, v3], dim=2)

        fused_feature = self.conv_3d(v).squeeze(2)

        return fused_feature

class MGATE(nn.Module):
    def __init__(self, channel_dim=512, alpha=0.5):
        super().__init__()

        self.gate_cell1 = GATE_Cell(channel_dim, alpha)
        self.gate_cell2 = GATE_Cell(channel_dim, alpha)
        self.gate_cell3 = GATE_Cell(channel_dim, alpha)

    def forward(self, x1, x2, x3):
        x1_out = self.gate_cell1(x1, x2, x3)
        x2_out = self.gate_cell2(x2, x3, x1)
        x3_out = self.gate_cell3(x3, x1, x2)
       
        out = x1_out + x2_out + x3_out
        return out
      

class GATE_Cell(nn.Module):
    def __init__(self, channel_dim=512, alpha=0.5):
        super().__init__()

        self.chunk_size = channel_dim
        self.conv = nn.Conv2d(channel_dim, channel_dim*3, kernel_size=3, stride=1, padding=1)
        self.alpha = torch.tensor(alpha, requires_grad=True)
    
    def forward(self, x1, x2, x3):
        y = self.conv(x1)
        i, f, r = torch.split(y, self.chunk_size, dim=1)
        f, r = torch.sigmoid(f + 1), torch.sigmoid(r + 1)
        c = self.alpha*f*x2 + (1 - self.alpha)*f*x3 + (1 - f)*i
        out = r * torch.tanh(c) + (1 - r)*x1

        return out
    
class TGFE(nn.Module):
    def __init__(self, channel_dim=512):
        super().__init__()
        
        self.channel_dim = channel_dim
        
        self.visual_conv = nn.Conv2d(in_channels=channel_dim, out_channels=channel_dim, kernel_size=1, stride=1, padding=0)
        self.lang_conv = nn.Conv2d(in_channels=channel_dim, out_channels=channel_dim, kernel_size=1, stride=1, padding=0)
        self.joint_conv = nn.Conv2d(in_channels=channel_dim * 2, out_channels=channel_dim, kernel_size=1, stride=1, padding=0)
        
        self.visual_trans = nn.Conv2d(in_channels=channel_dim, out_channels=channel_dim, kernel_size=1, stride=1, padding=0)
        self.lang_trans = nn.Conv2d(in_channels=channel_dim, out_channels=channel_dim, kernel_size=1, stride=1, padding=0) 
        
    def lang_se(self, visual_feat, lang_feat):
        """[summary]

        Args:
            visual_feat : B x C x H x W
            lang_feat   : B x C x 1 x 1
        """
        lang_feat_trans = self.lang_trans(lang_feat)
        lang_feat_trans = F.sigmoid(lang_feat_trans)
        
        visual_feat_trans = self.visual_trans(visual_feat)
        visual_feat_trans = F.relu(visual_feat_trans)
        
        feat_trans = visual_feat_trans * lang_feat_trans
        return feat_trans
        
    def global_vec(self, visual_feat, lang_feat):
        """[summary]

        Args:
            visual_feat : B x C x H x W
            lang_feat   : B x C x 1 x 1
        """
        
        visual_feat_key = F.relu(self.visual_conv(visual_feat))
        visual_feat_key = rearrange(visual_feat_key, "b c h w -> b (h w) c")
        
        lang_feat_query = F.relu(self.lang_conv(lang_feat))
        lang_feat_query = rearrange(lang_feat_query, "b c h w -> b c (h w)")
        
        attn_map = torch.matmul(visual_feat_key, lang_feat_query)
        # attn_map = torch.divide(attn_map, self.channel_dim**0.5)
        attn_map = F.softmax(attn_map, dim=1) # B x HW x 1
        
        visual_feat_reshape = rearrange(visual_feat, "b c h w -> b (h w) c") 
        
        gv_pooled = torch.matmul(attn_map.transpose(1, 2), visual_feat_reshape) # B x 1 x C
        gv_pooled = rearrange(gv_pooled, "b x c -> b c x 1")
        
        gv_lang = torch.cat([gv_pooled, lang_feat], dim=1)
        gv_lang = F.relu(self.joint_conv(gv_lang)) # B x C x 1 x 1
        gv_lang = F.normalize(gv_lang, p=2, dim=1)
        
        return gv_lang
    
    def gated_exchange(self, feat1, feat2, feat3, lang_feat):
        
        gv_lang = self.global_vec(feat1, lang_feat)
        feat2 = self.lang_se(feat2, gv_lang)
        feat3 = self.lang_se(feat3, gv_lang)
        
        feat_exg = feat1 + feat2 + feat3
        
        return feat_exg
        
    def forward(self, feat1, feat2, feat3, lang_feat):
        
        feat_exg1 = self.gated_exchange(feat1, feat2, feat3, lang_feat)
        feat_exg1 = F.normalize(feat_exg1, p=2, dim=1)
        
        feat_exg2 = self.gated_exchange(feat2, feat3, feat1, lang_feat)
        feat_exg2 = F.normalize(feat_exg2, p=2, dim=1)
        
        feat_exg3 = self.gated_exchange(feat3, feat1, feat2, lang_feat)
        feat_exg3 = F.normalize(feat_exg3, p=2, dim=1)
        
        feat_exg1_2 = self.gated_exchange(feat_exg1, feat_exg2, feat_exg3, lang_feat)
        feat_exg1_2 = F.normalize(feat_exg1_2, p=2, dim=1)
        
        feat_exg2_2 = self.gated_exchange(feat_exg2, feat_exg3, feat_exg1, lang_feat)
        feat_exg2_2 = F.normalize(feat_exg2_2, p=2, dim=1)
        
        feat_exg3_2 = self.gated_exchange(feat_exg3, feat_exg1, feat_exg2, lang_feat)
        feat_exg3_2 = F.normalize(feat_exg3_2, p=2, dim=1)
        
        fused_feature =  feat_exg1_2 + feat_exg2_2 + feat_exg3_2
        
        return fused_feature

class GBFM(nn.Module):
    def __init__(self, channel_dim, num_regions):
        super().__init__()
        
        self.num_regions = num_regions

        self.mm_conv = nn.Sequential(
            nn.Conv2d(channel_dim * 2, channel_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        
        self.feature_conv = nn.Sequential(
            nn.Conv2d(channel_dim * 2, channel_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim, kernel_size=3, stride=1, padding=1),
        )
    
    def get_gate(self, x1, x2):
        
        c_12 = torch.cat([x1, x2], dim=1)
        c_12 = self.feature_conv(c_12)
        gate = self.gate_conv(c_12)
        out = gate * x1 + x2
        return out
        
    
    def forward(self, level_features, H, W, phrase_mask):
        
        B, N, C, _ = level_features.shape

        visual_features = level_features[:, :, :, : self.num_regions]
        textual_features = level_features[:, :, :, self.num_regions :]

        masked_sum = textual_features * phrase_mask[:, None, None, :]
        textual_features = masked_sum.sum(dim=-1) / phrase_mask[:, None, None, :].sum(
            dim=-1
        )

        v1, v2, v3 = visual_features.view(B, N, C, H, W).unbind(dim=1)
        l1, l2, l3 = textual_features.unbind(dim=1)
        
        x1 = self.mm_conv(torch.cat([v1, l1[:, :, None, None].expand(B, C, H, W)], dim=1))
        x2 = self.mm_conv(torch.cat([v2, l2[:, :, None, None].expand(B, C, H, W)], dim=1))
        x3 = self.mm_conv(torch.cat([v3, l3[:, :, None, None].expand(B, C, H, W)], dim=1))
        
        l_12 = self.get_gate(x2, x1)
        l_23 = self.get_gate(x3, x2)
        left = self.get_gate(l_23, l_12)
        
        r_12 = self.get_gate(x1, x2)
        r_23 = self.get_gate(x2, x3)
        right = self.get_gate(r_12, r_23)
        
        out = left + right
        out = self.out_conv(out)
        
        return out
        
        
