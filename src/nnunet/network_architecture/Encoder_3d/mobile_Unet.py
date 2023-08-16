from typing import Dict
import torch.nn as nn
from network_architecture.Encoder_3d.model_components import MobileViTBlock, FMCA
from network_architecture.Encoder_3d.model_config import get_config


class InitConv(nn.Module):
    def __init__(self, in_channel=4, out_channel=16, dropout=0.1):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y


class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        super(Encoder, self).__init__()

        model_config = get_config("xx_small")

        self.InitConv = nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.EnBlock1_1 = FMCA(in_channels=base_channels, stride=1, expand_ratio=1)

        self.channel2 = nn.Conv3d(base_channels, base_channels*2, 1)
        self.EnDown1_1 = nn.Sequential(nn.MaxPool3d(2, 2),nn.GELU())
        self.EnBlock2_1 = FMCA(in_channels=base_channels*2, stride=1, expand_ratio=1)      

        self.channel3 = nn.Conv3d(base_channels*2, base_channels*4, 1)
        self.EnDown2_1 = nn.Sequential(nn.MaxPool3d(2, 2),nn.GELU())
        self.EnBlock3_2 = self._make_MobileViT_layer(input_channel=base_channels*4, cfg=model_config["layer4"])

        self.channel4 = nn.Conv3d(base_channels*4, base_channels*8, 1)
        self.EnDown3_1 = nn.Sequential(nn.MaxPool3d(2, 2),nn.GELU())
        self.EnBlock4_2 = self._make_MobileViT_layer(input_channel=base_channels*8, cfg=model_config["layer5"])


    @staticmethod
    def _make_MobileViT_layer(input_channel: int, cfg: Dict):
        block = []

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            patch_d=cfg.get("patch_d", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.Sequential(*block)


    def forward(self, x):
        x1 = self.InitConv(x)
        x1_0 = self.EnBlock1_1(x1)

        x1_0_1 = self.channel2(x1_0)
        x1_2_1 = self.EnDown1_1(x1_0_1)
        x2_0 = self.EnBlock2_1(x1_2_1)

        x2_0_1 = self.channel3(x2_0)
        x2_2_1 = self.EnDown2_1(x2_0_1)
        x3_0 = self.EnBlock3_2(x2_2_1)

        x3_0_1 = self.channel4(x3_0)
        x3_2_1 = self.EnDown3_1(x3_0_1)
        output = self.EnBlock4_2(x3_2_1)

        return x1_0,x2_0,x3_0,output

if __name__=="__main__":
    Encoder(in_channels=1, base_channels=16)
