from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from training.activation.swish import Swish

from network_architecture.Encoder_3d.transformer import TransformerEncoder


def make_divisible(v: Union[float, int], divisor = 8, min_value = None,) -> Union[float, int]:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.conv0_1 = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.conv1_1 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)

        self.conv3 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)

        attn_1 = self.conv1_1(attn)

        attn = attn + attn_1 + attn_0

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU() #####
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class FMCA(nn.Module):

    def __init__(self, in_channels: int, stride: int, expand_ratio: Union[int, float], skip_connection: Optional[bool] = True) -> None:
        assert stride in [1, 2]
        super().__init__()

        layer_scale_init_value = 1e-2
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)
        self.attn = SpatialAttention(in_channels)
        self.norm = nn.BatchNorm3d(in_channels)

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels,in_channels,3,1,1),
            nn.BatchNorm3d(in_channels),
            nn.GELU()
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        
        x1 = self.layer_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm(x)) #
        x2 = self.conv(x)
        y = x1+x2

        return y


class ConvLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
            int((kernel_size[2] - 1) / 2)
        )

        block = nn.Sequential()

        conv_layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding, bias=bias)

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer =  nn.BatchNorm3d(num_features=out_channels)
            block.add_module(name="norm", module=norm_layer)

        if use_act:
            act_layer = Swish()  #####
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MobileViTBlock(nn.Module):

    def __init__(
        self,
        in_channels: 64,
        transformer_dim: 80,
        ffn_dim: 160,
        n_transformer_blocks: int = 1,
        head_dim: int = 20,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        ffn_dropout: float = 0.0,
        patch_h: int = 2,
        patch_w: int = 2,
        patch_d: int = 2,
        conv_ksize: Optional[int] = 3,
        *args,
        **kwargs
    ) -> None:
        super().__init__()


        conv_3x3x3_in = ConvLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=conv_ksize, stride=1)
        conv_1x1x1_in = ConvLayer(in_channels=in_channels, out_channels=transformer_dim, kernel_size=1, stride=1, use_norm=False, use_act=False)

        conv_1x1x1_out = ConvLayer(in_channels=transformer_dim, out_channels=in_channels, kernel_size=1, stride=1)
        conv_3x3x3_out = ConvLayer(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=conv_ksize, stride=1)

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3x3", module=conv_3x3x3_in)
        self.local_rep.add_module(name="conv_1x1x1", module=conv_1x1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1x1_out
        self.fusion = conv_3x3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_d = patch_d
        self.patch_area = self.patch_w * self.patch_h * self.patch_d

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h, patch_d= self.patch_w, self.patch_h, self.patch_d
        patch_area = patch_w * patch_h * patch_d
        batch_size, in_channels, orig_h, orig_w, orig_d = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
        new_d = int(math.ceil(orig_d / self.patch_d) * self.patch_d)

        interpolate = False
        if new_w != orig_w or new_h != orig_h or new_d != orig_d:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w, new_d), mode="trilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patch_d = new_d // patch_d  # n_d
        num_patches = num_patch_h * num_patch_w * num_patch_d # Nw_d

        # [B, C, H, W, D] -> [B * C * n_h, p_h, n_w, p_w, n_d, p_d]
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w, num_patch_d, patch_d)
        # [B * C * n_h, p_h, n_w, p_w, n_d, p_d] -> [B * C * n_h, n_w, n_d, p_h, p_w, p_d]
        x = x.permute(0, 2, 4, 1, 3, 5)
        # [B * C * n_h, n_w, n_d, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w * p_d and N = n_h * n_w * n_d
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w, orig_d),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
            "num_patches_d": num_patch_d,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]
        num_patch_d = info_dict["num_patches_d"]

        # [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)
        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, num_patch_d, self.patch_h, self.patch_w, self.patch_d)
        # [B*C*n_h, n_w, n_d, p_h, p_w, p_d] -> [B*C*n_h, p_h, n_w, p_w, n_d, p_d]
        x = x.permute(0, 3, 1, 4, 2, 5)
        # [B*C*n_h, p_h, n_w, p_w, p_d] -> [B, C, H, W, D]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w, num_patch_d * self.patch_d)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="trilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        fm = self.folding(x=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm


if __name__=="__main__":
    from model_config import get_config

    model_config = get_config("xx_small")
