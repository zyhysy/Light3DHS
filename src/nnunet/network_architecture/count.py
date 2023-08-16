import os
import sys
GRANDFA = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) #/home/server1080/Documents/zyh/My_nnUNet/nnunet
sys.path.append(GRANDFA)


import torch
from ptflops import get_model_complexity_info
from torchprofile import profile_macs
from generic_UNet import LightMHS
from initialization import InitWeights_He
from monai.networks.nets import SwinUNETR
from monai.networks.nets import UNETR
from network_architecture.TransBTS.Transformer import TransformerModel

model = LightMHS(1, 16, 2, 3, True, False, lambda x: x, InitWeights_He(1e-2),True, True, True)
#model = LightMHS(1, 16, 4, 3, True, False, lambda x: x, InitWeights_He(1e-2),True, True, True)
#model = TransformerModel( _conv_repr=True, _pe_type="learned")
# model = SwinUNETR(
#         img_size=(128, 128, 128),
#         in_channels=4,
#         out_channels=4,
#         feature_size=48,
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#         dropout_path_rate=0.0,
#         use_checkpoint=False,
#     )
# input = torch.randn(1, 3, 64, 64, 64)

macs,params = get_model_complexity_info(model,(1, 64, 64, 64),as_strings=True,print_per_layer_stat=True)
# params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('params:', params)
print('macs:', macs,'   params:', params)

inputs = torch.randn(1, 1, 64, 64, 64)
macs = profile_macs(model, inputs) / 1e9
print(f'GFLOPs {macs}.')

