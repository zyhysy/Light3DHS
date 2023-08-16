from tkinter import Y
from utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
import torch.nn.functional
from network_architecture.initialization import InitWeights_He
from network_architecture.neural_network import SegmentationNetwork
from network_architecture.Encoder_3d.mobile_Unet import Encoder
from training.activation.swish import Swish
from network_architecture.BiFPN import MFS


class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels,conv_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()

        self.conv_kwargs = conv_kwargs
        self.output_channels = output_channels

        self.conv1 = nn.Conv3d(input_channels, output_channels, **self.conv_kwargs)
        self.norm1 = nn.BatchNorm3d(num_features=output_channels)
        self.relu1 = nn.GELU()

        self.conv2 = nn.Conv3d(output_channels, output_channels, **self.conv_kwargs)
        self.norm2 = nn.BatchNorm3d(num_features=output_channels)
        self.relu2 = nn.GELU()



    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        x1 = self.relu1(x1)

        y = self.conv2(x1)
        y = self.norm2(y)
        y = self.relu2(y)
        y = y+x1
        
        return y



class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv2 = Upsample(scale_factor=2, mode='trilinear')
        self.channel = nn.Conv3d(in_channels, out_channels,1)
        self.relu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x, prev):
        x1 = self.conv2(x)
        x1 = self.channel(x1)
        x1 = self.relu(x1) 
        y = torch.cat((x1, prev), dim=1)
        return y


class DeUp8_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp8_Cat, self).__init__()
        self.conv = Upsample(scale_factor=8, mode='trilinear')
        self.channel = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.lrelu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x, prev):
        x1 = self.conv(x)
        x1 = self.lrelu(self.channel(x1))
        y = torch.cat((x1, prev), dim=1)
        return y


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

        
class Light3DHS(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, deep_supervision=True, 
                 upscale_logits=False, final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), 
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 convolutional_pooling=False, convolutional_upsampling=False, seg_output_use_bias=False):
        super(Light3DHS, self).__init__()
        
        self.upscale_logits = upscale_logits  # False

        self.weightInitializer = weightInitializer  # InitWeights_He(1e-2)
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin   # softmax_helper
        self._deep_supervision = deep_supervision  # True
        self.do_ds = deep_supervision  # True


        upsample_mode = 'trilinear'
        self.output_features = base_num_features
        self.input_features = input_channels
        self.num_pool = num_pool

        
        self.encoder = Encoder(self.input_features, self.output_features)

        # now lets build the localization pathway
        self.conv_kwargs = {'kernel_size':3, 'stride': 1, 'dilation': 1, 'padding':1, 'bias': True}

        # self.deup1 = DeUp_Cat(base_num_features*2, base_num_features*1)
        # self.deblock1 = nn.Sequential(ConvDropoutNormNonlin(base_num_features*2, base_num_features, self.conv_kwargs))
        # self.seg_final = nn.Conv3d(base_num_features, num_classes, kernel_size=1, stride=1)

        # RFS
        # self.up = DeUp8_Cat(base_num_features*8, base_num_features)
        # self.conv = nn.Sequential(ConvDropoutNormNonlin(base_num_features*2, base_num_features, self.conv_kwargs),
        #                          ConvDropoutNormNonlin(base_num_features, base_num_features, self.conv_kwargs))
        # self.seg_final = nn.Conv3d(base_num_features, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))

        # PFS
        self.deup3 = DeUp_Cat(base_num_features*8, base_num_features*4)
        self.deblock3 = nn.Sequential(ConvDropoutNormNonlin(base_num_features*8, base_num_features*4, self.conv_kwargs))
        self.seg_outputs3 = nn.Conv3d(base_num_features*4, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))

        self.deup2 = DeUp_Cat(base_num_features*4, base_num_features*2)
        self.deblock2 = nn.Sequential(ConvDropoutNormNonlin(base_num_features*4, base_num_features*2, self.conv_kwargs))
        self.seg_outputs2 = nn.Conv3d(base_num_features*2, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        
        self.deup1 = DeUp_Cat(base_num_features*2, base_num_features*1)
        self.deblock1 = nn.Sequential(ConvDropoutNormNonlin(base_num_features*2, base_num_features, self.conv_kwargs))
        self.seg_final = nn.Conv3d(base_num_features, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))

        # MFS
        # self.mfs = nn.Sequential(MFS(base_num_features, True))

        # self.p1_change_channel = nn.Conv3d(base_num_features, base_num_features*3, kernel_size=1)
        # self.p2_change_channel = nn.Conv3d(base_num_features*2, base_num_features*3, kernel_size=1)
        # self.p3_change_channel = nn.Conv3d(base_num_features*4, base_num_features*3, kernel_size=1)
        # self.p4_change_channel = nn.Conv3d(base_num_features*8, base_num_features*3, kernel_size=1)
        # self.seg_outputs = nn.Conv3d(base_num_features*3, num_classes, kernel_size=1, stride=1)
        

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)


    def forward(self, x):
        skips = []
        seg_outputs = []
        x1_0, x2_0, x3_0, x4_0 = self.encoder(x)
        #x1_0, x2_0= self.unet_encoder(x)

        # y = self.up(x4_0,x1_0)
        # y = self.conv(y)
        # seg_final = self.final_nonlin(self.seg_final(y))
        # seg_outputs=[seg_final]


        #skips.extend([x1_0, x2_0, x3_0])

        
        y3_0 = self.deup3(x4_0, x3_0)
        y3 = self.deblock3(y3_0)
        seg_outputs3 = self.final_nonlin(self.seg_outputs3(y3))

        y2_0 = self.deup2(y3, x2_0)
        y2 = self.deblock2(y2_0)
        seg_outputs2 = self.final_nonlin(self.seg_outputs2(y2))

        y1_0 = self.deup1(y2, x1_0)
        y1 = self.deblock1(y1_0)
        seg_final = self.final_nonlin(self.seg_final(y1))



        # p1 = self.p1_change_channel(x1_0)
        # p2 = self.p2_change_channel(x2_0)
        # p3 = self.p3_change_channel(x3_0)
        # p4 = self.p4_change_channel(x4_0)

        # source = [p1, p2, p3, p4]
        # out_features = self.mfs(source)

        # seg_outputs3 = self.final_nonlin(self.seg_outputs(out_features[2]))
        # seg_outputs2 = self.final_nonlin(self.seg_outputs(out_features[1]))
        # seg_final = self.final_nonlin(self.seg_outputs(out_features[0]))

        seg_outputs=[seg_final, seg_outputs2, seg_outputs3]


        if self._deep_supervision and self.do_ds:
            return seg_outputs
        else:
            return seg_outputs[0]



    # 确保GPU显存目标得到满足
    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)  # [64, 64, 64]
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
