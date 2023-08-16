import torch
from torch import nn
from torch.nn import functional as F

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels):
        super(SeparableConvBlock, self).__init__()

        self.depthwise_conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)#groups=in_channels
        self.depthwise_conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm3d(in_channels)
        self.relu = nn.GELU()

    def forward(self, x):
        x1 = self.depthwise_conv1(x) 
        y = self.depthwise_conv2(x1)
        y = self.norm(y)
        y = self.relu(y)
        y = self.pointwise_conv(y)
        y = y + x

        return y


class MFS(nn.Module):
    def __init__(self, num_channels, is_last=False):
        super(MFS, self).__init__()
        self.num_channels = num_channels
        self.is_last = is_last
        self.epsilon = 1e-4

        self.layer_scale = nn.Parameter(1e-2 * torch.ones((num_channels*3)), requires_grad=True)

        self.convdown = SeparableConvBlock(num_channels*3)

        self.convup = SeparableConvBlock(num_channels*3)

        self.p_downsample = nn.Sequential(nn.MaxPool3d(2, 2),nn.GELU())
    

        ## 简易版注意力
        self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.GELU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.GELU()

        
        self.p1_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p1_w2_relu = nn.GELU()
        self.p2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p2_w2_relu = nn.GELU()
        self.p3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.GELU()
        self.p4_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.GELU()


    def forward(self, inputs):
        """ MFS模块结构示意图
            P1_0 -------------------------> P1_out -------->
               | ------------|                ↑
                             ↓                |
            P2_0 ---------> P2_1 ---------> P2_out -------->    32*32*32
               |-------------|--------------↑ ↑
                             ↓                |
            P3_0 ---------> P3_1 ---------> P3_out -------->    16*16*16 
               |-------------|                ↑
                             |--------------↓ |
            P4_0 -------------------------> P4_out -------->    8*8*8   
        """
        p1_0, p2_0, p3_0, p4_0 = inputs

        # p3_td = p3_0 + self.Upsample()(p4_0)
        # p3_1 = p3_td + self.convup(p3_td)

        # p2_td = p2_0 + self.Upsample()(p3_1)
        # p2_1 = p2_td + self.convup(p2_td)

        # p1_in = p1_0 + self.Upsample()(p2_1)
        # p1_out = p1_in + self.convup(p1_in)

        # p2_in = p2_0 + p2_1 + self.p_downsample(p1_out)
        # p2_out = p2_in + self.convdown(p2_in)

        # p3_in = p3_0 + p3_1 + self.p_downsample(p2_out)
        # p3_out = p3_in + self.convdown(p3_in)

        # p4_in = p4_0 + self.p_downsample(p3_out)
        # p4_out = p4_in + self.convdown(p4_in)

        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        p2_1 = self.convdown(weight[0] * p2_0 + weight[1] * self.p_downsample(p1_0))

        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_1 = self.convdown(weight[0] * p3_0 + weight[1] * self.p_downsample(p2_1))

        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.convdown(weight[0] * p4_0 + weight[1] * self.p_downsample(p3_1))

        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        p3_out = self.convup(weight[0] * p3_0 + weight[1] * p3_1 + weight[2] * self.Upsample()(p4_out))

        p2_w2 = self.p2_w2_relu(self.p2_w2)
        weight = p2_w2 / (torch.sum(p2_w2, dim=0) + self.epsilon)
        p2_out = self.convup(weight[0] * p2_0 + weight[1] * p2_1 + weight[2] * self.Upsample()(p3_out))

        p1_w2 = self.p1_w2_relu(self.p1_w2)
        weight = p1_w2 / (torch.sum(p1_w2, dim=0) + self.epsilon)
        p1_out = self.convup(weight[0] * p1_0 + weight[1] * self.Upsample()(p2_out))

        if self.is_last:
            return p1_out, p2_out, p3_out
        return p1_out, p2_out, p3_out, p4_out


    @staticmethod
    def Upsample(scale_factor=2, mode='trilinear'):
        upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        return upsample