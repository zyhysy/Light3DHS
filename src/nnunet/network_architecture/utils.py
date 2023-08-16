import torch.nn as nn

def down_ScalaNet(in_channel, out_channel, size):
    return nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1),
        nn.InstanceNorm3d(out_channel),
        nn.LeakyReLU(),
        nn.Conv3d(out_channel, out_channel, kernel_size=size, stride=size),
        nn.InstanceNorm3d(out_channel),
        nn.LeakyReLU(),
        nn.Conv3d(out_channel, out_channel, kernel_size=1, stride=1),
        nn.InstanceNorm3d(out_channel),
        )

def up_ScalaNet(in_channel, out_channel, size):
    return nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1),
        nn.InstanceNorm3d(out_channel),
        nn.LeakyReLU(),
        nn.ConvTranspose3d(out_channel, out_channel, kernel_size=size, stride=size),
        nn.InstanceNorm3d(out_channel),
        nn.LeakyReLU(),
        nn.Conv3d(out_channel, out_channel, kernel_size=1, stride=1),
        nn.InstanceNorm3d(out_channel),
        )