import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import os
import math
import numpy as np
import functools
from torch.nn import init
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
import torch
import torch.nn as nn
from torchvision import models
from convnext import ConvNeXt
import torch.nn.init as init

class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()

        self.resnet = models.resnet50(pretrained=pretrained)

        self.layer1 = self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool, self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        out_layer1 = self.resnet.layer1(x)  # Stage 1
        out_layer2 = self.resnet.layer2(out_layer1)  # Stage 2
        out_layer3 = self.resnet.layer3(out_layer2)  # Stage 3
        out_layer4 = self.resnet.layer4(out_layer3)  # Stage 4

        return out_layer1, out_layer2, out_layer3, out_layer4





def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 32

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]

## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        # self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.PixelUnshuffle(2))
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        # self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.PixelShuffle(2))
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

    def forward(self, x):
        return self.body(x)

class BasicDownBlock(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super(BasicDownBlock, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(dim, dim, kernel_size,
                      padding=(kernel_size // 2), bias=False),
            nn.InstanceNorm2d(dim, affine=True),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size,
                      padding=(kernel_size // 2), bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = x + self.conv(x)
        return x

class BasicUpBlock(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super(BasicUpBlock, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(dim, dim, kernel_size,
                      padding=(kernel_size // 2), bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size,
                      padding=(kernel_size // 2), bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = x + self.conv(x)
        return x

class Stage(nn.Module):
    def __init__(self,
                 dim=96,
                 num_blocks=[3, 3, 3,3],
                 first=False
                 ):
        super(Stage, self).__init__()
        self.first_channel = 3
        self.first = first

        self.encoder = ConvNeXt()

        self.decoder_4 = nn.Sequential(*[BasicUpBlock(dim=dim * 8) for _ in range(num_blocks[3])])
        self.up_3 = Upsample(dim*8)
        self.reduce_3 = nn.Conv2d(dim*8, dim*4, kernel_size=1, bias=False)
        self.decoder_3 = nn.Sequential(*[BasicUpBlock(dim=dim * 4) for _ in range(num_blocks[2])])
        self.up_2 = Upsample(dim * 4)
        self.reduce_2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=False)
        self.decoder_2 = nn.Sequential(*[BasicUpBlock(dim=dim * 2) for _ in range(num_blocks[1])])
        self.up_1 = Upsample(dim * 2)
        self.reduce_1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.decoder_1 = nn.Sequential(*[BasicUpBlock(dim=dim) for _ in range(num_blocks[0])])

        self.output = nn.Conv2d(dim, 2*16, kernel_size=3, stride=1, padding=1, bias=False)
        self.up = nn.PixelShuffle(4)

        pretrained_weights = torch.load("ddcolor_paper_tiny.pth", map_location="cpu")['params']
        encoder_weights = {k.replace("encoder.arch.", ""): v for k, v in pretrained_weights.items() if
                           k.startswith("encoder.arch.")}
        self.encoder.load_state_dict(encoder_weights, strict=False)

    def forward(self, x):
        N_input = x

        encoder1, encoder2, encoder3, encoder4  = self.encoder(N_input)
        input_decoder_4 = self.decoder_4(encoder4)

        input_decoder_3 = self.up_3(input_decoder_4)
        input_decoder_3 = torch.cat([input_decoder_3, encoder3], 1)
        input_decoder_3 = self.reduce_3(input_decoder_3)
        out_decoder_3 = self.decoder_3(input_decoder_3)

        input_decoder_2 = self.up_2(out_decoder_3)
        input_decoder_2 = torch.cat([input_decoder_2, encoder2], 1)
        input_decoder_2 = self.reduce_2(input_decoder_2)
        out_decoder_2 = self.decoder_2(input_decoder_2)

        input_decoder_1 = self.up_1(out_decoder_2)
        input_decoder_1 = torch.cat([input_decoder_1, encoder1], 1)
        input_decoder_1 = self.reduce_1(input_decoder_1)
        out_decoder_1 = self.decoder_1(input_decoder_1)

        res = self.output(out_decoder_1)
        res = self.up(res)

        return res

class Network(nn.Module):
    def __init__(self,
                 dim=96,
                 num_blocks=[2,2,2,2]
                 ):
        super(Network, self).__init__()
        self.stage = Stage(dim=dim,num_blocks=num_blocks,first=True)

    def forward(self, x):
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(x)

        output = self.stage(input)

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)

        return output


