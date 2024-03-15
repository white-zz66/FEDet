# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Block modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv,CoordConv
from .transformer import TransformerBlock
import torchvision
from .fusion_utils import ConvBatchNorm, DownBlock, UpBlock_attention

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3','MFGM','C3_CoT','C2f_faster','EGCA','FSCM','DeformableProjEmbed','FuseBlock','Shortcut1','FeatureAlign')
class Shortcut1(nn.Module):
    def __init__(self, select = 0):
        super(Shortcut1, self).__init__()
        self.select = select

    def forward(self, x):
        x[0] = x[0][self.select]
        return x[0]+x[1]
class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
    
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

#c2f_faster
class C2f_faster(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

# MFGM
class Shortcut(nn.Module):
    def __init__(self, dimension=0):
        super(Shortcut, self).__init__()
        self.d = dimension

    def forward(self, x):

        return x[0]+x[1]
class PCRC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv2d(2, 640 * 2, kernel_size=1)
        self.R1 = nn.Upsample(None, 2, 'nearest')  # 上采样扩充2倍采用邻近扩充
        self.mcrc = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(640 * 2, 640 * 2, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(640 * 2, 640 * 2, kernel_size=3, padding=1),
        )
        self.acrc = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(640 * 2, 640 * 2, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(640 * 2, 640 * 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1 = self.C1(x)
        x2 = self.mcrc(x1)
        x3 = self.acrc(x1)
        if x.shape[2] == 4:
            return self.R1(x2) + self.R1(x3)
        else:
            x4 = F.interpolate(x2, size=(x.shape[2], x.shape[3]), mode='nearest')
            x5 = F.interpolate(x3, size=(x.shape[2], x.shape[3]), mode='nearest')
            return x4 + x5


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class MFGM(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(MFGM, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)

        # 定义三个扩展卷积层及其对应的步长
        self.dcv1 = nn.Conv2d(c_, c_, 3, dilation=1, padding=1)
        self.dcv2 = nn.Conv2d(c_, c_, 3, dilation=3, padding=3)
        self.dcv3 = nn.Conv2d(c_, c_, 3, dilation=5, padding=5)

        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

        self.C1 = nn.Conv2d(640 + 640, 2, kernel_size=1)
        self.C4 = nn.Conv2d(1, 640, kernel_size=1, stride=1)
        self.pcrc = PCRC2()

        self.shortcut = Shortcut()

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.dcv1(x1)
        x3 = self.dcv2(x1)
        x4 = self.dcv3(x1)

        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, x4), 1)))

        y2 = self.cv2(x)
        # 添加注意力机制  类似于CBAM
        concat = self.C1(torch.cat((y1, y2), 1))
        Conv_1_1 = torch.split(torch.softmax(concat, dim=0), 1, 1)  # 第一维度1为步长进行分割
        Conv_1_2 = torch.split(self.pcrc(concat), 640, 1)  # 第一维度640为步长进行分割
 
        cbam1_s = self.cv6(self.cv5(torch.cat((x1, x2, x3, x4), 1))) * self.C4(Conv_1_1[0])
        cbam1_c = Conv_1_2[0] * self.cv6(self.cv5(torch.cat((x1, x2, x3, x4), 1)))

        y3 = cbam1_s + cbam1_c

        y4 = (self.cv2(x) * self.C4(Conv_1_1[1])) + (Conv_1_2[1] * y2)


        return self.cv7(torch.cat((y3, y4), dim=1))

# class ASFF(nn.Module):
#     def __init__(self, level, c1, rfb=False, vis=False):
#         super(ASFF, self).__init__()
#         self.level = level
#         self.c1 = c1
#         # 特征金字塔从上到下三层的channel数
#         # 对应特征图大小(以640*640输入为例)分别为20*20, 40*40, 80*80
#         # self.dim = [512, 256, 128]
#         # 特征金字塔从上到下四层的channel数
#         # self.dim = [1280, 960, 640, 320]
#         self.dim = c1
#         self.inter_dim = c1
#         if level == 0:  # 特征图最小的一层，channel数1280
#             self.stride_level_1 = add_conv(self.dim, self.inter_dim, 3, 1)
#             self.stride_level_2 = add_conv(self.dim, self.inter_dim, 3, 1)
#             self.stride_level_3 = add_conv(self.dim, self.inter_dim, 3, 1)
#             self.expand = add_conv(self.inter_dim, self.inter_dim, 3, 1)
#         elif level == 1:  # 特征图大小适中的一层，channel数960
#             self.compress_level_0 = add_conv(self.dim, self.inter_dim, 1, 1)
#             self.stride_level_2 = add_conv(self.dim, self.inter_dim, 3, 1)
#             self.stride_level_3 = add_conv(self.dim, self.inter_dim, 3, 1)
#             self.expand = add_conv(self.inter_dim, self.dim, 3, 1)
#         elif level == 2:  # 特征图次最大的一层，channel数640
#             self.compress_level_0 = add_conv(self.dim, self.inter_dim, 1, 1)
#             self.compress_level_1 = add_conv(self.dim, self.inter_dim, 1, 1)
#             self.stride_level_3 = add_conv(self.dim, self.inter_dim, 3, 1)
#             self.expand = add_conv(self.inter_dim, self.dim, 3, 1)
#         elif level == 3:  # 特征图最大的一层，channel数320
#             self.compress_level_0 = add_conv(self.dim, self.inter_dim, 1, 1)
#             self.compress_level_1 = add_conv(self.dim, self.inter_dim, 1, 1)
#             self.compress_level_2 = add_conv(self.dim, self.inter_dim, 1, 1)
#             self.expand = add_conv(self.inter_dim, self.dim, 3, 1)
#
#         compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory
#
#         self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_3 = add_conv(self.inter_dim, compress_c, 1, 1)
#
#         self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)
#         self.vis = vis
#
#     # def forward(self, x_level_0, x_level_1, x_level_2, x_level_3):
#     def forward(self, x):
#         self.dim = x[0].size()[1]
#         if self.level == 0:
#             level_0_resized = x[0]
#             level_1_resized = self.stride_level_1(x[1])
#             level_2_downsampled_inter = F.max_pool2d(x[2], 3, stride=1, padding=1)
#             level_2_resized = self.stride_level_2(level_2_downsampled_inter)
#             # level_2_resized = self.stride_level_2(x_level_2)
#             level_3_downsampled_inter = F.max_pool2d(x[3], 3, stride=1, padding=1)
#             level_3_resized = self.stride_level_3(level_3_downsampled_inter)
#             # level_3_resized = self.stride_level_3(x_level_3)
#
#         elif self.level == 1:
#             # level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized = self.compress_level_0(x[0])
#             # level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
#             level_1_resized = x[1]
#             level_2_resized = self.stride_level_2(x[2])
#             level_3_downsampled_inter = F.max_pool2d(x[3], 3, stride=1, padding=1)
#             level_3_resized = self.stride_level_3(level_3_downsampled_inter)
#
#         elif self.level == 2:
#             # level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized = self.compress_level_0(x[0])
#             # level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
#             level_1_compressed = self.compress_level_1(x[1])
#             level_1_resized = F.interpolate(level_1_compressed, scale_factor=1, mode='nearest')
#             level_2_resized = x[2]
#             level_3_resized = self.stride_level_3(x[3])
#
#         elif self.level == 3:
#             # level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized = self.compress_level_0(x[0])
#             # level_0_resized = F.interpolate(level_0_compressed, scale_factor=8, mode='nearest')
#             # level_1_compressed = self.compress_level_1(x_level_1)
#             level_1_resized = self.compress_level_1(x[1])
#             # level_1_resized = F.interpolate(level_1_compressed, scale_factor=4, mode='nearest')
#             level_2_compressed = self.compress_level_2(x[2])
#             level_2_resized = F.interpolate(level_2_compressed, scale_factor=1, mode='nearest')
#             level_3_resized = x[3]
#
#         level_0_weight_v = self.weight_level_0(level_0_resized)
#         level_1_weight_v = self.weight_level_1(level_1_resized)
#         level_2_weight_v = self.weight_level_2(level_2_resized)
#         level_3_weight_v = self.weight_level_3(level_3_resized)
#         levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
#         levels_weight = self.weight_levels(levels_weight_v)
#         levels_weight = F.softmax(levels_weight, dim=1)
#
#         fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
#                             level_1_resized * levels_weight[:, 1:2, :, :] + \
#                             level_2_resized * levels_weight[:, 2:3, :, :] + \
#                             level_3_resized * levels_weight[:, 3:, :, :]
#
#         out = self.expand(fused_out_reduced)
#
#         if self.vis:
#             return out, levels_weight, fused_out_reduced.sum(dim=1)
#         else:
#             return out


# 扩展卷积
class DilatedConv(nn.Module):
    def __init__(self, c1=64, c2=64, k=3, p=1, dilation=1):
        super(DilatedConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, padding=p, dilation=dilation)
        self.bn = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
    
#CCSPNet
class CoTAttention(nn.Module):

    def __init__(self, dim=512,kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)

        return k1+k2

class Bottleneck_CoT(nn.Module):
    # Standard bottleneck
    # COTNet
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        # self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.cv2 = CoTAttention(dim=c_, kernel_size=3)
        self.cv3 = Conv(c_, c2, 1, 1)
        # self.add = shortcut and c1 == c2

    def forward(self, x):
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        # x1 = self.cv1(x)
        # x1 = self.cv2(x1)
        # x1 = self.cv3(x1)
        x1 = self.cv1(x) + x
        x2 = self.cv2(x1)
        x3 = self.cv3(x2) + x2

        return x3 + x

class C3_CoT(nn.Module):
    # CSP Bottleneck with 3 convolutions
    # COTNet
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_CoT(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

#fasterblock
from timm.models.layers import DropPath

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class Faster_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x
 
# SPFNet里面的EGCA
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), group=1, bn_act=False,
                 bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.act = nn.PReLU(out_channels)
        self.act = nn.SiLU(out_channels)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)

    return x

class EGCA(nn.Module):
    def __init__(self, init_channel, in_channels: int, groups=2) -> None:
        super(EGCA, self).__init__()
        print(init_channel,in_channels)
        self.groups = groups
        self.conv_dws1 = conv_block(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0,
                                    group=in_channels // 2, bn_act=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pw1 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.softmax = nn.Softmax(dim=1)

        self.conv_dws2 = conv_block(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0,
                                    group=in_channels // 2,
                                    bn_act=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pw2 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=False)

        self.branch3 = nn.Sequential(
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, group=in_channels, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True))
        self.conv = Conv(init_channel, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x0, x1 = x.chunk(2, dim=1)
        out1 = self.conv_dws1(x0)
        out1 = self.maxpool1(out1)
        out1 = self.conv_pw1(out1)

        out2 = self.conv_dws1(x1)
        out2 = self.maxpool1(out2)
        out2 = self.conv_pw1(out2)

        out = torch.add(out1, out2)

        b, c, h, w = out.size()
        out = self.softmax(out.view(b, c, -1))
        out = out.view(b, c, h, w)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)
        out = torch.add(out, x)
        out = channel_shuffle(out, groups=self.groups)

        br3 = self.branch3(x)

        output = br3 + out

        return output
    

#FaPN FAM
import warnings
import fvcore.nn.weight_init as weight_init
from .ops_dcnv3.modules.dcnv3 import DCNv3 as dcnv3

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            # Dynamo doesn't support context managers yet
            is_dynamo_compiling = False
            if not is_dynamo_compiling:
                with warnings.catch_warnings(record=True):
                    if x.numel() == 0 and self.training:
                        # https://github.com/pytorch/pytorch/issues/12013
                        assert not isinstance(
                            self.norm, torch.nn.SyncBatchNorm
                        ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class FSCM(nn.Module):  # FaPN full version
    def __init__(self, c1,out_nc=128, norm=None):
        super(FSCM, self).__init__()
        # print('out_nc',out_nc)
        # self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        # self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dcpack_L2 = dcnv3(out_nc,  kernel_size=3, stride=1, pad=1, dilation=1, group=8).to(self.device)
        self.relu = nn.ReLU(inplace=True)
        # weight_init.c2_xavier_fill(self.offset)
        self.conv = Conv(out_nc*2,out_nc,k=1,s=1)
        self.conv1 = Conv(out_nc//2,out_nc,k=1,s=1)
        self.conv2 = Conv(out_nc,out_nc,k=3,s=2)
        self.conv3 = Conv(out_nc * 2, out_nc, k=1, s=1)
        self.SGE = SpatialGroupEnhance(64)

    def forward(self, x):
        # x[0] = feat_l,x[1] = feat_s
        x[0] = self.SGE(x[0])
        x[1] = self.SGE(x[1])
        HW = x[0].size()[2:]
        if x[1].shape[1] > x[0].shape[1]:
            x[1] = self.conv(x[1])
        elif x[1].shape[1] < x[0].shape[1]:
            x[1] = self.conv1(x[1])
        if x[0].size()[2:] > x[1].size()[2:]:
            feat_up = F.interpolate(x[1], HW, mode='bilinear', align_corners=False)
        elif x[0].size()[2:] < x[1].size()[2:]:
            feat_up = self.conv2(x[1])
        else:
            feat_up = x[1]


        feat_up = feat_up.permute(0,2,3,1)
        x[0] = x[0].permute(0, 2, 3, 1)

        offset_aid = self.conv3((torch.cat([feat_up,x[0]],dim=3).permute(0,3,1,2)))

        dcnv3 = self.dcpack_L2(feat_up,offset_aid)

        feat_align = self.relu(dcnv3)  # [feat, offset]
        feat_align = feat_align.permute(0,3,1,2)
        x[0]=x[0].permute(0,3,1,2)

        return feat_align + x[0]
   
class FeatureAlign(nn.Module):  # FaPN full version
    def __init__(self, c1,c2,out_nc=128, norm=None):
        super(FeatureAlign, self).__init__()
        print('out_nc',out_nc)
        # self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        # self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dcpack_L2 = dcnv3(out_nc,  kernel_size=3, stride=1, pad=1, dilation=1, group=8).to(self.device)
        self.relu = nn.ReLU(inplace=True)
        # weight_init.c2_xavier_fill(self.offset)
        self.conv = Conv(out_nc*2,out_nc,k=1,s=1)
        self.conv1 = Conv(out_nc//2,out_nc,k=1,s=1)
        self.conv2 = Conv(out_nc,out_nc,k=3,s=2)
        self.concat = Concat()

    def forward(self, x):
        # x[0] = feat_l,x[1] = feat_m ,x[2]=feat_s
        HW = x[0].size()[2:]
        if x[1].shape[1] > x[0].shape[1]:
            x[1] = self.conv(x[1])
        elif x[1].shape[1] < x[0].shape[1]:
            x[1] = self.conv1(x[1])
        if x[0].size()[2:] > x[1].size()[2:]:
            feat_up1 = F.interpolate(x[1], HW, mode='bilinear', align_corners=False)
        elif x[0].size()[2:] < x[1].size()[2:]:
            feat_up1 = self.conv2(x[1])
        else:
            feat_up1 = x[1]

        if x[2].shape[1] > x[0].shape[1]:
            x[2] = self.conv(x[2])
        elif x[2].shape[1] < x[0].shape[1]:
            x[2] = self.conv1(x[2])
        if x[0].size()[2:] > x[2].size()[2:]:
            feat_up2 = F.interpolate(x[2], HW, mode='bilinear', align_corners=False)
        elif x[0].size()[2:] < x[2].size()[2:]:
            feat_up2 = self.conv2(x[2])
        else:
            feat_up2 = x[2]

        feat_up1 = feat_up1.permute(0, 2, 3, 1)
        dcnv3_1 = self.dcpack_L2(feat_up1)
        # feat_up = feat_up.permute(0,2,3,1)
        # dcnv3 = self.dcpack_L2(feat_up)
        feat_up2 = feat_up2.permute(0, 2, 3, 1)
        dcnv3_2 = self.dcpack_L2(feat_up2)

        feat_align_1 = self.relu(dcnv3_1)  # [feat, offset]
        feat_align_1 = feat_align_1.permute(0,3,1,2)

        feat_align_2 = self.relu(dcnv3_2)  # [feat, offset]
        feat_align_2 = feat_align_2.permute(0, 3, 1, 2)

        return self.concat([feat_align_1,x[0],feat_align_2])


 #DMLP
import faulthandler
faulthandler.enable()

class DWConv2d(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, padding=1, bias=False):
        super(DWConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_chans, in_chans, kernel_size=kernel_size,
                                   padding=padding, groups=in_chans, bias=bias)
        self.pointwise = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=bias)

        nn.init.kaiming_uniform_(self.depthwise.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class GroupNorm(nn.GroupNorm):
    """
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class DWConvSeq(nn.Module):
    def __init__(self, dim=768):
        super(DWConvSeq, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=nn.Sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = nn.Sigmoid()
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class SEMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False, use_se=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.dwconv = DWConvSeq(hidden_features)
        self.gamma = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.se = SqueezeExcite(out_features, se_ratio=0.25) if use_se else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        # import pdb; pdb.set_trace()
        x = self.drop(self.gamma * self.dwconv(x, H, W)) + x
        x = self.fc2(x)
        x = self.drop(x)
        x = self.se(x.permute(0, 2, 1).reshape(B, C, H, W)).reshape(B, C, N).permute(0, 2, 1)
        return x

class DeformableProjEmbed(nn.Module):
    """ feature map to Projected Embedding
    """
    def __init__(self, in_chans=512, emb_chans=128):
        super().__init__()
        self.kernel_size = kernel_size = 3
        self.stride = stride = 1
        self.padding = padding = 1
        self.proj = nn.Conv2d(in_chans, emb_chans, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.offset_conv = nn.Conv2d(in_chans, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.modulator_conv = nn.Conv2d(in_chans, 1 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        self.norm = nn.BatchNorm2d(emb_chans)
        self.act = nn.GELU()

    def deform_proj(self, x):
        # h, w = x.shape[2:]
        max_offset = min(x.shape[-2], x.shape[-1]) // 4
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.proj.weight,
                                          bias=self.proj.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x

    def forward(self, x):
        x = self.deform_proj(x)
        x = self.act(self.norm(x))
        return x

    
#fusionnet
class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class FuseBlock(nn.Module):
    def __init__(self, base_channels):
        super(FuseBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(base_channels)
        self.norm2 = nn.BatchNorm2d(base_channels * 2)
        self.norm3 = nn.BatchNorm2d(base_channels * 2)

        self.up2 = UpFuseBlock(base_channels=base_channels * 2)
        self.up1 = UpFuseBlock(base_channels=base_channels)

        self.down1 = DownFuseBlock(base_channels=base_channels)
        self.down2 = DownFuseBlock(base_channels=base_channels * 2)

    def forward(self, x):
        """
        x = [fp1,fp2,fp3]  x[0]=fp1,x[1]=fp2,x[2]=fp3
        Args:
            fp1 (torch.Tensor): (B, C, H, W)
            fp2 : (B, C * 2, H // 2, W // 2)
            fp3: (B, C * 4, H // 4, W // 4)
            fp4: (B, C * 8, H //8, W // 8)

        """
        fp3 = self.norm3(x[2])
        fp2 = self.norm2(x[1])
        fp1 = self.norm1(x[0])

        # downsample fuse phase
        fp2 = self.down1(fp1, fp2)
        fp3 = self.down2(fp2, fp3)

        # upsample fuse phase
        fp2 = self.up2(fp2, fp3)
        fp1 = self.up1(fp1, fp2)

        return fp1, fp2, fp3


def reshape_downsample(x):
    '''using reshape method to do downsample

    Args:
        -x (torch.Tensor): (B, C, H, W)
    Return
        -ret (torch.Tensor): (B, C * 4, H // 2, W // 2)
    '''
    b, c, h, w = x.shape
    ret = torch.zeros_like(x)
    ret = ret.reshape(b, c * 4, h // 2, -1)
    ret[:, 0::4, :, :] = x[:, :, 0::2, 0::2]
    ret[:, 1::4, :, :] = x[:, :, 0::2, 1::2]
    ret[:, 2::4, :, :] = x[:, :, 1::2, 0::2]
    ret[:, 3::4, :, :] = x[:, :, 1::2, 1::2]

    return ret


def reshape_upsample(x):
    '''using reshape to do upsample
    '''
    b, c, h, w = x.shape
    assert c % 4 == 0, 'number of channels must be multiple of 4'
    ret = torch.zeros_like(x)
    ret = ret.reshape(b, c // 4, h * 2, w * 2)
    ret[:, :, 0::2, 0::2] = x[:, 0::4, :, :]
    ret[:, :, 0::2, 1::2] = x[:, 1::4, :, :]
    ret[:, :, 1::2, 0::2] = x[:, 2::4, :, :]
    ret[:, :, 1::2, 1::2] = x[:, 3::4, :, :]

    return ret


class DownFuseBlock(nn.Module):
    def __init__(self, base_channels, dropout_rate=0.1):
        super(DownFuseBlock, self).__init__()
        self.eca = ECA(base_channels * 2)
        self.down = reshape_downsample

        # we use group conv here since the reshape downsample split original feature map into
        # 4 pieces and group them in channel dimention. We want each conv group to have 4 channels
        # whicn contains exactly all informations in original HxW feature map
        self.conv1 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1, groups=base_channels)
        self.conv2 = nn.Conv2d(base_channels * 4, base_channels , 3, 1, 1, groups=base_channels)
        self.norm1 = nn.BatchNorm2d(base_channels * 2)
        self.norm2 = nn.BatchNorm2d(base_channels)

        self.fuse_conv = ConvBatchNorm(base_channels * 2, base_channels * 2)
        self.fuse_conv2 = ConvBatchNorm(base_channels , base_channels )
        self.relu = nn.ReLU()

    def forward(self, fp1, fp2):
        '''
            Args:
                -fp1: (B, C1, H1, W1)
                -fp2: (B, C1 * 2, H1 //2, W1 // 2)
        '''
        down = self.down(fp1)
        if(fp1.shape[1]==fp2.shape[1]):
            down =self.conv2(down)
            down = self.relu(self.norm2(down))
            fp2 = self.fuse_conv2(fp2 * 0.75 + down * 0.25) + fp2
        else:
            down = self.conv1(down)
            down = self.relu(self.norm1(down))
            fp2 = self.fuse_conv(fp2 * 0.75 + down * 0.25) + fp2



        fp2 = self.eca(fp2)

        return fp2



class UpFuseBlock(nn.Module):
    def __init__(self, base_channels, dropout_rate=0.1):
        super(UpFuseBlock, self).__init__()
        self.eca = ECA(base_channels)
        self.up = reshape_upsample

        self.conv1 = nn.Conv2d(base_channels // 4, base_channels, kernel_size=3, stride=1, padding=1,groups=base_channels // 4)
        self.conv2 = nn.Conv2d(base_channels //2, base_channels, kernel_size=3, stride=1, padding=1,groups=base_channels //2)
        self.norm1 = nn.BatchNorm2d(base_channels)
        self.norm2 = nn.BatchNorm2d(base_channels)

        self.relu = nn.ReLU()
        self.fuse_conv = ConvBatchNorm(base_channels, base_channels)

    def forward(self, fp1, fp2):
        '''
            Args:
                -fp1: (B, C1, H1, W1)
                -fp2: (B, C1 * 2, H1 //2, W1 // 2)
        '''


        up = self.up(fp2)
        if fp1.shape[1] == fp2.shape[1]:
            up = self.conv1(up)
            up = self.relu(self.norm1(up))

            fp1 = self.fuse_conv((fp1 * 0.75 + up * 0.25)) + fp1
        else:
            up = self.conv2(up)
            up = self.relu(self.norm2(up))

            fp1 = self.fuse_conv((fp1 * 0.75 + up * 0.25)) + fp1
        fp1 = self.eca(fp1)

        return fp1
    
    
#SGA  
class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        return x