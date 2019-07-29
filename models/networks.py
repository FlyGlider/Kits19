import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import functools
from torch.optim import lr_scheduler
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from torchvision.models.resnet import resnet101

###############################################################################
# Helper Functions
###############################################################################

def get_conv_layer(in_ch, out_ch, kernel_size, use_bias, mode='2D'):
    if kernel_size not in [1, 3]:
        raise NotImplementedError('kernel size must be 1 or 3!')
    if mode == '2D':
        if kernel_size == 3:
            conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1, bias=use_bias)
        else:
            conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=0, bias=use_bias)
    elif mode in ['3D', '3D_LOW']:
        if kernel_size == 3:
            conv_layer = nn.Conv3d(in_ch, out_ch, kernel_size, stride=1, padding=1, bias=use_bias)
        else:
            conv_layer = nn.Conv3d(in_ch, out_ch, kernel_size, stride=1, padding=0, bias=use_bias)
    else:
        raise NotImplementedError('mode [%s] is not found' % mode)
    return conv_layer

def get_norm_layer(in_ch, norm_type='instance', mode='2D'):
    if mode == '2D':
        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d(num_features=in_ch, affine=True)
            use_bias = False
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(num_features=in_ch, affine=False, track_running_stats=False)
            use_bias = True
        elif norm_type == 'none':
            norm_layer = None
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    elif mode in ['3D', '3D_LOW']:
        if norm_type == 'batch':
            norm_layer = nn.BatchNorm3d(num_features=in_ch, affine=True)
            use_bias = False
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm3d(num_features=in_ch, affine=False, track_running_stats=False)
            use_bias = True
        elif norm_type == 'none':
            norm_layer = None
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    else:
        raise NotImplementedError('mode [%s] is not found' % mode)
    return norm_layer, use_bias

def get_nonlin_layer(nonlin_type='relu'):
    if nonlin_type == 'relu':
        relu = nn.ReLU(inplace=True)
    elif nonlin_type == 'leaky':
        relu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
    elif nonlin_type == 'None':
        relu = None
    else:
        raise NotImplementedError('non_linear layer [%s] is not found' % nonlin_type)
    return relu

def get_pool_layer(pool_type='max', mode='2D'):
    if mode == '2D':
        if pool_type == 'max':
            pool = nn.MaxPool2d
        elif pool_type == 'avg':
            pool = nn.AvgPool2d
        else:
            raise NotImplementedError('[%s] pool layer  is not found' % pool_type)
    elif mode in ['3D', '3D_LOW']:
        if pool_type == 'max':
            pool = nn.MaxPool3d
        elif pool_type == 'avg':
            pool = nn.AvgPool3d
        else:
            raise NotImplementedError('[%s] pool layer  is not found' % pool_type)
    else:
        raise NotImplementedError('mode [%s] is not found' % mode)
    return pool


def get_scheduler(optimizer, args):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + args.epoch_count - args.niter) / float(args.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_iters, gamma=0.1)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler



class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real and (seg is None):
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=True):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class CBR(nn.Module):
    def __init__(self, in_ch, out_ch,
                 norm_type='batch',
                 nonlin_type='relu', mode='2D'):
        super(CBR, self).__init__()
        if mode not in ['2D', '3D', '3D_LOW']:
            raise NotImplementedError('mode [%s] is not recognized' % mode)
        self.bn, use_bias = get_norm_layer(out_ch, norm_type, mode=mode)
        self.relu = get_nonlin_layer(nonlin_type)
        self.conv = get_conv_layer(in_ch, out_ch, 3, use_bias, mode=mode)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class StackedConvLayers(nn.Module):
    def __init__(self, in_ch, out_ch, num_convs,
                 norm_type='batch',
                 nonlin_type='relu', mode='2D'):
        super(StackedConvLayers, self).__init__()
        self.out_ch = out_ch
        if mode not in ['2D', '3D', '3D_LOW']:
            raise NotImplementedError('mode [%s] is not recognized' % mode)
        self.blocks = nn.Sequential(
            CBR(in_ch, out_ch, norm_type, nonlin_type, mode),
            *[CBR(out_ch, out_ch, norm_type, nonlin_type, mode) for _ in range(num_convs - 1)]
        )
    
    def forward(self, x):
        x = self.blocks(x)
        return x
        

class ResidualBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, downsample=None, norm_type='batch', nonlin_type='relu', mode='2D'):
        super(ResidualBasicBlock, self).__init__()
        if mode not in ['2D', '3D', '3D_LOW']:
            raise NotImplementedError('mode [%s] is not recognized' % mode)
        expansion = 1
        self.out_ch = out_ch
        self.downsample = downsample
        self.cbr = CBR(in_ch, out_ch, norm_type, nonlin_type, mode)
        self.bn, use_bias = get_norm_layer(out_ch, norm_type, mode)
        self.relu = get_nonlin_layer(nonlin_type)
        self.conv = get_conv_layer(out_ch, out_ch, 3, use_bias, mode=mode)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        x = self.cbr(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(residual + x)
        return x

class StackedResidualBlocks(nn.Module):
    def __init__(self, block, in_ch, out_ch, blocks=1, norm_type='batch', nonlin_type='relu', mode='2D'):
        super(StackedResidualBlocks, self).__init__()
        if mode not in ['2D', '3D', '3D_LOW']:
            raise NotImplementedError('mode [%s] is not recognized' % mode)
        self.out_ch = out_ch
        self.bn, use_bias = get_norm_layer(out_ch, norm_type, mode=mode)
        downsample = None
        
        if in_ch != out_ch * block.expansion:
            downsample = nn.Sequential(
                get_conv_layer(in_ch, out_ch * block.expansion, 1, use_bias, mode=mode),
                get_norm_layer(out_ch * block.expansion, norm_type, mode)[0])
        layers = []
        layers.append(block(in_ch, out_ch, downsample, norm_type, nonlin_type, mode))
        in_ch = out_ch * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_ch, out_ch))

        self. blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.blocks(x)
        return x

class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_ch, out_ch=512, norm_type='batch', sizes=(1, 2, 3, 6), mode='3D'):
        super(PSPModule, self).__init__()
        if mode not in ['2D', '3D', '3D_LOW']:
            raise NotImplementedError('mode [%s] is not recognized' % mode)
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_ch, out_ch // len(sizes), norm_type, size, mode) for size in sizes])
        self.bottleneck = nn.Sequential(
            get_conv_layer(in_ch + out_ch, out_ch, kernel_size=3, use_bias=False, mode=mode),
            get_norm_layer(out_ch, norm_type, mode)[0],
            # nn.Dropout2d(0.1)
        )
        if mode == '2D':
            self.upsample_mode = 'bilinear'
        else:
            self.upsample_mode = 'trilinear'

    def _make_stage(self, in_ch, out_ch, norm_type, size, mode):
        if mode == '2D':
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
            conv = get_conv_layer(in_ch, out_ch, kernel_size=1, use_bias=False, mode=mode)
            bn = get_norm_layer(out_ch, norm_type, mode)[0]
        else:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
            conv = get_conv_layer(in_ch, out_ch, kernel_size=1, use_bias=False, mode=mode)
            bn = get_norm_layer(out_ch, norm_type, mode)[0]
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        shape = feats.size()[2:]
        priors = [F.interpolate(input=stage(feats), size=shape, mode=self.upsample_mode, align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle
    
class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, in_ch, inner_ch=256, out_ch=512, norm_type='batch', dilations=(2, 4, 6), mode='3D'):
        super(ASPPModule, self).__init__()
        if mode not in ['2D', '3D', '3D_LOW']:
            raise NotImplementedError('mode [%s] is not recognized' % mode)

        if mode == '2D':
            self.upsample_mode = 'bilinear'
            self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Conv2d(in_ch, inner_ch, kernel_size=1, padding=0, dilation=1, bias=False),
                                       get_norm_layer(inner_ch, norm_type, mode)[0])
            self.conv2 = nn.Sequential(nn.Conv2d(in_ch, inner_ch, kernel_size=1, padding=0, dilation=1, bias=False),
                                       get_norm_layer(inner_ch, norm_type, mode)[0])
            self.conv3 = nn.Sequential(nn.Conv2d(in_ch, inner_ch, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                       get_norm_layer(inner_ch, norm_type, mode)[0])
            self.conv4 = nn.Sequential(nn.Conv2d(in_ch, inner_ch, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                       get_norm_layer(inner_ch, norm_type, mode)[0])
            self.conv5 = nn.Sequential(nn.Conv2d(in_ch, inner_ch, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                       get_norm_layer(inner_ch, norm_type, mode)[0])
            self.bottleneck = nn.Sequential(
                nn.Conv2d(inner_ch * 5, out_ch, kernel_size=1, padding=0, dilation=1, bias=False),
                get_norm_layer(out_ch, norm_type, mode)[0],
            )
        else:
            self.upsample_mode = 'trilinear'
            self.conv1 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                       nn.Conv3d(in_ch, inner_ch, kernel_size=1, padding=0, dilation=1, bias=False),
                                       get_norm_layer(inner_ch, norm_type, mode)[0])
            self.conv2 = nn.Sequential(nn.Conv3d(in_ch, inner_ch, kernel_size=1, padding=0, dilation=1, bias=False),
                                       get_norm_layer(inner_ch, norm_type, mode)[0])
            self.conv3 = nn.Sequential(nn.Conv3d(in_ch, inner_ch, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                       get_norm_layer(inner_ch, norm_type, mode)[0])
            self.conv4 = nn.Sequential(nn.Conv3d(in_ch, inner_ch, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                       get_norm_layer(inner_ch, norm_type, mode)[0])
            self.conv5 = nn.Sequential(nn.Conv3d(in_ch, inner_ch, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                       get_norm_layer(inner_ch, norm_type, mode)[0])
            self.bottleneck = nn.Sequential(
                nn.Conv3d(inner_ch * 5, out_ch, kernel_size=1, padding=0, dilation=1, bias=False),
                get_norm_layer(out_ch, norm_type, mode)[0],
            )
        

        
        
    def forward(self, x):
        shape = x.size()[2:]

        feat1 = F.interpolate(self.conv1(x), size=shape, mode=self.upsample_mode, align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle
    
class Bottleneck(nn.Module):
    expansion = 4
    # in_planes = 256 planes = 64，输出通道数是planes的4倍

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super.__init__(Bottleneck, self)
        self.conv1 = nn.Conv3d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

