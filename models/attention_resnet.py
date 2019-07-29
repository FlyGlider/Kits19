import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 对于规格相同的数据,batchnorm的作用不大,但对于规格不同的数据，batchnorm的影响很大


class ResNet(nn.Module):
    def __init__(self, block, layers, nclasses=2, use_linear=True):
        super(ResNet, self).__init__()
        self.inc = inconv(1, 64)
        self.down1 = down(block, layers[0], 64, 128)
        # 在resnet做分类时，这里的stride设置为2，代替了maxpool
        self.down2 = down(block, layers[1], 128, 256)
        self.down3 = down(block, layers[2], 256, 512)
        self.up3 = up(block, 512, 256, use_linear)
        self.up2 = up(block, 256, 128, use_linear)
        self.up1 = up(block, 128, 64, use_linear)
        self.outc = outconv(64, nclasses)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # 使用kaiming初始化效果并不好
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.outc(x)

        return F.log_softmax(x, dim=1)


def conv3x3(in_ch, out_ch, stride=1):
    '''3x3 convolution with padding'''
    return nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch, out_ch):
    return nn.Conv3d(in_ch, out_ch, 1, bias=False)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
        # 一开始就做跳层连接
        # BasicBlock(in_ch,out_ch,downsample=nn.Conv3d(in_ch,out_ch,1))

    def forward(self, x):
        x = self.conv(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)

        return out


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
        self.conv3 = nn.Conv3d(planes, planes*self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes*self.expansion)
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


class down(nn.Module):
    def __init__(self, block, blocks, in_ch, out_ch):
        super(down, self).__init__()
        self.layer = self._make_layer(block, in_ch, out_ch, blocks)
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            self.layer
        )

    def _make_layer(self, block, in_ch, out_ch, blocks=1, stride=1):
        '''blocks是block的数量，以bottelneck为例，第一次输入通道数为64，
        经一个block后，输出通道数为256，所以需要一个downsample将输入转换
        为256通道的，后面输入就是256通道的了，所以就不需要downsample了'''
        downsample = None
        if stride != 1 or in_ch != out_ch*block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch * block.expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch * block.expansion)
            )
        layers = []
        layers.append(block(in_ch, out_ch, stride, downsample))
        in_ch = out_ch * block.expansion
        for i in range(1, blocks):
            layers.append(block(in_ch, out_ch))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.mpconv(x)
        return x

# 此up在长连接上做的是sum操作,换成cat试试
class up(nn.Module):
    def __init__(self, block, in_ch, out_ch, use_linear=True):
        super(up, self).__init__()
        self.use_linear = use_linear
        if use_linear:
            self.up = nn.Sequential(
                # 对调位置看看会怎样
                nn.Conv3d(in_ch, out_ch, 1),
                nn.BatchNorm3d(out_ch)
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch, out_ch, 4, 2, padding=1),
                nn.BatchNorm3d(out_ch)
            )
        # 如果forward阶段的长连接是cat操作,需要用到downsample
        self.conv = block(out_ch, out_ch)  # ,downsample=nn.Sequential(
        # nn.Conv3d(in_ch,out_ch,1)
        # nn.BatchNorm3d(out_ch)
        # ))

    def forward(self, x1, residual):
        if self.use_linear:
            x1 = F.interpolate(x1, scale_factor=2, mode='trilinear', align_corners=True)
        x1 = self.up(x1)
        x = residual + x1
        x = F.relu(x, inplace=True)
        #x = torch.cat([residual,x1],dim=1)
        out = self.conv(x)
        return out


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        # 这里改为卷积3x3试试
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------2d版本------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------


class AttentionResNet2D(nn.Module):
    def __init__(self, block, layers, nclasses=2, use_linear=True):
        super(AttentionResNet2D, self).__init__()
        self.inc = inconv_2d(1, 64)
        self.down1 = down_2d(block, layers[0], 64, 128)
        self.down2 = down_2d(block, layers[1], 128, 256)
        self.down3 = down_2d(block, layers[2], 256, 512)
        self.down4 = down_2d(block, layers[3], 512, 1024)
        self.up4 = up_2d(block, 1024, 512, use_linear)
        self.up3 = up_2d(block, 512, 256, use_linear)
        self.up2 = up_2d(block, 256, 128, use_linear)
        self.up1 = up_2d(block, 128, 64, use_linear)
        self.outc = outconv_2d(64, nclasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.outc(x)

        return F.log_softmax(x, dim=1)


def conv3x3_2d(in_ch, out_ch, stride=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


def conv1x1_2d(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 1, bias=False)


class inconv_2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv_2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # 一开始就做跳层连接
        # BasicBlock(in_ch,out_ch,downsample=nn.Conv2d(in_ch,out_ch,1))

    def forward(self, x):
        x = self.conv(x)
        return x


class BasicBlock_2d(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasicBlock_2d, self).__init__()
        self.conv1 = conv3x3_2d(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_2d(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck_2d(nn.Module):
    expansion = 4
    # in_planes = 256 planes = 64，输出通道数是planes的4倍

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super.__init__(Bottleneck_2d, self)
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
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


class down_2d(nn.Module):
    def __init__(self, block, blocks, in_ch, out_ch):
        super(down_2d, self).__init__()
        self.layer = self._make_layer(block, in_ch, out_ch, blocks)
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            self.layer
        )

    def _make_layer(self, block, in_ch, out_ch, blocks=1, stride=1):
        '''blocks是block的数量，以bottelneck为例，第一次输入通道数为64，
        经一个block后，输出通道数为256，所以需要一个downsample将输入转换
        为256通道的，后面输入就是256通道的了，所以就不需要downsample了'''
        downsample = None
        if stride != 1 or in_ch != out_ch*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * block.expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * block.expansion)
            )
        layers = []
        layers.append(block(in_ch, out_ch, stride, downsample))
        in_ch = out_ch * block.expansion
        for i in range(1, blocks):
            layers.append(block(in_ch, out_ch))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up_2d(nn.Module):
    def __init__(self, block, in_ch, out_ch, use_linear=True):
        super(up_2d, self).__init__()
        self.use_linear = use_linear
        if use_linear:
            self.up = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, padding=1),
                nn.BatchNorm2d(out_ch)
            )
        # 如果forward阶段的长连接是cat操作,需要用到downsample
        self.conv = block(out_ch, out_ch)  # ,downsample=nn.Sequential(
        # nn.Conv2d(in_ch,out_ch,1)
        # nn.BatchNorm2d(out_ch)
        # ))
        self.cab = ChannelAttention(in_planes=in_ch, out_planes=out_ch, reduction=1)

    def forward(self, x1, residual):
        if self.use_linear:
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.up(x1)
        residual = self.cab(residual, x1)
        x = residual + x1
        x = F.relu(x, inplace=True)
        #x = torch.cat([residual,x1],dim=1)

        out = self.conv(x)
        return out


class outconv_2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv_2d, self).__init__()
        # 这里改为卷积3x3试试
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class SELayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.BatchNorm1d(out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, reduction):
        super(ChannelAttention, self).__init__()
        self.channel_attention = SELayer(in_planes, out_planes, reduction)

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], 1)
        channel_attetion = self.channel_attention(fm)
        fm = x1 * channel_attetion
        return fm   

# for i in range(64):
#     plt.imshow(x1[0, i].cpu().numpy())
#     plt.show()
# plt.imshow(np.repeat(channel_attetion[:, :, 0, 0].cpu().numpy(), (512), axis=0))