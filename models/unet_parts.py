# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2   BN =>ReLU =>conv'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, use_linear=True):
        super(up, self).__init__()
        self.use_linear = use_linear
        #这里加了个激活函数
        if use_linear:
            self.up = nn.Sequential([
                nn.Conv3d(in_ch,out_ch,1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            ])
        else:
           self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        if self.use_linear:
            x1 = F.interpolate(x1, scale_factor=2, mode='trilinear', align_corners=True)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

# -------------------------------------2d版本---------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
class double_conv_2d(nn.Module):
    '''
    2d版本
    (conv => BN => ReLU) * 2   BN =>ReLU =>conv
    '''
    def __init__(self, in_ch, out_ch):
        super(double_conv_2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv_2d(nn.Module):
    '''2d版本'''
    def __init__(self, in_ch, out_ch):
        super(inconv_2d, self).__init__()
        self.conv = double_conv_2d(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down_2d(nn.Module):
    '''2d版本'''
    def __init__(self, in_ch, out_ch):
        super(down_2d, self).__init__()
        self.mpconv = nn.Sequential(
            nn.AvgPool2d(2),
            double_conv_2d(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up_2d(nn.Module):
    '''2d版本'''
    def __init__(self, in_ch, out_ch, use_linear=True):
        super(up_2d, self).__init__()
        self.use_linear = use_linear
        #这里加了个激活函数
        if use_bilinear:
            self.up = nn.Sequential([
                nn.Conv2d(in_ch,out_ch,1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.conv = double_conv_2d(in_ch, out_ch)

    def forward(self, x1, x2):
        if self.use_linear:
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv_2d(nn.Module):
    '''2d版本'''
    def __init__(self, in_ch, out_ch):
        super(outconv_2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x