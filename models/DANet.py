import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import PAM,CAM
from UNet_parts import double_conv,inconv,outconv

class DANet(nn.Module):
    def __init__(self,n_classes=2):
        super(DANet,self).__init__()
        self.inc = inconv(1, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x4 = F.dropout(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return F.log_softmax(x,1)

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv(in_ch, out_ch)
        )
        self.pam = PAM(out_ch)

    def forward(self, x):
        x = self.mpconv(x)
        x = self.pam(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.bilinear = bilinear
        if ~bilinear:
            self.up = nn.Sequential(
                #kernel_size=4,stride=2,padding=1
                nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2,padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )

        else:
            self.up = nn.Sequential([
                nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
                nn.Conv3d(in_ch,out_ch,1),
                nn.InstanceNorm3d(out_ch),
                nn.ReLU(inplace=True)
            ])
        self.pam = PAM(out_ch)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.pam(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        diffZ = x1.size()[4] - x2.size()[4]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2),
                        diffZ // 2, int(diffZ / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


