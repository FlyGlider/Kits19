# full assembly of the sub-parts to form the complete net

from models.unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_classes=2, use_linear=True, deep_supervised=False):
        super(UNet, self).__init__()
        self.inc = inconv(1, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.up3 = up(512, 256, use_linear)
        self.up2 = up(256, 128, use_linear)
        self.up1 = up(128, 64, use_linear)
        self.outc = outconv(64, n_classes)
        self.deep_supervised = deep_supervised
        self.n_classes=n_classes
        if self.deep_supervised:
            self.blocks = []
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                #使用kaiming初始化效果并不好
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
        return F.log_softmax(x, 1)
    

class UNet2D(nn.Module):
    def __init__(self, n_classes=2, use_linear=True):
        super(UNet2D,self).__init__()
        self.inc = inconv_2d(1, 64)
        self.down1 = down_2d(64, 128)
        self.down2 = down_2d(128, 256)
        self.down3 = down_2d(256, 512)
        self.down4 = down_2d(512, 1024)
        self.up4 = up_2d(1024, 512, use_linear)
        self.up3 = up_2d(512, 256, use_linear)
        self.up2 = up_2d(256, 128, use_linear)
        self.up1 = up_2d(128, 64, use_linear)
        self.outc = outconv_2d(64, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #使用kaiming初始化效果并不好
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x4 = F.dropout(x4)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.outc(x)
        return F.log_softmax(x,1)
