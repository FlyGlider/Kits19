import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import BasicBlock


class PreActResNet(nn.Module):
    def __init__(self,block,layers,nclasses=3):
        super(PreActResNet,self).__init__()
        self.inc = inconv(1,64)
        self.down1 = down(PreActBlock,layers[0],64,128)
        self.down2 = down(PreActBlock,layers[1],128,256)
        self.down3 = down(PreActBlock,layers[2],256,512)
        self.down4 = down(PreActBlock,layers[3],512,1024)
        self.down5 = down(PreActBlock,layers[4],1024,2048)
        self.up1 = up(PreActBlock,2048,1024)
        self.up2 = up(PreActBlock,1024,512)
        self.up3 = up(PreActBlock,512,256)
        self.up4 = up(PreActBlock,256,128)
        self.up5 = up(PreActBlock,128,64)
        self.outc = outconv(64,nclasses)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #使用kaiming初始化效果并不好
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = F.dropout(x4)
        x5 = self.down4(x4)
        x5 = F.dropout(x5)
        x6 = self.down5(x5)
        x6 = F.dropout(x6)
        x = self.up1(x6,x5)
        x = self.up2(x,x4)
        x = self.up3(x,x3)
        x = self.up4(x,x2)
        x = self.up5(x,x1)
        x = self.outc(x)
        
        return F.log_softmax(x,dim=1)

def conv3x3(in_ch,out_ch,stride=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_ch,out_ch,3,stride=stride,padding=1,bias=False)

def conv1x1(in_ch,out_ch):
    return nn.Conv2d(in_ch,out_ch,1,bias=False)

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1,bias=False),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1,bias=False)
            # nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x

class PreActBlock(nn.Module):
    expansion = 1
    def __init__(self,in_ch,out_ch,stride=1,downsample=None):
        super(PreActBlock,self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()
        self.conv1 = conv3x3(in_ch,out_ch)
        # self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch,out_ch)
        self.stride = stride
        self.downsample = downsample
    
    def forward(self,x):
        
        # out = self.bn1(x)
        out = self.relu(x)
        if self.downsample is not None:
            residual = self.downsample(out)
        else:
            residual = x
        out = self.conv1(out)
        # out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out

class PreActBottleBlock(nn.Module):
    expansion = 4
    def __init__(self,in_ch,out_ch,stride=1,downsample=None):
        super(PreActBlock,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()
        self.conv1 = conv1x1(in_ch,out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch,out_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.conv3 = conv1x1(out_ch,out_ch*self.expansion)
        self.stride = stride
        self.downsample = downsample
    
    def forward(self,x):
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        else:
            residual = x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out

class down(nn.Module):
    def __init__(self,block,blocks,in_ch, out_ch):
        super(down, self).__init__()
        #这里的参数还差个stride,默认为1就不写了
        self.layer = self._make_layer(block,in_ch,out_ch,blocks)
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            self.layer
        )

    def _make_layer(self,block,in_ch,out_ch,blocks=1,stride=1):
        '''blocks是block的数量，以bottelneck为例，第一次输入通道数为64，
        经一个block后，输出通道数为256，所以需要一个downsample将输入转换
        为256通道的，后面输入就是256通道的了，所以就不需要downsample了'''
        downsample = None
        if stride != 1 or in_ch != out_ch*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch,out_ch * block.expansion,1,stride=stride,bias=False)
                
            )
        layers = []
        layers.append(block(in_ch,out_ch,stride,downsample))
        in_ch = out_ch * block.expansion
        for i in range(1,blocks):
            layers.append(block(in_ch,out_ch))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.mpconv(x)
        return x

#x1是由BRCBRC得来的
class up(nn.Module):
    def __init__(self,block,in_ch,out_ch,bilinear = True):
        super(up,self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                # nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
                nn.Conv2d(in_ch,out_ch,1,bias=False)
            )   
        else:
            self.up = nn.Sequential(
                # nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_ch,out_ch,4,2,padding=1)
            )
        
        self.conv = block(out_ch,out_ch)
        
    def forward(self,x1,residual):
        x1 = self.up(x1)
        x = residual + x1
        # x = F.relu(x,inplace=True)
        # x = torch.cat([residual,x1],dim=1)

        out = self.conv(x)
        return out

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()

        self.conv = nn.Sequential(
            # nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch,out_ch,1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
