import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class DFN(nn.Module):
    def __init__(self, out_planes, criterion=None, aux_criterion=None, alpha=None,
                 pretrained_model=None,
                 norm_layer=nn.BatchNorm3d, stem_width=32):
        super(DFN, self).__init__()
        self.backbone = resnet101(pretrained_model, norm_layer=norm_layer,
                                  bn_eps=1e-05,
                                  bn_momentum=0.1,
                                  deep_stem=True, stem_width=stem_width)
        self.business_layer = []

        smooth_inner_channel = 256
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            ConvBnRelu(32 * stem_width, smooth_inner_channel, 1, 1, 0, # 2048
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer)
        )
        self.business_layer.append(self.global_context)

        stage = [32 * stem_width, 16 * stem_width, 8 * stem_width, 4 * stem_width] # 2048 ,1024, 512, 256
        self.smooth_pre_rrbs = []
        self.cabs = []
        self.smooth_aft_rrbs = []
        self.smooth_heads = []

        for i, channel in enumerate(stage):
            self.smooth_pre_rrbs.append(
                RefineResidual(channel, smooth_inner_channel, 3, has_bias=False,
                               has_relu=True, norm_layer=norm_layer))
            self.cabs.append(
                ChannelAttention(smooth_inner_channel * 2,
                                 smooth_inner_channel, 1))
            self.smooth_aft_rrbs.append(
                RefineResidual(smooth_inner_channel, smooth_inner_channel, 3,
                               has_bias=False,
                               has_relu=True, norm_layer=norm_layer))
            self.smooth_heads.append(
                DFNHead(smooth_inner_channel, out_planes, 2 ** (4 - i),
                        norm_layer=norm_layer))

        stage.reverse()
        border_inner_channel = 2
        self.border_pre_rrbs = []
        self.border_aft_rrbs = []
        self.border_heads = []

        for i, channel in enumerate(stage):
            self.border_pre_rrbs.append(
                RefineResidual(channel, border_inner_channel, 3, has_bias=False,
                               has_relu=True, norm_layer=norm_layer))
            self.border_aft_rrbs.append(
                RefineResidual(border_inner_channel, border_inner_channel, 3,
                               has_bias=False,
                               has_relu=True, norm_layer=norm_layer))
            self.border_heads.append(
                DFNHead(border_inner_channel, 1, 2, norm_layer=norm_layer)) # 这里好像不对

        self.smooth_pre_rrbs = nn.ModuleList(self.smooth_pre_rrbs)
        self.cabs = nn.ModuleList(self.cabs)
        self.smooth_aft_rrbs = nn.ModuleList(self.smooth_aft_rrbs)
        self.smooth_heads = nn.ModuleList(self.smooth_heads)
        self.border_pre_rrbs = nn.ModuleList(self.border_pre_rrbs)
        self.border_aft_rrbs = nn.ModuleList(self.border_aft_rrbs)
        self.border_heads = nn.ModuleList(self.border_heads)

        self.business_layer.append(self.smooth_pre_rrbs)
        self.business_layer.append(self.cabs)
        self.business_layer.append(self.smooth_aft_rrbs)
        self.business_layer.append(self.smooth_heads)
        self.business_layer.append(self.border_pre_rrbs)
        self.business_layer.append(self.border_aft_rrbs)
        self.business_layer.append(self.border_heads)

        self.criterion = criterion
        self.aux_criterion = aux_criterion
        self.alpha = alpha

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # 使用kaiming初始化效果并不好
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data, label=None, aux_label=None):
        blocks = self.backbone(data)
        blocks.reverse()

        global_context = self.global_context(blocks[0])
        global_context = F.interpolate(global_context,
                                       size=blocks[0].size()[2:],
                                       mode='trilinear', align_corners=True)

        last_fm = global_context
        pred_out = []
        # smooth network
        for i, (fm, pre_rrb,
                cab, aft_rrb, head) in enumerate(zip(blocks,
                                                     self.smooth_pre_rrbs,
                                                     self.cabs,
                                                     self.smooth_aft_rrbs,
                                                     self.smooth_heads)):
            fm = pre_rrb(fm)
            fm = cab(fm, last_fm)
            fm = aft_rrb(fm)
            pred_out.append(head(fm))
            if i != 3:
                last_fm = F.interpolate(fm, scale_factor=2, mode='trilinear',
                                        align_corners=True)

        blocks.reverse()
        last_fm = None
        boder_out = []
        for i, (fm, pre_rrb,
                aft_rrb, head) in enumerate(zip(blocks,
                                                self.border_pre_rrbs,
                                                self.border_aft_rrbs,
                                                self.border_heads)):
            fm = pre_rrb(fm)
            if last_fm is not None:
                fm = F.interpolate(fm, scale_factor=2 ** i, mode='trilinear',
                                   align_corners=True)
                last_fm = last_fm + fm
                last_fm = aft_rrb(last_fm)

            else:
                last_fm = fm
            boder_out.append(head(last_fm))

        if label is not None and aux_label is not None:
            loss0 = self.criterion(pred_out[0], label)
            loss1 = self.criterion(pred_out[1], label)
            loss2 = self.criterion(pred_out[2], label)
            loss3 = self.criterion(pred_out[3], label)

            aux_loss0 = self.aux_criterion(boder_out[0], aux_label)
            aux_loss1 = self.aux_criterion(boder_out[1], aux_label)
            aux_loss2 = self.aux_criterion(boder_out[2], aux_label)
            aux_loss3 = self.aux_criterion(boder_out[3], aux_label)

            loss = loss0 + loss1 + loss2 + loss3
            aux_loss = aux_loss0 + aux_loss1 + aux_loss2 + aux_loss3
            return loss, self.alpha * aux_loss

        return F.log_softmax(pred_out[-1], dim=1)


class DFNHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale, norm_layer=nn.BatchNorm3d):
        super(DFNHead, self).__init__()
        self.rrb = RefineResidual(in_planes, out_planes * 9, 3, has_bias=False,
                                  has_relu=False, norm_layer=norm_layer)
        self.conv = nn.Conv3d(out_planes * 9, out_planes, kernel_size=1,
                              stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        x = self.rrb(x)
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear',
                          align_corners=True)

        return x


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None,
                 bn_eps=1e-5, bn_momentum=0.1, downsample=None, inplace=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.inplace:
            out += residual
        else:
            out = out + residual

        out = self.relu_inplace(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 norm_layer=None, bn_eps=1e-5, bn_momentum=0.1,
                 downsample=None, inplace=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

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

        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, norm_layer=nn.BatchNorm3d, bn_eps=1e-5,
                 bn_momentum=0.1, deep_stem=False, stem_width=32, inplace=True):
        self.inplanes = stem_width
        super(ResNet, self).__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv3d(1, stem_width, kernel_size=3, stride=1, padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv3d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace)
            )
        else:
            self.conv1 = nn.Conv3d(3, stem_width, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, stem_width, layers[0],
                                       inplace,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, stem_width * 2, layers[1],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, stem_width * 4, layers[2],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, norm_layer, stem_width * 8, layers[3],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)

    def _make_layer(self, block, norm_layer, planes, blocks, inplace=True,
                    stride=1, bn_eps=1e-5, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion, eps=bn_eps,
                           momentum=bn_momentum),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, norm_layer, bn_eps,
                            bn_momentum, downsample, inplace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer, bn_eps=bn_eps,
                                bn_momentum=bn_momentum, inplace=inplace))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        blocks = []
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        x = self.layer4(x)
        blocks.append(x)

        return blocks


def resnet18(pretrained_model=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    # if pretrained_model is not None:
    #     model = load_model(model, pretrained_model)
    return model


def resnet34(pretrained_model=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    # if pretrained_model is not None:
    #     model = load_model(model, pretrained_model)
    return model


def resnet50(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    # if pretrained_model is not None:
    #     model = load_model(model, pretrained_model)
    return model


def resnet101(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    # if pretrained_model is not None:
    #     model = load_model(model, pretrained_model)
    return model


def resnet152(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    # if pretrained_model is not None:
    #     model = load_model(model, pretrained_model)
    return model

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm3d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class SeparableConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, dilation=1,
                 has_relu=True, norm_layer=nn.BatchNorm3d):
        super(SeparableConvBnRelu, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size, stride,
                               padding, dilation, groups=in_channels,
                               bias=False)
        self.bn = norm_layer(in_channels)
        self.point_wise_cbr = ConvBnRelu(in_channels, out_channels, 1, 1, 0,
                                         has_bn=True, norm_layer=norm_layer,
                                         has_relu=has_relu, has_bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.point_wise_cbr(x)
        return x


class SELayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1, 1)
        return y


# For DFN
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, reduction):
        super(ChannelAttention, self).__init__()
        self.channel_attention = SELayer(in_planes, out_planes, reduction)

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], 1)
        channel_attetion = self.channel_attention(fm)
        fm = x1 * channel_attetion + x2

        return fm


class BNRefine(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, has_bias=False,
                 has_relu=False, norm_layer=nn.BatchNorm3d, bn_eps=1e-5):
        super(BNRefine, self).__init__()
        self.conv_bn_relu = ConvBnRelu(in_planes, out_planes, ksize, 1,
                                       ksize // 2, has_bias=has_bias,
                                       norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv3d(out_planes, out_planes, kernel_size=ksize,
                                     stride=1, padding=ksize // 2, dilation=1,
                                     bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        t = self.conv_bn_relu(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x


class RefineResidual(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, has_bias=False,
                 has_relu=False, norm_layer=nn.BatchNorm3d, bn_eps=1e-5):
        super(RefineResidual, self).__init__()
        self.conv_1x1 = nn.Conv3d(in_planes, out_planes, kernel_size=1,
                                  stride=1, padding=0, dilation=1,
                                  bias=has_bias)
        self.cbr = ConvBnRelu(out_planes, out_planes, ksize, 1,
                              ksize // 2, has_bias=has_bias,
                              norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv3d(out_planes, out_planes, kernel_size=ksize,
                                     stride=1, padding=ksize // 2, dilation=1,
                                     bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv_1x1(x)
        t = self.cbr(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25,
                 reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_bce = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        b, d, h, w = target.size()
        pred = pred.view(b, d, h, w)
        pred_sigmoid = pred.sigmoid()

        loss = self.loss_bce(pred_sigmoid, target)
        focal = ((1 - pred_sigmoid) ** self.gamma * target + pred_sigmoid ** self.gamma * (1 - target)).detach()
       
        loss = focal * loss
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(
        group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group

if __name__ == '__main__':
    model = resnet101(deep_stem=True, stem_width=64)