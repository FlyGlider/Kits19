import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用的是batch_dice
class Criterion_dice(nn.modules.loss._Loss):
    def __init__(self, do_bg=False, size_average=None, reduce=None, reduction='mean'):
        super(Criterion_dice, self).__init__()
        self.do_bg = do_bg

    def forward(self, input, target):
        smooth = 1.
        one_hot = torch.zeros_like(input).scatter_(1, target.unsqueeze(1), 1)
        input = F.softmax(input, dim=1)
        total_loss = 0
        intersection = input * one_hot
        if self.do_bg:
            for i in range(input.size(1)):
                total_loss += (2. * intersection[:, i].sum() + smooth) / (input[:, i].sum() + one_hot[:, i].sum() + smooth)
                total_loss = total_loss / input.size(1)
        else:
            for i in range(1, input.size(1)):
                total_loss += (2. * intersection[:, i].sum() + smooth) / (input[:, i].sum() + one_hot[:, i].sum() + smooth)
            total_loss = total_loss / (input.size(1) - 1)
        return 1 - total_loss

class Criterion_surface(nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(Criterion_surface, self).__init__()

    def forward(self, input, target, idx=None):
        '''input是dist_map和softmax结果
        idx是花式索引,用于过滤某些类别
        '''
        if not (input.size() == target.size()):
            raise ValueError("input size ({}) must be the same as target size ({})".format(input.size(), target.size()))
        input = torch.exp(input)
        if idx is None:
            x = input * target
        else:
            x = input[:, idx] * target[:, idx]
        return x.mean()