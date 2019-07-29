import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli


class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x.shape[2])

            # sample from a mask
            mask_reduction = self.block_size - 1
            mask_height = x.shape[2] - mask_reduction
            mask_width = x.shape[3] - mask_reduction
            mask = Bernoulli(gamma).sample(
                (x.shape[0],x.shape[1], mask_height, mask_width))
            mask = F.pad(mask,(self.block_size//2,)*4)
            # compute block mask
            # block_mask = F.conv2d(mask, torch.ones(
            # (x.shape[1], 1, self.block_size, self.block_size)), padding=self.block_size//2,groups=x.shape[1])
            block_mask = F.max_pool2d(mask,(self.block_size,)*2,stride=(1,1),padding=self.block_size//2)

            # block_mask = 1 - (block_mask > 0).type_as(block_mask)
            block_mask = (1 - block_mask).to(device=x.device,dtype=x.dtype)

            #apple block mask
            out = x * block_mask

            #scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_gamma(self, feat_size):
        if feat_size < self.block_size:
            raise ValueError(
                'input.shape[2] can not be smaller than block_size')
        return (self.drop_prob / (self.block_size ** 2)) * ((feat_size ** 2)/((feat_size - self.block_size + 1) ** 2))

class DropBlock3D(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x.shape[2])

            # sample from a mask
            mask_reduction = self.block_size - 1
            mask_height = x.shape[2] - mask_reduction
            mask_width = x.shape[3] - mask_reduction
            mask = Bernoulli(gamma).sample(
                (x.shape[0],x.shape[1], mask_height, mask_width))
            mask = F.pad(mask,(self.block_size//2,)*4)
            # compute block mask
            # block_mask = F.conv2d(mask, torch.ones(
            # (x.shape[1], 1, self.block_size, self.block_size)), padding=self.block_size//2,groups=x.shape[1])
            block_mask = F.max_pool2d(mask,(self.block_size,)*2,stride=(1,1),padding=self.block_size//2)

            # block_mask = 1 - (block_mask > 0).type_as(block_mask)
            block_mask = (1 - block_mask).to(device=x.device,dtype=x.dtype)

            #apple block mask
            out = x * block_mask

            #scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_gamma(self, feat_size):
        if feat_size < self.block_size:
            raise ValueError(
                'input.shape[2] can not be smaller than block_size')
        return (self.drop_prob / (self.block_size ** 2)) * ((feat_size ** 2)/((feat_size - self.block_size + 1) ** 2))