import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import *

#  在nnunet中当池化为1时,卷积也为1,但俺不是
class ResUNet(nn.Module):
    MAX_NUM_FILTERS_3D = 320
    MAX_NUM_FILTERS_2D = 480
    NUM_CONVS = 1
    def __init__(self, in_ch=1, base_num_features=30, num_classes=3,
                 norm_type='batch', nonlin_type='relu', pool_type='max',
                 pool_layer_kernel_sizes=None, deep_supervision=True, mode='2D'):
        super(ResUNet, self).__init__()
        if mode not in ['2D', '3D', '3D_LOW']:
            raise NotImplementedError('mode [%s] is not found' % mode)
        if mode == '2D':
            upsample_mode = 'bilinear'
        else:
            upsample_mode = 'trilinear'
        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.conv_layers = []
        self.upsample_layers = []
        self.num_pool = len(pool_layer_kernel_sizes)
        self.deep_supervision = deep_supervision
        self.mode = mode
        pool_layer = get_pool_layer(pool_type=pool_type, mode=mode)
        
        out_ch = base_num_features
        in_ch = in_ch

        # down
        for d in range(self.num_pool):
            self.conv_blocks_context.append(StackedResidualBlocks(block=ResidualBasicBlock, in_ch=in_ch, out_ch=out_ch, blocks=self.NUM_CONVS,
                                            norm_type=norm_type, nonlin_type=nonlin_type, mode=mode))
            self.td.append(pool_layer(pool_layer_kernel_sizes[d]))

            in_ch = out_ch
            out_ch = out_ch * 2
            if mode == '2D':
                out_ch = min(out_ch, self.MAX_NUM_FILTERS_2D)
            else:
                out_ch = min(out_ch, self.MAX_NUM_FILTERS_3D)
        
        # bottleneck
        # nnunet的做法是上采样后直接cat,而我原来是上采样后conv+bn再cat
        final_ch = self.conv_blocks_context[-1].out_ch
        self.conv_blocks_context.append(nn.Sequential(
            StackedResidualBlocks(block=ResidualBasicBlock, in_ch=in_ch, out_ch=final_ch, blocks=self.NUM_CONVS,
                                  norm_type=norm_type, nonlin_type=nonlin_type, mode=mode)
        ))
        # up
        pool_layer_kernel_sizes.reverse()
        for u in range(self.num_pool):
            ch_from_skip = self.conv_blocks_context[-(2 + u)].out_ch
            # 注意这里换成了sum
            ch_after_tu_and_sum = ch_from_skip
            if u != self.num_pool - 1:
                final_ch = self.conv_blocks_context[-(3 + u)].out_ch
            else:
                final_ch = ch_from_skip

            self.tu.append(Upsample(scale_factor=pool_layer_kernel_sizes[u], mode=upsample_mode))
            self.conv_blocks_localization.append(nn.Sequential(
                StackedResidualBlocks(block=ResidualBasicBlock, in_ch=ch_after_tu_and_sum, out_ch=final_ch, blocks=self.NUM_CONVS,
                                      norm_type=norm_type, nonlin_type=nonlin_type, mode=mode)
            ))
        # 做深度监督需要用到的conv,将输出通道数映射到类别数
        for ds in range(self.num_pool):
            self.conv_layers.append(get_conv_layer(in_ch=self.conv_blocks_localization[ds][-1].out_ch,
                                    out_ch=num_classes, kernel_size=1, use_bias=True, mode=mode))
        # 需要将结果倒序
        pool_layer_kernel_sizes.reverse()
        cum_upsample = np.cumprod(np.vstack(pool_layer_kernel_sizes), axis=0)[::-1].tolist()
        # bottleneck不使用,最后一个localization不需要上采样
        for usl in range(self.num_pool - 1):
            self.upsample_layers.append(Upsample(scale_factor=tuple(cum_upsample[usl + 1]), mode=upsample_mode))
        
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.upsample_layers = nn.ModuleList(self.upsample_layers)


    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(self.num_pool):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            x = self.td[d](x)
        x = self.conv_blocks_context[-1](x)
        for u in range(self.num_pool):
            x = self.tu[u](x)
            x = x + skips[-(u + 1)]
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.conv_layers[u](x))
        if self.deep_supervision:
            # 顺序是从高分辨率到低分辨率
            return tuple([seg_outputs[-1]] + [i(j) for i, j in zip(self.upsample_layers[::-1], seg_outputs[:-1][::-1])])
        else:
            return (seg_outputs[-1])

if __name__ == '__main__':
    tmp = torch.randn(2, 1, 64, 64, 16).cuda()
    model = ResUNet(in_ch=1, base_num_features=30, num_classes=3, norm_type='batch', nonlin_type='relu', pool_type='max',
                    pool_layer_kernel_sizes=[(2, 2, 2), (2, 2, 2), (2, 2, 2)], deep_supervision=True, mode='3D').cuda()
    output = model(tmp)
    plt.imshow(output[0].detach().cpu().numpy()[0, 0, 1])
    print('')

