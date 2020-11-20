import torch.nn as nn
from model import common

class EDSR(nn.Module):
    def __init__(self, upscale_factor, input_channels, target_channels, n_resblocks, n_feats, res_scale, bn=False, act=nn.ReLU(True), conv=common.default_conv, head_patch_extraction_size=5, kernel_size=3, early_upsampling=False):
        super(EDSR, self).__init__()

        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.scale = res_scale
        self.act = nn.ReLU(True)
        self.bn = bn
        self.input_channels = input_channels
        self.target_channels = target_channels

        # define head module
        m_head = [conv(len(input_channels), n_feats, head_patch_extraction_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale, bn=self.bn
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail_late_upsampling = [
            common.Upsampler(conv, upscale_factor, n_feats, act=False),
            conv(n_feats, len(target_channels), kernel_size)
        ]
        m_tail_early_upsampling = [
            conv(n_feats, len(target_channels), kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        if early_upsampling:
            self.tail = nn.Sequential(*m_tail_early_upsampling)
        else:
            self.tail = nn.Sequential(*m_tail_late_upsampling)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        return x 

