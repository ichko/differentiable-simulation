import torch
import torch.nn as nn

import torch_utils as tu


class ActionEncoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            tu.dense(1, 32, tu.get_activation()),
            tu.dense(32, 32, tu.get_activation()),
            tu.dense(32, 64, tu.get_activation()),
            tu.lam(lambda x: x.reshape(-1, 1, 8, 8)),
            tu.deconv_block(i=1, o=32, ks=5, s=1, p=0, d=2),
            tu.deconv_block(i=32, o=32, ks=5, s=1, p=0, d=2),
            tu.deconv_block(i=32, o=out_channels, ks=5, s=1, p=0, d=2),
        )

    def forward(self, x):
        # x      - (bs, 1)
        # return - (bs, 32, 32, 32)
        return self.net(x)


class PreconditionEncoder(nn.Module):
    def __init__(self, precondition_size, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            tu.conv_block(i=precondition_size, o=32, ks=3, s=1, p=1),
            tu.conv_block(i=32, o=32, ks=3, s=1, p=1),
            tu.conv_block(i=32, o=out_channels, ks=3, s=1, p=1),
        )

    def forward(self, x):
        return self.net(x)


class ActionPreconditionFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            tu.lam(lambda x: torch.cat(x, dim=1)),
            tu.conv_block(i=in_channels, o=32, ks=3, s=1, p=1),
            tu.conv_block(i=32, o=32, ks=3, s=1, p=1),
            tu.conv_block(i=32, o=32, ks=3, s=1, p=1),
            tu.conv_block(i=32, o=out_channels, ks=3, s=1, p=1),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    out_channels = 32

    action = torch.rand(10, 1)
    ae = ActionEncoder(out_channels)
    action_activation = ae(action)

    print(action_activation.shape)

    precondition_size = 2
    precondition = torch.rand(10, precondition_size, 32, 32)
    pe = PreconditionEncoder(precondition_size, out_channels)
    precondition_activation = pe(precondition)

    print(precondition_activation.shape)

    apf = ActionPreconditionFusion(64, 32)
    fusion_activation = apf([action_activation, precondition_activation])

    print(fusion_activation.shape)
