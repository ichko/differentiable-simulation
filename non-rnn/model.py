import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ForwardModel(tu.PersistedModule):
    def __init__(
        self,
        action_output_channels,
        precondition_size,
        precondition_out_channels,
    ):
        super().__init__()
        self.ae = ActionEncoder(action_output_channels)
        self.pe = PreconditionEncoder(
            precondition_size,
            precondition_out_channels,
        )
        self.apf = ActionPreconditionFusion(
            action_output_channels + precondition_out_channels,
            out_channels=3,  # RGB
        )

    def forward(self, x):
        actions, preconditions = x

        action_activation = self.ae(actions)
        precondition_activation = self.pe(preconditions)
        pred_future_frame = self.apf(
            [action_activation, precondition_activation])

        return pred_future_frame

    def loss(self, x, y):
        y_pred = self.forward(x)
        return F.mse_loss(y_pred, y)


def sanity_check():
    out_channels = 32

    actions = torch.rand(10, 1)
    ae = ActionEncoder(out_channels)
    action_activation = ae(actions)

    print(action_activation.shape)

    precondition_size = 2
    preconditions = torch.rand(10, precondition_size, 32, 32)
    pe = PreconditionEncoder(precondition_size, out_channels)
    precondition_activation = pe(preconditions)

    print(precondition_activation.shape)

    apf = ActionPreconditionFusion(64, 32)
    fusion_activation = apf([action_activation, precondition_activation])

    print(fusion_activation.shape)

    fm = ForwardModel(
        action_output_channels=32,
        precondition_size=2,
        precondition_out_channels=32,
    )
    future_frame = fm([actions, preconditions])
    future_frame_loss = fm.loss([actions, preconditions], future_frame)

    print(future_frame.shape)
    print(future_frame_loss)
    print('--- SANITY CHECK END --- ')


if __name__ == '__main__':
    sanity_check()