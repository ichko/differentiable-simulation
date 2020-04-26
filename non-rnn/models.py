import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as torch_data

import torch_utils as tu


class ActionEncoder(nn.Module):
    def __init__(self, num_actions, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            tu.dense(num_actions, 32, tu.get_activation()),
            tu.dense(32, 32, tu.get_activation()),
            tu.dense(32, 64, tu.get_activation()),
            tu.lam(lambda x: x.reshape(-1, 1, 8, 8)),
            tu.deconv_block(i=1, o=32, ks=5, s=1, p=0, d=2),
            tu.deconv_block(i=32, o=32, ks=5, s=1, p=0, d=2),
            tu.deconv_block(i=32, o=out_channels, ks=5, s=1, p=0, d=2, a=None),
        )

    def forward(self, x):
        # x      - (bs, 1)
        # return - (bs, 32, 32, 32)
        return self.net(x)


class PreconditionEncoder(nn.Module):
    def __init__(self, precondition_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            tu.conv_block(i=precondition_channels, o=32, ks=3, s=1, p=1),
            tu.conv_block(i=32, o=32, ks=3, s=1, p=1),
            tu.conv_block(i=32, o=out_channels, ks=3, s=1, p=1, a=None),
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
            tu.conv_block(
                i=32,
                o=out_channels,
                ks=3,
                s=1,
                p=1,
                a=nn.Sigmoid(),
            ),
        )

    def forward(self, x):
        return self.net(x)


class ForwardModel(tu.PersistedModule):
    def __init__(
        self,
        num_actions,
        action_output_channels,
        precondition_channels,
        precondition_out_channels,
    ):
        super().__init__()
        self.ae = ActionEncoder(num_actions, action_output_channels)
        self.pe = PreconditionEncoder(
            precondition_channels,
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

    def optim_init(self, lr):
        self.optim = torch.optim.Adam(self.parameters(), lr)

    def preprocess_input(self, x):
        actions, preconditions = x
        actions = tu.one_hot(torch.LongTensor(actions))
        preconditions = preconditions / 255
        return actions, preconditions

    def preprocess_targets(self, y):
        # label smoothing
        return y / (255 + 5)

    def to_dataloader(self, data, bs):
        data = [torch.FloatTensor(t) for t in data]
        dataset = torch_data.TensorDataset(*data)
        return torch_data.DataLoader(dataset, batch_size=hparams.bs)

    def optim_step(self, batch):
        x, y = batch
        x, y = self.preprocess_input(x), self.preprocess_targets(y)

        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)

        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        info = dict(y_pred=y_pred, y=y)

        return loss, info


class ForwardGym(ForwardModel):
    def __init__(self, precondition, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_action = None
        self.first_precondition = precondition
        self.last_precondition = precondition

    def reset(self):
        self.last_precondition = self.first_precondition

    def step(self, action):
        self.last_action = action

        pred_frame = self.render('rgb_array')
        self.last_precondition = np.roll(
            self.last_precondition,
            shift=-1,
            axis=0,
        )
        self.last_precondition[-3:] = pred_frame

        # obs, reward, done, info
        return pred_frame, 0, False, {}

    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            raise NotImplementedError(f'mode "{mode}" is not supported')

        last_precondition = self.last_precondition
        x = [[self.last_action], [last_precondition]]
        x = [np.array(t) for t in x]
        x = self.preprocess_input(x)
        x = [torch.FloatTensor(t) for t in x]

        frame = self(x)[0].detach().cpu().numpy()
        return frame


def sanity_check():
    num_actions = 3
    out_channels = 32

    actions = torch.rand(10, num_actions)
    ae = ActionEncoder(num_actions, out_channels)
    action_activation = ae(actions)

    print(action_activation.shape)

    precondition_size = 2
    preconditions = torch.rand(10, precondition_size * 3, 32, 32)
    pe = PreconditionEncoder(precondition_size * 3, out_channels)
    precondition_activation = pe(preconditions)

    print(precondition_activation.shape)

    apf = ActionPreconditionFusion(64, 32)
    fusion_activation = apf([action_activation, precondition_activation])

    print(fusion_activation.shape)

    fm = ForwardModel(
        num_actions=num_actions,
        action_output_channels=32,
        precondition_channels=precondition_size * 3,
        precondition_out_channels=32,
    )
    future_frame = fm([actions, preconditions])
    fm.optim_init(lr=0.1)
    future_frame_loss, _info = fm.optim_step(
        [[actions, preconditions], future_frame], )

    print(future_frame.shape)
    print(future_frame_loss)
    print('--- SANITY CHECK END --- ')


if __name__ == '__main__':
    sanity_check()
