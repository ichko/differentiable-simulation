import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as torch_data

from models import torch_utils as tu


class ActionEncoder(tu.BaseModule):
    def __init__(self, num_actions, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            tu.dense(num_actions, 32, tu.get_activation()),
            tu.dense(32, 32, tu.get_activation()),
            tu.dense(32, 32, tu.get_activation()),
            tu.dense(32, out_channels, a=None),
        )

    def forward(self, x):
        return self.net(x)


class PreconditionEncoder(tu.BaseModule):
    def __init__(self, precondition_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            tu.conv_block(i=precondition_channels, o=32, ks=5, s=2, p=2),
            tu.conv_block(i=32, o=64, ks=5, s=2, p=2),
            tu.conv_block(i=64, o=64, ks=5, s=1, p=2),
            tu.conv_block(i=64, o=32, ks=3, s=2, p=1),
            tu.conv_block(i=32, o=32, ks=3, s=2, p=1),
            nn.Flatten(),
            tu.dense(512, out_channels, a=None),
        )

    def forward(self, x):
        return self.net(x)


class ActionPreconditionFusion(tu.BaseModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            tu.lam(lambda x: torch.cat(x, dim=1)),
            tu.dense(in_channels, 512, tu.get_activation()),
            tu.lam(lambda x: x.reshape(-1, 8, 8, 8)),
            tu.deconv_block(i=8, o=32, ks=5, s=1, p=2, d=1),
            tu.deconv_block(i=32, o=32, ks=5, s=1, p=2, d=1),
            tu.deconv_block(i=32, o=64, ks=5, s=2, p=0, d=2),
            tu.deconv_block(i=64, o=32, ks=9, s=2, p=2, d=2),
            tu.deconv_block(i=32, o=32, ks=9, s=1, p=2, d=1),
            tu.deconv_block(i=32, o=32, ks=3, s=1, p=0, d=1),
            tu.conv_block(
                i=32,
                o=out_channels,
                ks=2,
                s=1,
                p=1,
                a=nn.Sigmoid(),
            ),
        )

    def forward(self, x):
        return self.net(x)


class ForwardModel(tu.BaseModule):
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
        self.optim = torch.optim.Adam(self.parameters(), 1)

        def lr_lambda(epoch):
            return lr / (epoch // 20 + 1)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda=lr_lambda,
        )

    def preprocess_input(self, x, one_hot_size=None):
        actions, preconditions = x
        hot_actions = tu.one_hot(actions, one_hot_size).to(actions.device)
        preconditions = preconditions / 255
        return hot_actions, preconditions

    def preprocess_targets(self, y):
        # label smoothing
        return y / 255

    def to_dataloader(self, data, bs):
        actions, preconditions, futures = data

        actions = torch.LongTensor(actions)
        preconditions = torch.FloatTensor(preconditions)
        futures = torch.FloatTensor(futures)

        dataset = torch_data.TensorDataset(actions, preconditions, futures)
        return torch_data.DataLoader(dataset, batch_size=bs)

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
    def reset(self, precondition, num_actions):
        self.last_action = None
        self.num_actions = num_actions
        self.first_precondition = precondition
        self.last_precondition = precondition

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

        x = self.preprocess_input(
            [
                torch.LongTensor([self.last_action]),
                torch.FloatTensor([self.last_precondition])
            ],
            self.num_actions,
        )

        frame = self(x)[0].detach().cpu().numpy()
        return (frame * 255).astype(np.uint8)


def sanity_check():
    num_actions = 3

    actions = torch.rand(10, num_actions)
    ae = ActionEncoder(num_actions, out_channels=32)
    action_activation = ae(actions)

    print(action_activation.shape)

    precondition_size = 2
    preconditions = torch.rand(10, precondition_size * 3, 64, 64)
    pe = PreconditionEncoder(precondition_size * 3, out_channels=512)
    precondition_activation = pe(preconditions)

    print(precondition_activation.shape)

    apf = ActionPreconditionFusion(in_channels=512 + 32, out_channels=64)
    fusion_activation = apf([action_activation, precondition_activation])

    print(fusion_activation.shape)

    fm = ForwardModel(
        num_actions=num_actions,
        action_output_channels=32,
        precondition_channels=precondition_size * 3,
        precondition_out_channels=512,
    )

    future_frame = fm([actions, preconditions])
    fm.optim_init(lr=0.1)

    action_ids = torch.randint(0, num_actions, size=(10, ))
    future_frame_loss, _info = fm.optim_step(
        [[action_ids, preconditions], future_frame], )

    print(future_frame.shape)
    print(future_frame_loss)

    print(f'''
        MODELS SIZES:
            - ACTION ENCODER       {ae.count_parameters():09,}
            - PRECONDITION ENCODER {pe.count_parameters():09,}
            - FUSION               {apf.count_parameters():09,}
            - WHOLE MODEL          {fm.count_parameters():09,}
    ''')

    print('--- SANITY CHECK END --- ')


if __name__ == '__main__':
    sanity_check()
