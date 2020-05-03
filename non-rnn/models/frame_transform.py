import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as torch_data

from models import torch_utils as tu


class ActionEncoder(tu.BaseModule):
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            tu.dense(num_actions, 32, a=nn.Tanh()),
        )

    def forward(self, x):
        return self.net(x)


class PreconditionEncoder(tu.BaseModule):
    def __init__(self, precondition_channels):
        super().__init__()
        self.net = nn.Sequential(
            tu.conv_block(i=precondition_channels, o=32, ks=3, s=1, p=0),
            tu.conv_block(i=32, o=32, ks=3, s=1, p=0),
            tu.conv_block(i=32, o=32, ks=5, s=1, p=2),
            tu.conv_block(i=32, o=32, ks=7, s=2, p=3),
            tu.conv_block(i=32, o=4, ks=8, s=2, p=4, a=None),
        )

    def forward(self, x):
        return self.net(x)


class ActionPreconditionFusion(tu.BaseModule):
    def __init__(self, input_size):
        super().__init__()

        def block(a):
            return nn.Sequential(
                tu.dense(input_size, 256, tu.get_activation()),
                tu.dense(256, 128, tu.get_activation()),
                tu.dense(128, 256, tu.get_activation()),
                tu.lam(lambda x: x.reshape(-1, 4, 8, 8)),
                tu.deconv_block(i=4, o=32, ks=5, s=1, p=1, d=1),
                tu.deconv_block(i=32, o=32, ks=5, s=1, p=1, d=1),
                tu.deconv_block(i=32, o=4, ks=5, s=1, p=2, d=2, a=a),
            )

        self.update_gate = block(a=nn.Sigmoid())
        self.update_state = block(a=nn.Tanh())

        self.to_rgb = nn.Sequential(
            tu.deconv_block(i=4, o=32, ks=4, s=1, p=1, d=1),
            tu.deconv_block(i=32, o=64, ks=5, s=2, p=2, d=1),
            tu.deconv_block(i=64, o=128, ks=7, s=2, p=3, d=1),
            tu.deconv_block(i=128, o=3, ks=2, s=1, p=1, d=1, a=nn.Sigmoid()),
        )

    def forward(self, x):
        actions, preconditions = x
        flat_preconditions = torch.flatten(preconditions, start_dim=1)
        cat_input = torch.cat([actions, flat_preconditions], dim=1)

        update_gate = F.sigmoid(self.update_gate(cat_input))
        update_state = self.update_state(cat_input)

        out = update_gate * preconditions + (1 - update_gate) * update_state
        out = self.to_rgb(out)

        return out


class ForwardModel(tu.BaseModule):
    def __init__(
        self,
        num_actions,
        precondition_channels,
        apf_in_size,
    ):
        super().__init__()
        self.num_actions = num_actions

        self.ae = ActionEncoder(num_actions)
        self.pe = PreconditionEncoder(precondition_channels)
        self.apf = ActionPreconditionFusion(apf_in_size)

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
        preconditions = preconditions / (255 + 0)
        return hot_actions, preconditions

    def preprocess_targets(self, y):
        # label smoothing
        return y / (255 + 0)

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
    def reset(self, precondition):
        self.last_action = None
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
        # should not label smooth
        return (frame * 255).astype(np.uint8)


def sanity_check():
    num_actions = 3

    actions = torch.rand(10, num_actions)
    ae = ActionEncoder(num_actions)
    action_activation = ae(actions)

    print(action_activation.shape)

    precondition_size = 1
    preconditions = torch.rand(10, precondition_size * 3, 64, 64)
    pe = PreconditionEncoder(precondition_size * 3)
    precondition_activation = pe(preconditions)

    print(precondition_activation.shape)

    apf = ActionPreconditionFusion(1024 + 32)
    fusion_activation = apf([action_activation, precondition_activation])

    print(fusion_activation.shape)

    fm = ForwardModel(
        num_actions=num_actions,
        precondition_channels=precondition_size * 3,
        apf_in_size=1024 + 32,
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
