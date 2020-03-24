import torch
import torch.nn as nn

DEVICE = 'cuda'


def dense(i, o, a=nn.Sigmoid):
    l = nn.Linear(i, o)
    return nn.Sequential(l, a())


def lam(func):
    class Lambda(nn.Module):
        def forward(self, *args):
            return func(*args)

    return Lambda()


class DQNAgent(nn.Module):
    def __init__(self, obs_size, num_actions):
        super(DQNAgent, self).__init__()

        hidden_dim = 128

        def make_dqn():
            return nn.Sequential(
                nn.Flatten(),
                dense(obs_size, hidden_dim, nn.ReLU),
                dense(hidden_dim, hidden_dim, nn.ReLU),
                nn.Linear(hidden_dim, num_actions),
            ).to(DEVICE)

        self.eval_net = make_dqn()
        self.target_net = make_dqn().eval()

    def forward(self, obs):
        obs = torch.FloatTensor([obs]).to(DEVICE)
        q_values = self.target_net(obs)[0]
        return q_values.argmax().cpu().detach().numpy()

    def replace_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    # https://github.com/cyoon1729/deep-Q-networks/blob/master/vanillaDQN/dqn.py#L51
    def loss(self, i, batch):
        discount = 0.95
        criterion = nn.MSELoss()

        if i % 50:
            self.replace_target()

        obs, actions, rewards, next_obs, done = batch
        obs = torch.FloatTensor(obs).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_obs = torch.FloatTensor(next_obs).to(DEVICE)
        done = torch.BoolTensor(done).to(DEVICE)

        q_values = self.eval_net(obs).gather(
            dim=1,
            index=actions.unsqueeze(0).T,
        ).squeeze()

        next_q_values, _ = self.target_net(next_obs).max(dim=1)
        target_q_values = next_q_values * discount + rewards
        target_q_values[done] = rewards[done]

        return criterion(q_values, target_q_values)

    def optimizer(self, lr):
        optim = torch.optim.Adam(params=self.eval_net.parameters(), lr=lr)
        i = 0

        def optimize(args):
            batch, episode_rewards, _randomness_list = args

            nonlocal i
            optim.zero_grad()
            loss = self.loss(i, batch)
            loss.backward()
            optim.step()

            return dict(
                loss=loss.item(),
                episode_rewards=episode_rewards[:-1],
            )

        return optimize


class RNNWorldModel(nn.Module):
    def __init__(self, obs_size):
        super(RNNWorldModel, self).__init__()

        self.obs_encoder = nn.Sequential(
            nn.Flatten(),
            dense(obs_size, 512, nn.ReLU),
            dense(512, 512, nn.ReLU),
            dense(512, 128, nn.Tanh),
        )

        self.time_transition = nn.GRU(128, 512, num_layers=2, batch_first=True)

        self.obs_decoder = nn.Sequential(
            dense(128, 512, nn.ReLU),
            dense(512, 512, nn.ReLU),
            dense(512, obs_size, nn.Sigmoid),
            lam(lambda x: x.reshape(16, 16, 3)),
        )

        self.obs_discriminator = nn.Sequential(
            nn.Flatten(),
            dense(obs_size, 512, nn.ReLU),
            dense(512, 1, nn.Sigmoid),
        )

        g_params = list(self.obs_encoder.parameters()) + \
                   list(self.time_transition.parameters()) + \
                   list(self.obs_decoder.parameters())
        d_params = self.obs_discriminator.parameters()

        self.g_optim = torch.optim.Adam(g_params)
        self.d_optim = torch.optim.Adam(d_params)
        self.criterion = nn.BCELoss()

    def optimize_G(self, X):
        pass

    def fit(self, batch, lr):
        X, Y = batch
