import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation():
    LEAKY_SLOPE = 0.2
    return nn.LeakyReLU(LEAKY_SLOPE, inplace=True)


def cat_channels(t):
    """
        Concatenate number of channels in a single tensor
        Converts tensor with shape:
            (bs, num_channels, channel_size, h, w)
        to tensor with shape:
            (bs, num_channels * channel_size, h, w)

    """
    shape = t.size()
    cat_dim_size = shape[1] * shape[2]
    return t.view(-1, cat_dim_size, *shape[3:])


class PersistedModule(nn.Module):
    def make_persisted(self, path):
        self.path = path

    def persist(self):
        torch.save(self.state_dict(), self.path)

    def preload_weights(self):
        self.load_state_dict(torch.load(self.path))

    def can_be_preloaded(self):
        return os.path.isfile(self.path)


def dense(i, o, a=None):
    l = nn.Linear(i, o)
    return l if a is None else nn.Sequential(l, a)


def lam(forward):
    class Lambda(nn.Module):
        def forward(self, *args):
            return forward(*args)

    return Lambda()


def resize(t, size):
    return F.interpolate(t, size, mode='bicubic', align_corners=True)


def conv_block(i, o, ks, s, p, a=get_activation(), d=1):
    return nn.Sequential(
        nn.Conv2d(i, o, kernel_size=ks, stride=s, padding=p, dilation=d),
        nn.BatchNorm2d(o),
        a,
    )


def deconv_block(i, o, ks, s, p, a=get_activation(), d=1):
    return nn.Sequential(
        nn.ConvTranspose2d(
            i,
            o,
            kernel_size=ks,
            stride=s,
            padding=p,
            dilation=d,
        ),
        nn.BatchNorm2d(o),
        a,
    )
