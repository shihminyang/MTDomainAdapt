import torch
from torch import nn
import torch.nn.functional as F


###############################################################################
# Basic Blocks
###############################################################################
class ResBlocks(nn.Module):
    def __init__(self, dim, m_dim, num_blocks):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [Bottleneck(dim, m_dim)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Bottleneck(nn.Module):
    def __init__(self, dim, m_dim):
        super().__init__()
        model = [nn.Conv2d(dim, m_dim, 1, 1), nn.InstanceNorm2d(m_dim), nn.ReLU(inplace=True),
                 nn.Conv2d(m_dim, m_dim, 3, 1, 1), nn.InstanceNorm2d(m_dim), nn.ReLU(inplace=True),
                 nn.Conv2d(m_dim, dim, 1, 1), nn.InstanceNorm2d(dim)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = self.model(x)
        out = x + residual
        return out


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.interpolate = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


##################################################################################
# Normalization layers
##################################################################################
class LayerNorm(nn.Module):
    """ Normalize along the channel. """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
