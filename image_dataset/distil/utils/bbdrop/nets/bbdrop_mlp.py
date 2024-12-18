import torch.nn.functional as F
from modules.gated_layers import GatedLinear
from nets.gated_net import GatedNet
from torch import nn


class BBDROPMLP(GatedNet):
    def __init__(self, n_layers=5):
        super(BBDROPMLP, self).__init__()
        self.dense0 = GatedLinear(16, 50)

        layers = []
        for i in range(n_layers - 1):
            layers.append(GatedLinear(50, 50))

        self.layers = nn.ModuleList(layers)
        self.out_layer = GatedLinear(50, 1)

        self.gated_layers = [self.dense0, self.out_layer] + layers

    def forward(self, x):
        x = F.relu(self.dense0(x))

        for layer in self.layers:
            x = F.relu(layer(x)) + x

        x = self.out_layer(x)
        return x
