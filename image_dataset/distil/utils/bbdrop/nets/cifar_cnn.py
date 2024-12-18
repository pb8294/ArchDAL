import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from distil.utils.bbdrop.modules.gated_layers import GatedLinear, GatedConv2d
from distil.utils.bbdrop.nets.gated_net import GatedNet
from distil.utils.bbdrop.utils.flops import *
from distil.utils.bbdrop.utils.memory import *

class BBDROPCNN(GatedNet):
    def __init__(self, in_channels=1, n_layers=7, filters=256):
        super(BBDROPCNN, self).__init__()
        self.conv0 = GatedConv2d(in_channels, filters, 5)
        self.n_layers, self.filters = n_layers, filters
        layers = []
        self.bns = []
        for i in range(n_layers - 1):
            layers.append(GatedConv2d(filters, filters, 3, 1, 1))

        self.layers = nn.ModuleList(layers)

        self.dense0 = nn.Linear(filters, filters)
        self.dense1 = nn.Linear(filters, 10)
        self.gated_layers = [self.conv0] + layers

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                x = F.max_pool2d(F.relu(self.conv0(x)), 2)
                for i, layer in enumerate(self.layers):
                    x = F.leaky_relu(layer(x)) + x

                x = x.mean(dim=(2, 3))
                e = F.relu(self.dense0(x)) + x
        else:
            x = F.max_pool2d(F.relu(self.conv0(x)), 2)
            for i, layer in enumerate(self.layers):
                x = F.leaky_relu(layer(x)) + x

            x = x.mean(dim=(2, 3))
            e = F.relu(self.dense0(x)) + x
        x = self.dense1(e)
            
        if last:
            return x, e
        else:
            return x
    
    def get_masks(self):
        masks = torch.empty(self.n_layers, self.filters)
        counts = torch.empty(self.n_layers)
        for ii, m in enumerate(self.gated_layers):
            masks[ii] = m.get_mask().detach().cpu()
            counts[ii] = m.get_num_active()
        print(masks, counts)

    def get_embedding_dim(self):
        return self.filters