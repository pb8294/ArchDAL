import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.gated_layers import GatedLinear, GatedConv2d
from nets.gated_net import GatedNet
from utils.flops import count_flops_dense, count_flops_dense_dbb
from utils.memory import count_memory_dense, count_memory_dbb

class LeNetMLP(GatedNet):
    def __init__(self):
        super(LeNetMLP, self).__init__()
        self.dense0 = GatedLinear(784, 500)
        self.dense1 = GatedLinear(500, 300)
        self.dense2 = GatedLinear(300, 10)
        self.gated_layers = [self.dense0, self.dense1, self.dense2]
        self.full_flops = self.count_flops([784, 500, 300])
        self.full_mem = self.count_memory([784, 500, 300])

    def forward(self, x):
        x = F.relu(self.dense0(x.view(-1, 784)))
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x

    def count_flops(self, num_units):
        return count_flops_dense(num_units[0], num_units[1]) \
                + count_flops_dense(num_units[1], num_units[2]) \
                + count_flops_dense(num_units[2], 10, activation=False)

    def count_flops_dep(self, num_units, num_units_dep):
        return count_flops_dense_dbb(num_units[0]) \
                + count_flops_dense(num_units_dep[0], num_units[1]) \
                + count_flops_dense_dbb(num_units[1]) \
                + count_flops_dense(num_units_dep[1], num_units[2]) \
                + count_flops_dense_dbb(num_units[2]) \
                + count_flops_dense(num_units_dep[2], 10)

    def count_memory(self, num_units):
        return count_memory_dense(num_units[0], num_units[1]) \
                + count_memory_dense(num_units[1], num_units[2]) \
                + count_memory_dense(num_units[2], 10)

    def count_memory_dep(self, num_units, num_units_dep):
        return count_memory_dbb(num_units[0]) \
                + count_memory_dense(num_units_dep[0], num_units[1]) \
                + count_memory_dbb(num_units[1]) \
                + count_memory_dense(num_units_dep[1], num_units[2]) \
                + count_memory_dbb(num_units[2]) \
                + count_memory_dense(num_units_dep[2], 10)
