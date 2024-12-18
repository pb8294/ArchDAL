import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.gated_layers import GatedLinear, GatedConv2d
from nets.gated_net import GatedNet
from utils.flops import *
from utils.memory import *

class LeNetCaffe(GatedNet):
    def __init__(self):
        super(LeNetCaffe, self).__init__()
        self.conv0 = GatedConv2d(1, 20, 5)
        self.conv1 = GatedConv2d(20, 50, 5)
        self.dense0 = GatedLinear(800, 500)
        self.dense1 = GatedLinear(500, 10)
        self.gated_layers = [self.conv0, self.conv1, self.dense0, self.dense1]
        self.full_flops = self.count_flops([20, 50, 800, 500])
        self.full_mem = self.count_memory([20, 50, 800, 500])

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv0(x)), 2)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.relu(self.dense0(x.view(-1, 800)))
        x = self.dense1(x)
        return x

    def count_flops(self, num_units):
        return count_flops_conv(28, 28, 1, num_units[0], 5) \
                + count_flops_max_pool(24, 24, num_units[0], 2) \
                + count_flops_conv(12, 12, num_units[0], num_units[1], 5) \
                + count_flops_max_pool(8, 8, num_units[1], 2) \
                + count_flops_dense(num_units[2], num_units[3]) \
                + count_flops_dense(num_units[3], 10, activation=False)

    def count_flops_dep(self, num_units, num_units_dep):
        return count_flops_conv(28, 28, 1, num_units[0], 5) \
                + count_flops_conv_dbb(24, 24, num_units[0]) \
                + count_flops_max_pool(24, 24, num_units_dep[0], 2) \
                + count_flops_conv(12, 12, num_units_dep[0], num_units[1], 5) \
                + count_flops_conv_dbb(8, 8, num_units[1]) \
                + count_flops_max_pool(8, 8, num_units_dep[1], 2) \
                + count_flops_dense_dbb(num_units[2]) \
                + count_flops_dense(num_units_dep[2], num_units[3]) \
                + count_flops_dense_dbb(num_units[3]) \
                + count_flops_dense(num_units_dep[3], 10, activation=False)

    def count_memory(self, num_units):
        return count_memory_conv(28, 28, 1, num_units[0], 5) \
                + count_memory_conv(12, 12, num_units[0], num_units[1], 5) \
                + count_memory_dense(num_units[2], num_units[3]) \
                + count_memory_dense(num_units[3], 10)

    def count_memory_dep(self, num_units, num_units_dep):
        return count_memory_conv(28, 28, 1, num_units[0], 5) \
                + count_memory_dbb(num_units[0]) \
                + count_memory_conv(12, 12, num_units_dep[0], num_units[1], 5) \
                + count_memory_dbb(num_units[1]) \
                + count_memory_dbb(num_units[2]) \
                + count_memory_dense(num_units_dep[2], num_units[3]) \
                + count_memory_dbb(num_units[3]) \
                + count_memory_dense(num_units_dep[3], 10)
