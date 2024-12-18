from nets.gated_net import GatedNet
from modules.gated_layers import *
import torch.nn.functional as F
from utils.flops import *
from utils.memory import *

class VGG(GatedNet):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.num_classes = num_classes

        def create_block(in_channels, out_channels):
            conv = GatedConv2d(in_channels, out_channels, 3, padding=1)
            bn = nn.BatchNorm2d(out_channels)
            relu = nn.ReLU()
            self.gated_layers.append(conv)
            return nn.Sequential(conv, bn, relu)

        self.block0 = create_block(3, 64)
        self.block1 = create_block(64, 64)

        self.block2 = create_block(64, 128)
        self.block3 = create_block(128, 128)

        self.block4 = create_block(128, 256)
        self.block5 = create_block(256, 256)
        self.block6 = create_block(256, 256)

        self.block7 = create_block(256, 512)
        self.block8 = create_block(512, 512)
        self.block9 = create_block(512, 512)

        self.block10 = create_block(512, 512)
        self.block11 = create_block(512, 512)
        self.block12 = create_block(512, 512)

        dense = GatedLinear(512, 512)
        bn = nn.BatchNorm1d(512)
        relu = nn.ReLU()
        self.gated_layers.append(dense)
        self.block13 = nn.Sequential(dense, bn, relu)

        self.block14 = GatedLinear(512, num_classes)
        self.gated_layers.append(self.block14)

        self.full_flops = self.count_flops([
            64, 64,
            128, 128,
            256, 256, 256,
            512, 512, 512,
            512, 512, 512,
            512, 512])
        self.full_mem = self.count_memory([
            64, 64,
            128, 128,
            256, 256, 256,
            512, 512, 512,
            512, 512, 512,
            512, 512])

    def count_flops(self, num_units):
        flops = count_flops_conv(32, 32, 3, num_units[0], 3, padding=1) \
                + count_flops_conv(32, 32, num_units[0], num_units[1], 3, padding=1) \
                + count_flops_max_pool(32, 32, num_units[1], 2)
        flops += count_flops_conv(16, 16, num_units[1], num_units[2], 3, padding=1) \
                + count_flops_conv(16, 16, num_units[2], num_units[3], 3, padding=1) \
                + count_flops_max_pool(16, 16, num_units[3], 2)
        flops += count_flops_conv(8, 8, num_units[3], num_units[4], 3, padding=1) \
                + count_flops_conv(8, 8, num_units[4], num_units[5], 3, padding=1) \
                + count_flops_conv(8, 8, num_units[5], num_units[6], 3, padding=1) \
                + count_flops_max_pool(8, 8, num_units[6], 2)
        flops += count_flops_conv(4, 4, num_units[6], num_units[7], 3, padding=1) \
                + count_flops_conv(4, 4, num_units[7], num_units[8], 3, padding=1) \
                + count_flops_conv(4, 4, num_units[8], num_units[9], 3, padding=1) \
                + count_flops_max_pool(4, 4, num_units[9], 2)
        flops += count_flops_conv(2, 2, num_units[9], num_units[10], 3, padding=1) \
                + count_flops_conv(2, 2, num_units[10], num_units[11], 3, padding=1) \
                + count_flops_conv(2, 2, num_units[11], num_units[12], 3, padding=1) \
                + count_flops_max_pool(2, 2, num_units[12], 2)
        flops += count_flops_dense(num_units[13], num_units[14])
        flops += count_flops_dense(num_units[14], self.num_classes)
        return flops

    def count_flops_dep(self, num_units, num_units_dep):
        flops = count_flops_conv(32, 32, 3, num_units[0], 3, padding=1) \
                + count_flops_conv_dbb(32, 32, num_units[0]) \
                + count_flops_conv(32, 32, num_units_dep[0], num_units[1], 3, padding=1) \
                + count_flops_conv_dbb(32, 32, num_units[1]) \
                + count_flops_max_pool(32, 32, num_units_dep[1], 2)
        flops += count_flops_conv(16, 16, num_units_dep[1], num_units[2], 3, padding=1) \
                + count_flops_conv_dbb(16, 16, num_units[2]) \
                + count_flops_conv(16, 16, num_units_dep[2], num_units[3], 3, padding=1) \
                + count_flops_conv_dbb(16, 16, num_units[3]) \
                + count_flops_max_pool(16, 16, num_units_dep[3], 2)
        flops += count_flops_conv(8, 8, num_units_dep[3], num_units[4], 3, padding=1) \
                + count_flops_conv_dbb(8, 8, num_units[4]) \
                + count_flops_conv(8, 8, num_units_dep[4], num_units[5], 3, padding=1) \
                + count_flops_conv_dbb(8, 8, num_units[5]) \
                + count_flops_conv(8, 8, num_units_dep[5], num_units[6], 3, padding=1) \
                + count_flops_conv_dbb(8, 8, num_units[6]) \
                + count_flops_max_pool(8, 8, num_units_dep[6], 2)
        flops += count_flops_conv(4, 4, num_units_dep[6], num_units[7], 3, padding=1) \
                + count_flops_conv_dbb(4, 4, num_units[7]) \
                + count_flops_conv(4, 4, num_units_dep[7], num_units[8], 3, padding=1) \
                + count_flops_conv_dbb(4, 4, num_units[8]) \
                + count_flops_conv(4, 4, num_units_dep[8], num_units[9], 3, padding=1) \
                + count_flops_conv_dbb(4, 4, num_units[9]) \
                + count_flops_max_pool(4, 4, num_units_dep[9], 2)
        flops += count_flops_conv(2, 2, num_units_dep[9], num_units[10], 3, padding=1) \
                + count_flops_conv_dbb(2, 2, num_units[10]) \
                + count_flops_conv(2, 2, num_units_dep[10], num_units[11], 3, padding=1) \
                + count_flops_conv_dbb(2, 2, num_units[11]) \
                + count_flops_conv(2, 2, num_units_dep[11], num_units[12], 3, padding=1) \
                + count_flops_conv_dbb(2, 2, num_units[12]) \
                + count_flops_max_pool(2, 2, num_units_dep[12], 2)
        flops += count_flops_dense_dbb(num_units[13]) \
                + count_flops_dense(num_units_dep[13], num_units[14])
        flops += count_flops_dense_dbb(num_units[14]) \
                + count_flops_dense(num_units_dep[14], self.num_classes)
        return flops

    def count_memory(self, num_units):
        mem = 0
        c = [3] + num_units
        h = [32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
        for i in range(13):
            mem += count_memory_conv(h[i], h[i], c[i], c[i+1], 3,
                    padding=1, batch_norm=True)
        mem += count_memory_dense(num_units[13], num_units[14], batch_norm=True)
        mem += count_memory_dense(num_units[14], self.num_classes)
        return mem

    def count_memory_dep(self, num_units, num_units_dep):
        c = [3] + num_units
        c_dep = num_units_dep
        h = [32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
        mem = count_memory_conv(h[0], h[0], 3, num_units[0], 3,
                padding=1, batch_norm=True)
        mem += count_memory_dbb(num_units[0])
        for i in range(1, 13):
            mem += count_memory_conv(h[i], h[i],
                    num_units_dep[i], num_units[i+1], 3,
                    padding=1, batch_norm=True)
            mem += count_memory_dbb(num_units[i+1])

        mem += count_memory_dbb(num_units[13]) \
                + count_memory_dense(num_units_dep[13], num_units[14],
                        batch_norm=True)
        mem += count_memory_dbb(num_units[13]) \
                + count_memory_dense(num_units_dep[14], self.num_classes)
        return mem

    def forward(self, x):
        def _dropout(x, p):
            return (x if self.use_gate else \
                    F.dropout(x, p=p, training=self.training))
        x = _dropout(self.block0(x), 0.3)
        x = F.max_pool2d(self.block1(x), 2)

        x = _dropout(self.block2(x), 0.4)
        x = F.max_pool2d(self.block3(x), 2)

        x = _dropout(self.block4(x), 0.4)
        x = _dropout(self.block5(x), 0.4)
        x = F.max_pool2d(self.block6(x), 2)

        x = _dropout(self.block7(x), 0.4)
        x = _dropout(self.block8(x), 0.4)
        x = F.max_pool2d(self.block9(x), 2)

        x = _dropout(self.block10(x), 0.4)
        x = _dropout(self.block11(x), 0.4)
        x = F.max_pool2d(self.block12(x), 2)

        x = x.view(-1, 512)
        x = self.block13(_dropout(x, 0.5))
        x = self.block14(_dropout(x, 0.5))
        return x
