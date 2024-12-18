'''ResNet in PyTorch.

Reference
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
'''


import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.in_planes = in_planes
        self.planes = planes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, mask=None):
        # print("\nst", x.shape, self.in_planes, self.planes)
        out = F.relu(self.bn1(self.conv1(x)))
        if mask is not None:
            mask = mask.view(1, mask.shape[0], 1, 1)
            out *= mask
        # print(out.shape)
        out = self.bn2(self.conv2(out))
        # print(out.shape)
        if mask is not None:
            # mask = mask.view(1, mask.shape[0], 1, 1)
            out *= mask
        # print('sk', self.shortcut(x).shape)
        out += self.shortcut(x)
        # print(out.shape)
        out = F.relu(out)
        # print(out.shape); exit(0)
        return out


class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, residual=True, drop=None):
        super(BasicBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.residual = residual
        self.bn1 = nn.BatchNorm2d(planes)
        self.in_planes = in_planes
        self.planes = planes
        self.shortcut = nn.Sequential()
        self.drop = drop
        
        if self.drop != None:
            self.dropact = nn.Dropout2d(p=self.drop)
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, mask=None):
        # print("\nst", x.shape, self.in_planes, self.planes)
        out = self.bn1(self.conv1(x))
        
        # print(out.shape)
        # if self.residual:
        # print("Inside residual")
        out += self.shortcut(x)
        # print(out.shape)
        if self.drop:
            out = self.dropact(F.relu(out))
        else:
            out = F.relu(out)
        
        if mask is not None:
            mask = mask.view(1, mask.shape[0], 1, 1)
            out *= mask
        # print(out.shape)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def forward(self, x):
        # print("\nst", x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = F.relu(self.bn2(self.conv2(out)))
        # print(out.shape)
        out = self.bn3(self.conv3(out))
        # print(out.shape)
        # print('sk', self.shortcut(x).shape)
        out += self.shortcut(x)
        # print(out.shape)
        out = F.relu(out)
        # print(out.shape)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.embDim = 8 * self.in_planes * block.expansion
        
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = F.avg_pool2d(out, 4)
                e = out.view(out.size(0), -1)
        else:
            # print("RN st")
            # print(x.shape)
            out = F.relu(self.bn1(self.conv1(x)))
            # print(out.shape)
            out = self.layer1(out)
            # print(out.shape)
            out = self.layer2(out)
            # print(out.shape)
            out = self.layer3(out)
            # print(out.shape)
            out = self.layer4(out)
            # print(out.shape)
            out = F.avg_pool2d(out, 4)
            # print(out.shape)
            e = out.view(out.size(0), -1)
            # print(e.shape)
        # exit(0)
        out = self.linear(e)
        if last:
            return out, e
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim
    
class ResBlock(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, channels=3, residual=True):
        super(ResBlock, self).__init__()
        self.in_planes = 64
        self.embDim = 8 * self.in_planes * block.expansion
        self.residual = residual
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.linear = nn.Linear(64*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, residual=True, drop=0.5))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = out.mean(dim=(2, 3))
                e = out.view(out.size(0), -1)
        else:
            # print("RN st")
            # print(x.shape)
            out = F.relu(self.bn1(self.conv1(x)))
            # print(out.shape)
            out = self.layer1(out)
            # print(out.shape)
            out = out.mean(dim=(2, 3))
            # print(out.shape)
            e = out.view(out.size(0), -1)
            # print(e.shape); exit(0)
        out = self.linear(e)
        if last:
            return out, e
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim

def ResNet18(num_classes=10, channels=3):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, channels)


def ResNet34(num_classes=10, channels=3):
    return ResNet(BasicBlock, [3,4,6,3], num_classes, channels)


def ResNet50(num_classes=10, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)


def ResNet101(num_classes=10, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)


def ResNet152(num_classes=10, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)

def ResBlock18(num_classes=10, channels=3):
    return ResBlock(BasicBlock1, [1], num_classes, channels)

def ResCNN18(num_classes=10, channels=3):
    return ResBlock(BasicBlock1, [6], num_classes, channels)

#test()
