from torch import nn as nn
import torch.nn.functional as F

def conv3x3(in_channels: int, out_channels: int, kernel_size=3, stride: int = 1, padding: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1, padding: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=False)

def transconv3x3(in_channels: int, out_channels: int, kernel_size=3, stride: int = 1, padding: int = 1):
    """3x3 transpose convolution with padding"""
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

class global_mean_pool(nn.Module):
    """
    Global averaging of the image
    """

    def __init__(self):
        super(global_mean_pool, self).__init__()

    def forward(self, x):
        """
        Computes global average of image
        Parameters
        ----------
        x : input feature map (Batch_size * Channels * Width * Height)

        Returns
        -------
        feature vector (Batch_size * Channels)
        """
        return x.mean(dim=(2, 3))

# class Swish(nn.Module):
#     """https://arxiv.org/abs/1710.05941"""
#     def forward(self, x):
#         return x * torch.sigmoid(x)

class MLPBlock(nn.Module):
    """
    Mask the output of fully connected layer with binary vectors from Beta-Bernoulli prior and add residual
    """

    def __init__(self, in_neurons, out_neurons, residual=False):
        super(MLPBlock, self).__init__()

        self.linear = nn.Linear(in_neurons, out_neurons)
        self.act_layer_fn = nn.LeakyReLU()
        # self.norm_layer = nn.BatchNorm1d(out_neurons)
        self.norm_layer = nn.GroupNorm(1, out_neurons) # Old Version Matching
        self.residual = residual

    def forward(self, x, mask=None):
        """
        Transforms x, applies mask and add residual
        Parameters
        ----------
        x : input feature matrix (Batch_size * out_neurons)
        mask : binary vector to mask the feature matrix (out_neurons)

        Returns
        -------
        Output of the layer (Batch_size * out_neurons)
        """

        residual = x
        # print(x.shape)
        out = self.act_layer_fn(self.linear(x))
        out = self.norm_layer(out)

        if mask is not None:
            out *= mask.view(1, -1)

        if self.residual:
            out  = out + residual

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

      
            
class RNNBlock(nn.Module):
    expansion = 1
    """
    Conv Block for RESNET
    """
    def __init__(self, in_planes, planes, stride=1):
        super(RNNBlock, self).__init__()
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
        out = F.relu(self.bn1(self.conv1(x)))
        if mask is not None:
            mask = mask.view(1, mask.shape[0], 1, 1)
            out *= mask
            # print("mask1", mask.shape, out.shape)
            
        out = self.bn2(self.conv2(out))
        # print("\nHere", out.shape, mask.shape)
        if mask is not None:
            # mask = mask.view(1, mask.shape[0], 1, 1)
            out *= mask
            # print("mask2", mask.shape, out.shape)
        out += self.shortcut(x)
        out = F.relu(out)
        # print("out", out.shape)
        return out  

class ConvBlock(nn.Module):
    """
    Mask the output of CNN with binary vectors from Beta-Bernoulli prior and add residual
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=False, residual=False, drop=None, norm=True):
        super(ConvBlock, self).__init__()

        self.conv_layer = conv3x3(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        # self.act_layer = nn.ReLU()
        # self.norm_layer = nn.BatchNorm2d(out_channels)
        self.act_layer = nn.LeakyReLU() #Old Version Matching
        self.norm_layer = nn.GroupNorm(1, out_channels) #Old Version Matching
        
        self.pool = pool
        self.drop = drop
        self.norm = norm
        
        if self.drop != None:
            self.dropact = nn.Dropout2d(p=self.drop)
        
        if pool:
            self.pool_layer = nn.AvgPool2d(2, 2)

        self.residual = residual

    def forward(self, x, mask=None):
        """
        Transforms feature matrix x, applies mask and add residual
        Parameters
        ----------
        x : input features (Batch_size * out_channels * width * height)
        mask : binary vector to mask the feature matrix (out_channels)

        Returns
        -------
        If pool is false:
            Output of the layer (Batch_size * out_channels * width * height)
        else:
            Output of the layer (Batch_size * out_channels * width' * height')
        """
        residual = x

        # output = self.norm_layer(self.conv_layer(x))
        # output = self.act_layer(output)
        
        # Old Version Matching
        
        if self.drop == None:
            if self.norm:
                output = self.norm_layer(self.act_layer(self.conv_layer(x)))
            else:
                output = self.act_layer(self.conv_layer(x))
        else:
            if self.norm:
                output = self.norm_layer(self.dropact(self.act_layer(self.conv_layer(x))))
            else:
                output = self.dropact(self.act_layer(self.conv_layer(x)))
            # output = self.bn(self.dropact(self.act(self.conv_layer(x))))

        if self.pool:
            output = self.pool_layer(output)

        if mask is not None:
            mask = mask.view(1, mask.shape[0], 1, 1)
            output = output * mask

        if self.residual:
            output += residual

        return output
    
class ConvBlockRN(nn.Module):
    """
    Conv Block for RESNET
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=False, residual=False, act=False):
        super(ConvBlockRN, self).__init__()
        self.conv_layer = conv3x3(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.act = act
        if act:
            self.act_layer = nn.ReLU()
        self.norm_layer = nn.BatchNorm2d(out_channels)
        self.pool = pool
        if pool:
            self.pool_layer = nn.AvgPool2d(2, 2)

        self.residual = residual

    def forward(self, x, mask=None):
        residual = x
        out = self.conv_layer(x)

        if self.act:
            out = self.act_layer(out)
        out = self.norm_layer(out)

        if self.pool:
            out = self.pool_layer(out)

        if mask is not None:
            mask = mask.view(1, mask.shape[0], 1, 1)
            out *= mask

        if self.residual:
            out += residual

        return out
    
class TransConvBlock(nn.Module):
    """
    Mask the output of CNN with binary vectors from Beta-Bernoulli prior and add residual
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=False, residual=False, norm=False):
        super(TransConvBlock, self).__init__()

        self.conv_layer = transconv3x3(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        # self.act = nn.LeakyReLU()
        # self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = pool
        self.norm = norm
        if pool:
            self.pool_layer = nn.MaxUnpool2d(2, stride=2)

        self.residual = residual

        self.downsample = False
        if out_channels != in_channels and residual:
            self.downsample = True
            self.downsample_conv_layer = conv1x1(in_channels, out_channels, stride=2, padding=padding)
            self.downsample_norm_layer = nn.BatchNorm2d(out_channels)

    def forward(self, x, mask=None):
        if self.norm:
            output = self.bn(self.conv_layer(x))
        else:
            output = self.conv_layer(x)

        if self.pool:
            output = self.pool_layer(output)

        if mask is not None:
            mask = mask.view(1, mask.shape[0], 1, 1)
            output *= mask

        if self.residual:
            if self.downsample:
                residual = self.downsample_norm_layer(self.downsample_conv_layer(x))
                output += residual
            else:
                output += x

        return output