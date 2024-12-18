import torch
from torch import nn
import torch.nn.functional as F
from utils.layers.layers import MLPBlock, ConvBlock, ConvBlockRN, RNNBlock
from distil.utils.models.resnet import BasicBlock, BasicBlock1
from utils.models.NetworkStructureSampler import NetworkStructureSampler, SampleNetworkArchitecture

        
        

class AdaptiveBlock(nn.Module):
    def __init__(self, input_channels, num_classes, num_channels=64, kernel_size=3, args=None, device=None, fixed=0):
        super(AdaptiveBlock, self).__init__()
        self.mode = "NN"
        self.args = args

        self.max_channels = num_channels
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.truncation_level = args.truncation_level
        self.num_samples = args.num_samples
        self.device = device
        self.out_layer_dim = 512
        self.embDim = 8 * 64 * 1
        

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.structure_sampler_1 = NetworkStructureSampler(args, self.device)
        self.fixed = fixed
        if fixed == 1:
            self.layer1_c = BasicBlock(64, 64, stride=1).to(self.device)
            self.layer2_c = BasicBlock(64, 64, stride=1).to(self.device)
        elif fixed == 0:
            self.layers_1 = nn.ModuleList([])
            for i in range(0, self.truncation_level):
                self.layers_1.append(BasicBlock(64, 64).to(self.device))

        self.linear1 = nn.Linear(64 * 1, 10)
        # self.out_layer = nn.Sequential(MLPBlock(self.out_layer_dim, self.out_layer_dim, residual=True), nn.Linear(self.out_layer_dim, self.num_classes))
    
    def _forward(self, x, mask_matrix, threshold, layers):
        """
        Transform the input feature matrix with a sampled network structure
        Parameters
        ----------
        x : input feature matrix
        mask_matrix : mask matrix from Beta-Bernoulli Process
        threshold : number of layers in sampled structure

        Returns
        -------
        out : Output from the sampled architecture
        """

        # if number of layers sampled during testing is greater than truncation level, we use the network with
        # truncation level.
        if not self.training and threshold > len(layers):
            threshold = len(layers)

        for layer in range(threshold):
            mask = mask_matrix[:, layer]
            # print(mask.shape, x.shape)
            x = layers[layer](x, mask)

        return x
        

    def forward(self, x, num_samples=2, last=False, freeze=False):
        """
        Transforms the input feature matrix with different samples of network structures Z

        Parameters
        ----------
        x : data
        num_samples : Number of samples of network structure Z

        Returns
        -------
        act_vec : Tensor by stacking the output from different structures Z
        """
        # print("i", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print("c1", x.shape)
        if self.fixed == 1:
            x = self.layer1_c(x)
            # print("l1c", x.shape)
            x = self.layer2_c(x)
            # print("l2c", x.shape)
        elif self.fixed == 0:
            act_vec = []
            Z, threshold = self.structure_sampler_1(num_samples)
            print(Z, threshold); exit()
            # print(Z.shape, threshold)
            for s in range(num_samples):
                out = self._forward(x, Z[s], threshold, self.layers_1)
                act_vec.append(out.unsqueeze(0))
            act_vec = torch.cat(act_vec, dim=0)
            # print("cat", act_vec.shape)
            act_vec = torch.mean(act_vec, dim=0)
            # print("m", act_vec.shape)
        e = x.mean(dim=(2, 3))
        # print("m", e.shape)
        out = self.linear1(e)
        # print("out", out.shape); exit()
        if last:
            return out, e
        else:
            return out
        # return act_vec

    def get_E_loglike(self, neg_loglike_fun, output, target):
        """

        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        output : scores predicted by model
        target : real labels

        Returns
        -------
        mean negative log likelihood of the model based on different architectures
        """
        neg_loglike = neg_loglike_fun(output, target)
        return neg_loglike
        # E_neg_loglike = neg_loglike.mean(0).mean()
        # return E_neg_loglike
        num_samples = self.num_samples
        # print(output.shape, target.shape, self.num_samples)
        batch_sze = target.shape[1]
        target_expand = target.repeat(num_samples)
        # print(output.shape, target_expand.shape, self.num_samples); exit(0)
        output = output.view(num_samples * batch_sze, -1)
        print(output.shape, target_expand.shape, target.shape); exit()
        

    def estimate_ELBO(self, neg_loglike_fun, act_vec, y, N_train, kl_weight=1):
        """
        Estimate the ELBO
        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        act_vec : scores from different samples of latent neural architecture Z
        y : real labels
        N_train: number of training data points
        kl_weight: coefficient to scale KL component

        Returns
        -------
        ELBO
        """
        E_loglike = self.get_E_loglike(neg_loglike_fun, act_vec, y)
        KL1 = self.structure_sampler_1.get_kl()
        # print(act_vec.shape, y.shape, KL1.shape); exit()
        ELBO = E_loglike + (kl_weight * KL1)  
        return ELBO, KL1, (kl_weight * KL1)  , E_loglike
    
    def get_embedding_dim(self):
        return self.embDim

class AdaptiveCNNBlock(nn.Module):
    def __init__(self, input_channels, num_classes, num_channels=64, kernel_size=3, args=None, device=None, fixed=0):
        super(AdaptiveCNNBlock, self).__init__()
        self.mode = "NN"
        self.args = args

        self.max_channels = num_channels
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.truncation_level = args.truncation_level
        self.num_samples = args.num_samples
        self.device = device
        self.out_layer_dim = 512
        self.embDim = 8 * num_channels * 1
        self.fixed = fixed
        

        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.layer1_c = BasicBlock1(64, 64, stride=1).to(self.device)
        self.structure_sampler = NetworkStructureSampler(args, self.device)
        self.layers_1 = nn.ModuleList([])
        if self.fixed == 1:
            # self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # self.bn1 = nn.BatchNorm2d(64)
            for i in range(self.truncation_level):
                self.layers_1.append(BasicBlock1(64, 64, stride=1, drop=0.5).to(device))
        elif self.fixed == 0:
            # self.layers_1 = nn.ModuleList([])
            for i in range(self.truncation_level):
                self.layers_1.append(BasicBlock1(64, 64, stride=1).to(device))

        # Previous code has another linear layer with relu activation and residual
        # followed by the following linear layer
        self.linear1 = nn.Linear(num_channels * 1, 10)
        # self.out_layer = nn.Sequential(MLPBlock(self.out_layer_dim, self.out_layer_dim, residual=True), nn.Linear(self.out_layer_dim, self.num_classes))
    
    def _forward(self, x, mask_matrix, threshold, layers):
        """
        Transform the input feature matrix with a sampled network structure
        Parameters
        ----------
        x : input feature matrix
        mask_matrix : mask matrix from Beta-Bernoulli Process
        threshold : number of layers in sampled structure

        Returns
        -------
        out : Output from the sampled architecture
        """

        # if number of layers sampled during testing is greater than truncation level, we use the network with
        # truncation level.
        if not self.training and threshold > len(layers):
            threshold = len(layers)

        for layer in range(threshold):
            mask = mask_matrix[:, layer]
            # print(mask.shape, x.shape)
            x = layers[layer](x, mask)

        return x
        

    def forward(self, x, num_samples=4, last=False, freeze=False):
        """
        Transforms the input feature matrix with different samples of network structures Z

        Parameters
        ----------
        x : data
        num_samples : Number of samples of network structure Z

        Returns
        -------
        act_vec : Tensor by stacking the output from different structures Z
        """
        # print("i", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print("c1", x.shape)
        # x = self.layer1_c(x)
        if self.fixed == 1:
            # x = F.relu(self.bn1(self.conv1(x)))
            # print("c1", x.shape)
            for l in self.layers_1:
                x = l(x)
            # print("l1c", x.shape)
            act_vec = x
            # print("l2c", x.shape)
        elif self.fixed == 0:
            act_vec = []
            Z, threshold = self.structure_sampler(num_samples)
            # print(Z.shape, threshold)
            for s in range(num_samples):
                out = self._forward(x, Z[s], threshold, self.layers_1)
                act_vec.append(out.unsqueeze(0))
            act_vec = torch.cat(act_vec, dim=0)
            # print("cat", act_vec.shape)
            act_vec = torch.mean(act_vec, dim=0)
            # print("m", act_vec.shape)
        e = act_vec.mean(dim=(2, 3))
        # print("m", e.shape)
        out = self.linear1(e)
        # print("out", out.shape); exit()
        if last:
            return out, e
        else:
            return out
        # return act_vec

    def get_E_loglike(self, neg_loglike_fun, output, target):
        """

        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        output : scores predicted by model
        target : real labels

        Returns
        -------
        mean negative log likelihood of the model based on different architectures
        """
        neg_loglike = neg_loglike_fun(output, target)
        return neg_loglike
        # E_neg_loglike = neg_loglike.mean(0).mean()
        # return E_neg_loglike
        num_samples = self.num_samples
        # print(output.shape, target.shape, self.num_samples)
        batch_sze = target.shape[1]
        target_expand = target.repeat(num_samples)
        # print(output.shape, target_expand.shape, self.num_samples); exit(0)
        output = output.view(num_samples * batch_sze, -1)
        print(output.shape, target_expand.shape, target.shape); exit()
        

    def estimate_ELBO(self, neg_loglike_fun, act_vec, y, N_train, kl_weight=1):
        """
        Estimate the ELBO
        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        act_vec : scores from different samples of latent neural architecture Z
        y : real labels
        N_train: number of training data points
        kl_weight: coefficient to scale KL component

        Returns
        -------
        ELBO
        """
        E_loglike = self.get_E_loglike(neg_loglike_fun, act_vec, y)
        KL1 = self.structure_sampler.get_kl()
        ELBO = E_loglike + (kl_weight * KL1)
        return ELBO, KL1, (kl_weight * KL1) , E_loglike
    
    def get_embedding_dim(self):
        return self.embDim


class AdaptiveRNBaseCNN(nn.Module):
    def __init__(self, input_channels, num_classes, num_channels=64, kernel_size=3, args=None, device=None):
        super(AdaptiveRNBaseCNN, self).__init__()
        self.mode = "NN"
        self.args = args

        self.max_channels = num_channels
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.truncation_level = args.truncation_level
        self.num_samples = args.num_samples
        self.device = device
        self.out_layer_dim = 512
        self.embDim = 8 * 64 * 1
        

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.structure_sampler_1 = NetworkStructureSampler(args, self.device)
        self.layer1_c = RNNBlock(64, 64, stride=1).to(self.device)
        self.layers_1 = nn.ModuleList([])
        for i in range(0, self.truncation_level):
            self.layers_1.append(ConvBlockRN(64, 64, 3, 1, 1, False, False, False).to(device))
        
        args.max_width=128
        self.block2 = nn.Sequential(RNNBlock(args.max_width // 2, args.max_width, 2).to(device), 
                                    RNNBlock(args.max_width, args.max_width, 1).to(device))
        
        
        args.max_width=256
        self.block3 = nn.Sequential(RNNBlock(args.max_width // 2, args.max_width, 2).to(device), 
                                    RNNBlock(args.max_width, args.max_width, 1).to(device))
        
        args.max_width=512
        self.block4 = nn.Sequential(RNNBlock(args.max_width // 2, args.max_width, 2).to(device), 
                                    RNNBlock(args.max_width, args.max_width, 1).to(device))

        self.linear1 = nn.Linear(512 * 1, 10)
        # self.out_layer = nn.Sequential(MLPBlock(self.out_layer_dim, self.out_layer_dim, residual=True), nn.Linear(self.out_layer_dim, self.num_classes))
    
    def _forward(self, x, mask_matrix, threshold, layers):
        """
        Transform the input feature matrix with a sampled network structure
        Parameters
        ----------
        x : input feature matrix
        mask_matrix : mask matrix from Beta-Bernoulli Process
        threshold : number of layers in sampled structure

        Returns
        -------
        out : Output from the sampled architecture
        """

        # if number of layers sampled during testing is greater than truncation level, we use the network with
        # truncation level.
        if not self.training and threshold > len(layers):
            threshold = len(layers)

        for layer in range(threshold):
            mask = mask_matrix[:, layer]
            # print(mask.shape, x.shape)
            x = layers[layer](x, mask)

        return x
        

    def forward(self, x, num_samples=2, last=False, freeze=False):
        """
        Transforms the input feature matrix with different samples of network structures Z

        Parameters
        ----------
        x : data
        num_samples : Number of samples of network structure Z

        Returns
        -------
        act_vec : Tensor by stacking the output from different structures Z
        """
        # print("i", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print("c1", x.shape)
        x = self.layer1_c(x)
        # print("l1c", x.shape)
        act_vec = []
        Z, threshold = self.structure_sampler_1(num_samples)
        # print(Z.shape, threshold)
        for s in range(num_samples):
            out = self._forward(x, Z[s], threshold, self.layers_1)
            act_vec.append(out.unsqueeze(0))
        act_vec = torch.cat(act_vec, dim=0)
        # print("cat", act_vec.shape)
        act_vec = torch.mean(act_vec, dim=0)
        # print("m", act_vec.shape)
        act_vec = self.block2(act_vec)
        # print("b2", act_vec.shape)
        act_vec = self.block3(act_vec)
        # print("b3", act_vec.shape)
        act_vec = self.block4(act_vec)
        # print("b4", act_vec.shape)
        
        out = F.avg_pool2d(act_vec, 4)
        # print("pool", out.shape)
        e = out.view(out.size(0), -1)
        # print("embedding", e.shape)
        out = self.linear1(e)
        # print("out", out.shape); exit()
        if last:
            return out, e
        else:
            return out
        # return act_vec

    def get_E_loglike(self, neg_loglike_fun, output, target):
        """

        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        output : scores predicted by model
        target : real labels

        Returns
        -------
        mean negative log likelihood of the model based on different architectures
        """
        neg_loglike = neg_loglike_fun(output, target)
        return neg_loglike
        # E_neg_loglike = neg_loglike.mean(0).mean()
        # return E_neg_loglike
        num_samples = self.num_samples
        # print(output.shape, target.shape, self.num_samples)
        batch_sze = target.shape[1]
        target_expand = target.repeat(num_samples)
        # print(output.shape, target_expand.shape, self.num_samples); exit(0)
        output = output.view(num_samples * batch_sze, -1)
        print(output.shape, target_expand.shape, target.shape); exit()
        

    def estimate_ELBO(self, neg_loglike_fun, act_vec, y, N_train, kl_weight=1):
        """
        Estimate the ELBO
        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        act_vec : scores from different samples of latent neural architecture Z
        y : real labels
        N_train: number of training data points
        kl_weight: coefficient to scale KL component

        Returns
        -------
        ELBO
        """
        E_loglike = self.get_E_loglike(neg_loglike_fun, act_vec, y)
        KL1 = self.structure_sampler_1.get_kl()
        ELBO = E_loglike + (kl_weight * KL1)/N_train
        return ELBO, KL1, kl_weight, E_loglike
    
    def get_embedding_dim(self):
        return self.embDim

class AdaptiveRNBase(nn.Module):
    def __init__(self, input_channels, num_classes, num_channels=64, kernel_size=3, args=None, device=None):
        super(AdaptiveRNBase, self).__init__()
        self.mode = "NN"
        self.args = args

        self.max_channels = num_channels
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.truncation_level = args.truncation_level
        self.num_samples = args.num_samples
        self.device = device
        self.out_layer_dim = 512
        self.embDim = 8 * 64 * 1
        

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.structure_sampler_1 = NetworkStructureSampler(args, self.device)
        self.layer1_c = RNNBlock(64, 64, stride=1).to(self.device)
        self.layers_1 = nn.ModuleList([])
        for i in range(0, self.truncation_level):
            self.layers_1.append(RNNBlock(64, 64).to(self.device))
        
        args.max_width=128
        self.block2 = nn.Sequential(RNNBlock(args.max_width // 2, args.max_width, 2).to(device), 
                                    RNNBlock(args.max_width, args.max_width, 1).to(device))
        
        
        args.max_width=256
        self.block3 = nn.Sequential(RNNBlock(args.max_width // 2, args.max_width, 2).to(device), 
                                    RNNBlock(args.max_width, args.max_width, 1).to(device))
        
        args.max_width=512
        self.block4 = nn.Sequential(RNNBlock(args.max_width // 2, args.max_width, 2).to(device), 
                                    RNNBlock(args.max_width, args.max_width, 1).to(device))

        self.linear1 = nn.Linear(512 * 1, 10)
        # self.out_layer = nn.Sequential(MLPBlock(self.out_layer_dim, self.out_layer_dim, residual=True), nn.Linear(self.out_layer_dim, self.num_classes))
    
    def _forward(self, x, mask_matrix, threshold, layers):
        """
        Transform the input feature matrix with a sampled network structure
        Parameters
        ----------
        x : input feature matrix
        mask_matrix : mask matrix from Beta-Bernoulli Process
        threshold : number of layers in sampled structure

        Returns
        -------
        out : Output from the sampled architecture
        """

        # if number of layers sampled during testing is greater than truncation level, we use the network with
        # truncation level.
        if not self.training and threshold > len(layers):
            threshold = len(layers)

        for layer in range(threshold):
            mask = mask_matrix[:, layer]
            # print(mask.shape, x.shape)
            x = layers[layer](x, mask)

        return x
        

    def forward(self, x, num_samples=2, last=False, freeze=False):
        """
        Transforms the input feature matrix with different samples of network structures Z

        Parameters
        ----------
        x : data
        num_samples : Number of samples of network structure Z

        Returns
        -------
        act_vec : Tensor by stacking the output from different structures Z
        """
        # print("i", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print("c1", x.shape)
        x = self.layer1_c(x)
        # print("l1c", x.shape)
        act_vec = []
        Z, threshold = self.structure_sampler_1(num_samples)
        # print(Z.shape, threshold)
        for s in range(num_samples):
            out = self._forward(x, Z[s], threshold, self.layers_1)
            act_vec.append(out.unsqueeze(0))
        act_vec = torch.cat(act_vec, dim=0)
        # print("cat", act_vec.shape)
        act_vec = torch.mean(act_vec, dim=0)
        # print("m", act_vec.shape)
        act_vec = self.block2(act_vec)
        # print("b2", act_vec.shape)
        act_vec = self.block3(act_vec)
        # print("b3", act_vec.shape)
        act_vec = self.block4(act_vec)
        # print("b4", act_vec.shape)
        
        out = F.avg_pool2d(act_vec, 4)
        # print("pool", out.shape)
        e = out.view(out.size(0), -1)
        # print("embedding", e.shape)
        out = self.linear1(e)
        # print("out", out.shape)
        # exit()
        if last:
            return out, e
        else:
            return out
        # return act_vec

    def get_E_loglike(self, neg_loglike_fun, output, target):
        """

        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        output : scores predicted by model
        target : real labels

        Returns
        -------
        mean negative log likelihood of the model based on different architectures
        """
        neg_loglike = neg_loglike_fun(output, target)
        return neg_loglike
        # E_neg_loglike = neg_loglike.mean(0).mean()
        # return E_neg_loglike
        num_samples = self.num_samples
        # print(output.shape, target.shape, self.num_samples)
        batch_sze = target.shape[1]
        target_expand = target.repeat(num_samples)
        # print(output.shape, target_expand.shape, self.num_samples); exit(0)
        output = output.view(num_samples * batch_sze, -1)
        print(output.shape, target_expand.shape, target.shape); exit()
        

    def estimate_ELBO(self, neg_loglike_fun, act_vec, y, N_train, kl_weight=1):
        """
        Estimate the ELBO
        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        act_vec : scores from different samples of latent neural architecture Z
        y : real labels
        N_train: number of training data points
        kl_weight: coefficient to scale KL component

        Returns
        -------
        ELBO
        """
        E_loglike = self.get_E_loglike(neg_loglike_fun, act_vec, y)
        KL1 = self.structure_sampler_1.get_kl()
        ELBO = E_loglike + (kl_weight * KL1)/N_train
        return ELBO, KL1
    
    def get_embedding_dim(self):
        return self.embDim

class AdaptiveRN(nn.Module):
    def __init__(self, input_channels, num_classes, num_channels=64, kernel_size=3, args=None, device=None):
        super(AdaptiveRN, self).__init__()
        self.mode = "NN"
        self.args = args

        self.max_channels = num_channels
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.truncation_level = args.truncation_level
        self.num_samples = args.num_samples
        self.device = device
        self.out_layer_dim = 512
        self.embDim = 8 * 64 * 1
        

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.structure_sampler = NetworkStructureSampler(args, self.device)
        self.layer1_c = RNNBlock(64, 64, stride=1).to(self.device)
        self.layers_1 = nn.ModuleList([])
        for i in range(0, self.truncation_level):
            self.layers_1.append(RNNBlock(64, 64).to(self.device))
        
        # args.max_width=128
        # self.structure_sampler_2 = NetworkStructureSampler(args, self.device)
        # self.layer2_c = RNNBlock(64, args.max_width, 2).to(device)
        # self.layers_2 = nn.ModuleList([])
        # for i in range(0, self.truncation_level):
        #     self.layers_2.append(RNNBlock(args.max_width, args.max_width).to(self.device))
            
        # args.max_width=256
        # self.structure_sampler_3 = NetworkStructureSampler(args, self.device)
        # self.layer3_c = RNNBlock(128, args.max_width, 2).to(device)
        # self.layers_3 = nn.ModuleList([])
        # for i in range(0, self.truncation_level):
        #     self.layers_3.append(RNNBlock(args.max_width, args.max_width).to(self.device))
            
        # args.max_width=512
        # self.structure_sampler_4 = NetworkStructureSampler(args, self.device)
        # self.layer4_c = RNNBlock(256, args.max_width, 2).to(device)
        # self.layers_4 = nn.ModuleList([])
        # for i in range(0, self.truncation_level):
        #     self.layers_4.append(RNNBlock(args.max_width, args.max_width).to(self.device))

        self.linear1 = nn.Linear(3136, 10)
        # self.out_layer = nn.Sequential(MLPBlock(self.out_layer_dim, self.out_layer_dim, residual=True), nn.Linear(self.out_layer_dim, self.num_classes))
    
    def _forward(self, x, mask_matrix, threshold, layers):
        """
        Transform the input feature matrix with a sampled network structure
        Parameters
        ----------
        x : input feature matrix
        mask_matrix : mask matrix from Beta-Bernoulli Process
        threshold : number of layers in sampled structure

        Returns
        -------
        out : Output from the sampled architecture
        """

        # if number of layers sampled during testing is greater than truncation level, we use the network with
        # truncation level.
        if not self.training and threshold > len(layers):
            threshold = len(layers)

        for layer in range(threshold):
            mask = mask_matrix[:, layer]
            # print(mask.shape, x.shape)
            x = layers[layer](x, mask)

        return x
        

    def forward(self, x, num_samples=2, last=False, freeze=False):
        """
        Transforms the input feature matrix with different samples of network structures Z

        Parameters
        ----------
        x : data
        num_samples : Number of samples of network structure Z

        Returns
        -------
        act_vec : Tensor by stacking the output from different structures Z
        """
        if freeze:
            with torch.no_grad():
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1_c(x)
                act_vec = []
                Z, threshold = self.structure_sampler(num_samples)
                # print(Z.shape, threshold)
                for s in range(num_samples):
                    out = self._forward(x, Z[s], threshold, self.layers_1)
                    act_vec.append(out.unsqueeze(0))
                act_vec = torch.cat(act_vec, dim=0)
                act_vec = torch.mean(act_vec, dim=0)
                # print("1", act_vec.shape)
                
                # act_vec2 = []
                # act_vec = self.layer2_c(act_vec)
                # Z, threshold = self.structure_sampler_2(num_samples)
                # # print(Z.shape, threshold)
                # for s in range(num_samples):
                #     out = self._forward(act_vec, Z[s], threshold, self.layers_2)
                #     act_vec2.append(out.unsqueeze(0))
                # act_vec2 = torch.cat(act_vec2, dim=0)
                # act_vec2 = torch.mean(act_vec2, dim=0)
                # # print("2", act_vec2.shape)
                
                # act_vec3 = []
                # act_vec2 = self.layer3_c(act_vec2)
                # Z, threshold = self.structure_sampler_3(num_samples)
                # # print(Z.shape, threshold)
                # for s in range(num_samples):
                #     out = self._forward(act_vec2, Z[s], threshold, self.layers_3)
                #     act_vec3.append(out.unsqueeze(0))
                # act_vec3 = torch.cat(act_vec3, dim=0)
                # act_vec3 = torch.mean(act_vec3, dim=0)
                # # print("3", act_vec3.shape)
                
                # act_vec4 = []
                # act_vec3 = self.layer4_c(act_vec3)
                # Z, threshold = self.structure_sampler_4(num_samples)
                # # print(Z.shape, threshold)
                # for s in range(num_samples):
                #     out = self._forward(act_vec3, Z[s], threshold, self.layers_4)
                #     act_vec4.append(out.unsqueeze(0))
                # act_vec4 = torch.cat(act_vec4, dim=0)
                # act_vec4 = torch.mean(act_vec4, dim=0)
                # print("4", act_vec4.shape)
                
                out = F.avg_pool2d(act_vec, 4)
                e = out.view(out.size(0), -1)
                
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1_c(x)
            act_vec = []
            Z, threshold = self.structure_sampler(num_samples)
            # print(Z.shape, threshold)
            for s in range(num_samples):
                out = self._forward(x, Z[s], threshold, self.layers_1)
                act_vec.append(out.unsqueeze(0))
            act_vec = torch.cat(act_vec, dim=0)
            act_vec = torch.mean(act_vec, dim=0)
            # print("1", act_vec.shape)
            
            # act_vec2 = []
            # act_vec = self.layer2_c(act_vec)
            # Z, threshold = self.structure_sampler_2(num_samples)
            # # print(Z.shape, threshold)
            # for s in range(num_samples):
            #     out = self._forward(act_vec, Z[s], threshold, self.layers_2)
            #     act_vec2.append(out.unsqueeze(0))
            # act_vec2 = torch.cat(act_vec2, dim=0)
            # act_vec2 = torch.mean(act_vec2, dim=0)
            # # print("2", act_vec2.shape)
            
            # act_vec3 = []
            # act_vec2 = self.layer3_c(act_vec2)
            # Z, threshold = self.structure_sampler_3(num_samples)
            # # print(Z.shape, threshold)
            # for s in range(num_samples):
            #     out = self._forward(act_vec2, Z[s], threshold, self.layers_3)
            #     act_vec3.append(out.unsqueeze(0))
            # act_vec3 = torch.cat(act_vec3, dim=0)
            # act_vec3 = torch.mean(act_vec3, dim=0)
            # # print("3", act_vec3.shape)
            
            # act_vec4 = []
            # act_vec3 = self.layer4_c(act_vec3)
            # Z, threshold = self.structure_sampler_4(num_samples)
            # # print(Z.shape, threshold)
            # for s in range(num_samples):
            #     out = self._forward(act_vec3, Z[s], threshold, self.layers_4)
            #     act_vec4.append(out.unsqueeze(0))
            # act_vec4 = torch.cat(act_vec4, dim=0)
            # act_vec4 = torch.mean(act_vec4, dim=0)
            # print("4", act_vec4.shape)
            
            out = F.avg_pool2d(act_vec, 4)
            # print(out.shape)
            e = out.view(out.size(0), -1)
            # print(e.shape)
        
        # e = act_vec4.view(act_vec4.size(0), -1)
        # act_vec4 = act_vec4.mean(dim=(2, 3))
        # print("4m", act_vec4.shape); exit()
        out = self.linear1(e)
        if last:
            return out, e
        else:
            return out
        # return act_vec

    def get_E_loglike(self, neg_loglike_fun, output, target):
        """

        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        output : scores predicted by model
        target : real labels

        Returns
        -------
        mean negative log likelihood of the model based on different architectures
        """
        neg_loglike = neg_loglike_fun(output, target)
        return neg_loglike
        # E_neg_loglike = neg_loglike.mean(0).mean()
        # return E_neg_loglike
        num_samples = self.num_samples
        # print(output.shape, target.shape, self.num_samples)
        batch_sze = target.shape[1]
        target_expand = target.repeat(num_samples)
        # print(output.shape, target_expand.shape, self.num_samples); exit(0)
        output = output.view(num_samples * batch_sze, -1)
        print(output.shape, target_expand.shape, target.shape); exit()
        

    def estimate_ELBO(self, neg_loglike_fun, act_vec, y, N_train, kl_weight=1):
        """
        Estimate the ELBO
        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        act_vec : scores from different samples of latent neural architecture Z
        y : real labels
        N_train: number of training data points
        kl_weight: coefficient to scale KL component

        Returns
        -------
        ELBO
        """
        # print(act_vec.shape, y.shape)
        E_loglike = self.get_E_loglike(neg_loglike_fun, act_vec, y)
        # print(E_loglike.shape)
        KL1 = self.structure_sampler.get_kl()
        # KL2 = self.structure_sampler_2.get_kl()
        # KL3 = self.structure_sampler_3.get_kl()
        # KL4 = self.structure_sampler_4.get_kl()
        # ELBO = E_loglike + (kl_weight * (KL1 + KL2 + KL3 + KL4))/N_train
        ELBO = E_loglike + (kl_weight * (KL1))/N_train
        return ELBO, KL1, kl_weight, E_loglike
    
    def get_embedding_dim(self):
        return self.embDim



class simpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes, num_channels=64, kernel_size=3, args=None, device=None):
        super(simpleCNN, self).__init__()
        self.mode = "NN"
        self.args = args

        self.max_channels = num_channels
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.truncation_level = args.truncation_level
        self.num_samples = args.num_samples
        self.device = device

        self.layers = nn.ModuleList([ConvBlock(self.input_channels, self.max_channels,
                                      kernel_size=5, pool=True).to(self.device)])
        
        for i in range(1, self.truncation_level):
            self.layers.append(ConvBlock(self.max_channels,
                                         self.max_channels,
                                         kernel_size=3,
                                         padding=1,
                                        residual=False,
                                        drop=0.5).to(self.device))
    
        self.out_layer = nn.Sequential(MLPBlock(self.max_channels, self.max_channels, residual=True),
                                       nn.Linear(self.max_channels, self.num_classes))
        
    def forward(self, x, num_samples=1, last=False, freeze=False):
        """
        Transforms the input feature matrix with different samples of network structures Z

        Parameters
        ----------
        x : data
        num_samples : Number of samples of network structure Z

        Returns
        -------
        act_vec : Tensor by stacking the output from different structures Z
        """
        # print("in", x.shape)
        # print("exit"); exit()
        if freeze:
            with torch.no_grad():
                for layer in self.layers:
                    x = layer(x)
                x = x.mean(dim=(2, 3))
                # print(x.shape); exit(0)
        else:        
            for layer in self.layers:
                x = layer(x)
            # print("af", x.shape)
            x = x.mean(dim=(2, 3))
            # print(out.shape); exit()
        
        out = self.out_layer(x)
        # print(out.shape, x.shape)
        if last:
            return out, x
    
        return out
        
    def get_embedding_dim(self):
        return self.max_channels
        
class AdaptiveConvNet(nn.Module):
    def __init__(self, input_channels, num_classes, num_channels=64, kernel_size=3, args=None, device=None):
        super(AdaptiveConvNet, self).__init__()
        self.mode = "NN"
        self.args = args

        self.max_channels = num_channels
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.truncation_level = args.truncation_level
        self.num_samples = args.num_samples
        self.device = device

        self.structure_sampler = NetworkStructureSampler(args, self.device)

        self.layers = nn.ModuleList([ConvBlock(self.input_channels, self.max_channels,
                                      kernel_size=5, pool=True).to(self.device)])

        for i in range(1, self.truncation_level):
            self.layers.append(ConvBlock(self.max_channels,
                                         self.max_channels,
                                         kernel_size=3,
                                         padding=1,
                                        residual=True).to(self.device))

        self.out_layer = nn.Sequential(MLPBlock(self.max_channels, self.max_channels, residual=True),
                                       nn.Linear(self.max_channels, self.num_classes))
        
    


    def _forward(self, x, mask_matrix, threshold, last=False):
        """
        Transform the input feature matrix with a sampled network structure
        Parameters
        ----------
        x : input feature matrix
        mask_matrix : mask matrix from Beta-Bernoulli Process
        threshold : number of layers in sampled structure

        Returns
        -------
        out : Output from the sampled architecture
        """
        # print("inpiut ", x.shape)
        # if number of layers sampled during testing is greater than truncation level, we use the network with
        # truncation level.
        if not self.training and threshold > len(self.layers):
            threshold = len(self.layers)
        # print("threshold", threshold)
        # running_sum = 0
        for layer in range(threshold):
            # print(layer, x.shape)
            # print(mask_matrix[:, layer].shape); exit(0)
            # running_sum += sum(p.numel() for p in self.layers[layer].parameters() if p.requires_grad)
            mask = mask_matrix[:, layer]
            x = self.layers[layer](x, mask)
            # print(layer, x.shape)
        # print(running_sum)
        # print("before mean ", x.shape)
        x = x.mean(dim=(2, 3))
        # print("before out", x.shape); exit(0)
        out = self.out_layer(x)
        if last:
            return out, x
        return out
    
    def _forward_nograd(self, x, mask_matrix, threshold):
        """
        Transform the input feature matrix with a sampled network structure
        Parameters
        ----------
        x : input feature matrix
        mask_matrix : mask matrix from Beta-Bernoulli Process
        threshold : number of layers in sampled structure

        Returns
        -------
        out : Output from the sampled architecture
        """
        # print("inpiut ", x.shape)
        # if number of layers sampled during testing is greater than truncation level, we use the network with
        # truncation level.
        if not self.training and threshold > len(self.layers):
            threshold = len(self.layers)
        # print("threshold", threshold)
        # running_sum = 0
        for layer in range(threshold):
            # print(mask_matrix[:, layer].shape); exit(0)
            # running_sum += sum(p.numel() for p in self.layers[layer].parameters() if p.requires_grad)
            mask = mask_matrix[:, layer]
            x = self.layers[layer](x, mask)
        # print(running_sum)
        # print("before mean ", x.shape)
        x = x.mean(dim=(2, 3))
        # print("before out", x.shape)
        return x

    def forward(self, x, num_samples=2, last=False, freeze=False):
        """
        Transforms the input feature matrix with different samples of network structures Z

        Parameters
        ----------
        x : data
        num_samples : Number of samples of network structure Z

        Returns
        -------
        act_vec : Tensor by stacking the output from different structures Z
        """
        self.num_samples = num_samples
        # print(self.num_samples, num_samples)
        if freeze:
            act_vec = []
            act_emb = []
            Z, threshold = self.structure_sampler(num_samples)
            for s in range(num_samples):
                with torch.no_grad():
                    l1 = self._forward_nograd(x, Z[s], threshold)
                    act_emb.append(l1.unsqueeze(0))
                out = self.out_layer(l1)
                act_vec.append(out.unsqueeze(0))
        else:
            act_vec = []
            act_emb = []
            # print(x.shape)
            Z, threshold = self.structure_sampler(num_samples)
            # print(Z.shape)
            for s in range(num_samples):
                if last:
                    out, l1 = self._forward(x, Z[s], threshold, last=True)
                    act_vec.append(out.unsqueeze(0))
                    act_emb.append(l1.unsqueeze(0))
                else:
                    out = self._forward(x, Z[s], threshold, last=False)
                    # print(out.shape)
                    act_vec.append(out.unsqueeze(0))

        if last:
            act_vec = torch.cat(act_vec, dim=0)
            act_emb = torch.cat(act_emb, dim=0)
            return act_vec, act_emb
        else:
            act_vec = torch.cat(act_vec, dim=0)
        # print(act_vec.shape)
            return act_vec

    def get_param_count(self, channels=1):
        Z, threshold = self.structure_sampler(1)
        running_sum = self.truncation_level
        # prev_channels = channels
        for layer in range(threshold):
        #     # print(mask_matrix[:, layer].shape); exit(0)
        #     if isinstance(self.layers[layer], ConvBlock):
        #         if layer == 0:
        #             total_filters = sum(torch.ceil(1 * Z[0][:, layer]))
        #             running_sum += (5 * 5 * prev_channels) * total_filters + (total_filters * 2)
        #             # print(total_filters); exit(0)
        #         else:
        #             total_filters = sum(torch.floor(1 * Z[0][:, layer]))
        #             running_sum += (3 * 3 * prev_channels) * total_filters + (total_filters * 2)
        #         print(layer, prev_channels, total_filters)
        #         prev_channels = total_filters
        #     elif isinstance(self.layers[layer], MLPBlock):
        #         running_sum += 4160 + 128 + 650
        # print(running_sum)    
        # exit()
                
            running_sum += sum(p.numel() for p in self.layers[layer].parameters() if p.requires_grad)
            # mask = mask_matrix[:, layer]
            # x = self.layers[layer](x, mask)
        return running_sum + 4169 + 128 + 650 + 1600 + 128

    def get_E_loglike(self, neg_loglike_fun, output, target):
        """

        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        output : scores predicted by model
        target : real labels

        Returns
        -------
        mean negative log likelihood of the model based on different architectures
        """

        num_samples = self.num_samples
        batch_sze = target.shape[0]
        target_expand = target.repeat(num_samples)
        output = output.view(num_samples * batch_sze, -1)
        # print(target_expand.shape, output.shape); exit()
        neg_loglike = neg_loglike_fun(output, target_expand)#.view(num_samples, batch_sze)
        # print(neg_loglike.shape); exit()
        # E_neg_loglike = neg_loglike.mean(0).mean()
        return neg_loglike

    def estimate_ELBO(self, neg_loglike_fun, act_vec, y, N_train, kl_weight=1):
        """
        Estimate the ELBO
        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        act_vec : scores from different samples of latent neural architecture Z
        y : real labels
        N_train: number of training data points
        kl_weight: coefficient to scale KL component

        Returns
        -------
        ELBO
        """
        E_loglike = self.get_E_loglike(neg_loglike_fun, act_vec, y)
        KL = self.structure_sampler.get_kl()
        ELBO = E_loglike + (kl_weight * KL)/N_train
        return ELBO, KL, (kl_weight * KL)/N_train, E_loglike
    
    def get_embedding_dim(self):
        return self.max_channels
    
class AdaptiveResnet(nn.Module):
    def __init__(self, input_channels, num_classes, num_channels=32, kernel_size=3, args=None, device=None):
        super(AdaptiveResnet, self).__init__()
        self.mode = "NN"
        self.args = args
        
        self.max_channels = num_channels
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.truncation_level = args.truncation_level
        self.num_samples = args.num_samples
        self.device = device

        print(self.input_channels)
        # self.conv1 = ConvBlockRN(input_channels, 64,)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        args.max_width=64
        self.l1_sampler = NetworkStructureSampler(args, self.device)
        self.layer1 = nn.ModuleList([RNNBlock(64, 64, stride=1).to(device)])
        for i in range(1, self.truncation_level):
            self.layer1.append(RNNBlock(64, 64).to(device))

        args.max_width=128
        self.l2_sampler = NetworkStructureSampler(args, self.device)
        self.layer2_c = RNNBlock(64, 128, 2).to(device)
        self.layer2 = nn.ModuleList([])
        for i in range(1, self.truncation_level):
            self.layer2.append(RNNBlock(128, 128).to(device))
            
        args.max_width=256
        self.l3_sampler = NetworkStructureSampler(args, self.device)
        self.layer3_c = RNNBlock(128, 256, 2).to(device)
        self.layer3 = nn.ModuleList([])
        for i in range(1, self.truncation_level):
            self.layer3.append(RNNBlock(256, 256).to(device))
            
        args.max_width=512
        self.l4_sampler = NetworkStructureSampler(args, self.device)
        self.layer4_c = RNNBlock(256, 512, 2).to(device)
        self.layer4 = nn.ModuleList([])
        for i in range(1, self.truncation_level):
            self.layer4.append(RNNBlock(512, 512).to(device))
        self.out_layer = nn.Linear(512, num_classes)
        self.linear1 = nn.Linear(128* 14 * 14, 10)
        # self.linear2 = nn.Linear(28 * 28, 10)


    def _forward(self, x, mask_matrix, threshold, sp_layer):
        """
        Transform the input feature matrix with a sampled network structure
        Parameters
        ----------
        x : input feature matrix
        mask_matrix : mask matrix from Beta-Bernoulli Process
        threshold : number of layers in sampled structure

        Returns
        -------
        out : Output from the sampled architecture
        """
        # Individual Resnet Block Layer
        if not self.training and threshold > len(sp_layer):
            threshold = len(sp_layer)

        # print(x.shape, mask_matrix.shape)
        for layer in range(threshold):
            mask = mask_matrix[:, layer]
            # print("lm", mask.shape, sp_layer[layer])
            x = sp_layer[layer](x, mask)
            # print("lmx", x.shape)
        # print("fin", x.shape)
        return x

    def forward(self, x, num_samples=5):
        """
        Transforms the input feature matrix with different samples of network structures Z

        Parameters
        ----------
        x : data
        num_samples : Number of samples of network structure Z

        Returns
        -------
        act_vec : Tensor by stacking the output from different structures Z
        """
        # First Resnet Block Layer
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        act_vec = []
        Z, threshold = self.l1_sampler(num_samples)
        for s in range(num_samples):
            out = self._forward(x, Z[s], threshold, self.layer1)

            act_vec.append(out.unsqueeze(0))

        act_vec = torch.cat(act_vec, dim=0)
        act_vec = torch.mean(act_vec, dim=0)
        # print(act_vec.shape)
        # Second Resnet Block Layer
        act_vec1 = []
        act_vec = self.layer2_c(act_vec)
        # print(act_vec.shape)
        Z, threshold = self.l2_sampler(num_samples)
        for s in range(num_samples):
            out = self._forward(act_vec, Z[s], threshold, self.layer2)

            act_vec1.append(out.unsqueeze(0))

        act_vec1 = torch.cat(act_vec1, dim=0)
        act_vec1 = torch.mean(act_vec1, dim=0)
        # print(act_vec1.shape)
        # return self.linear1(act_vec.view(-1, 128 * 14 * 14))
        
        # Third Resnet Block Layer
        act_vec2 = []
        act_vec1 = self.layer3_c(act_vec1)
        # print(act_vec1.shape)
        Z, threshold = self.l3_sampler(num_samples)
        for s in range(num_samples):
            out = self._forward(act_vec1, Z[s], threshold, self.layer3)

            act_vec2.append(out.unsqueeze(0))

        act_vec2 = torch.cat(act_vec2, dim=0)
        act_vec2 = torch.mean(act_vec2, dim=0)
        # print(act_vec2.shape)
        
        # Fourth Resnet Block Layer
        act_vec3 = []
        act_vec2 = self.layer4_c(act_vec2)
        # print(act_vec2.shape)
        Z, threshold = self.l4_sampler(num_samples)
        for s in range(num_samples):
            out = self._forward(act_vec2, Z[s], threshold, self.layer4)

            act_vec3.append(out.unsqueeze(0))

        act_vec3 = torch.cat(act_vec3, dim=0)
        act_vec3 = torch.mean(act_vec3, dim=0)
        # print(act_vec3.shape)
        out = F.avg_pool2d(out, 4)
        # print(out.shape)
        e = out.view(out.size(0), -1)
        # print(e.shape)
        out = self.out_layer(e)
        # print(out.shape); exit()
        return out
        

    def get_E_loglike(self, neg_loglike_fun, output, target):
        """

        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        output : scores predicted by model
        target : real labels

        Returns
        -------
        mean negative log likelihood of the model based on different architectures
        """

        # num_samples = self.num_samples
        # batch_sze = target.shape[0]
        # target_expand = target.repeat(num_samples)
        # output = output.view(num_samples * batch_sze, -1)
        neg_loglike = neg_loglike_fun(output, target)#.view(num_samples, batch_sze)
        # print(neg_loglike.shape); exit(0)
        # E_neg_loglike = neg_loglike.mean(0).mean()
        return neg_loglike

    def estimate_ELBO(self, neg_loglike_fun, act_vec, y, N_train, kl_weight=1):
        """
        Estimate the ELBO
        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        act_vec : scores from different samples of latent neural architecture Z
        y : real labels
        N_train: number of training data points
        kl_weight: coefficient to scale KL component

        Returns
        -------
        ELBO
        """
        print(act_vec.shape, y.shape)
        E_loglike = self.get_E_loglike(neg_loglike_fun, act_vec, y)
        KL1 = self.l1_sampler.get_kl()
        KL2 = self.l2_sampler.get_kl()
        KL3 = self.l3_sampler.get_kl()
        KL4 = self.l4_sampler.get_kl()
        ELBO = E_loglike + (kl_weight * (KL1  + KL2 + KL3 + KL4))/N_train
        return ELBO
    
class AdaptiveBlockEncoder(nn.Module):
    def __init__(self, input_dim, num_neurons=400, a_prior=1.1, b_prior=1., num_samples=5, truncation=2, device=torch.device("cuda:0")):
        super(AdaptiveBlockEncoder, self).__init__()
        self.mode = "NN"
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.truncation = truncation
        self.num_samples = num_samples
        self.device = device

        self.architecture_sampler = SampleNetworkArchitecture(num_neurons=num_neurons,
                                                              a_prior=a_prior,
                                                              b_prior=b_prior,
                                                              num_samples=num_samples,
                                                              truncation=truncation,
                                                              device=self.device)

        self.layers = nn.ModuleList([]).to(device)
        for i in range(self.truncation):
            self.layers.append(ConvBlock(self.num_neurons,
                                         self.num_neurons,
                                         kernel_size=3,
                                         padding=1,
                                        residual=True).to(self.device))
        print(self.layers)

    def _forward(self, x, mask_matrix, threshold):
        if not self.training and threshold > len(self.layers):
            threshold = len(self.layers)
            
        # print(threshold, mask_matrix.shape)
        for layer_idx in range(threshold):
            mask = mask_matrix[:, layer_idx]
            x = self.layers[layer_idx](x, mask)
        return x
    
    def _backward(self, x, mask_matrix, threshold):
        if not self.training and threshold > len(self.layers):
            threshold = len(self.layers)

        for layer_idx in range(threshold - 1, -1, -1):

            mask = mask_matrix[:, layer_idx]
            x = self.layers[layer_idx](x, mask)

        return x

    def forward(self, x, num_samples=5, reverse=False):
        """
        Fits the data with different samples of architectures

        Parameters
        ----------
        x : data
        num_samples : Number of architectures to sample for KL divergence

        Returns
        -------
        act_vec : Tensor
            output from different architectures
        kl_loss: Tensor
            Kl divergence for each sampled architecture
        thresholds: numpy array
            threshold sampled for different architectures
        """
        act_vec = []
        Z, pi, n_layers, _ = self.architecture_sampler(num_samples)
        # print("N layers", n_layers)
        # print('Z', Z.shape)
        if not reverse:
            # print("Forward")
            for s in range(num_samples):
                out = self._forward(x, Z[s], n_layers)
                act_vec.append(out.unsqueeze(0))
        else:
            print("Reverse")
            for s in range(num_samples):
                out = self._backward(x, Z[s], n_layers)
                act_vec.append(out.unsqueeze(0))
        act_vec = torch.cat(act_vec, dim=0)
        # print("actvec", act_vec.shape)
        act_vec = torch.mean(act_vec, dim= 0)
        # print("actvec mean", act_vec.shape)
        return act_vec
        # return out
        
    def get_network_mask(self, n_samples):
        Z, _, n_layers, _ = self.architecture_sampler(n_samples)
        return Z, n_layers

class AdaptiveBlockDecoder(nn.Module):
    def __init__(self, input_dim, num_neurons=400, a_prior=1.1, b_prior=1., num_samples=5, truncation=2, device=torch.device("cuda:0")):
        super(AdaptiveBlockDecoder, self).__init__()
        self.mode = "NN"
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.truncation = truncation
        self.num_samples = num_samples
        self.device = device

        self.architecture_sampler = SampleNetworkArchitecture(num_neurons=num_neurons,
                                                              a_prior=a_prior,
                                                              b_prior=b_prior,
                                                              num_samples=num_samples,
                                                              truncation=truncation,
                                                              device=self.device)

        self.layers = nn.ModuleList([]).to(device)
        for i in range(0, self.truncation):
            self.layers.append(ConvBlock(self.num_neurons,
                                         self.num_neurons,
                                         kernel_size=3,
                                         padding=1,
                                        residual=True).to(self.device))
        # print(self.layers)

    def _forward(self, x, mask_matrix, threshold):
        if not self.training and threshold > len(self.layers):
            threshold = len(self.layers)
        # print("threshold", threshold)
        # print(mask_matrix.shape)
        for layer_idx in range(threshold):
            mask = mask_matrix[:, layer_idx]
            # print(layer_idx, x.shape, mask.shape)
            x = self.layers[layer_idx](x, mask)
            # print(layer_idx, mask.shape, x.shape)
        return x
    
    def _backward(self, x, mask_matrix, threshold):
        if not self.training and threshold > len(self.layers):
            threshold = len(self.layers)

        for layer_idx in range(threshold - 1, -1, -1):

            mask = mask_matrix[:, layer_idx]
            x = self.layers[layer_idx](x, mask)

        return x

    # def forward(self, x, num_samples=5, reverse=False, sampler=None):
    def forward(self, x, num_samples=5, reverse=False, Z=None, n_layers=None):
        """
        Fits the data with different samples of architectures

        Parameters
        ----------
        x : data
        num_samples : Number of architectures to sample for KL divergence

        Returns
        -------
        act_vec : Tensor
            output from different architectures
        kl_loss: Tensor
            Kl divergence for each sampled architecture
        thresholds: numpy array
            threshold sampled for different architectures
        """
        act_vec = []
        # if sampler is None:
        #     Z, pi, n_layers, _ = self.architecture_sampler(num_samples)
        # else:
        #     self.architecture_sampler = sampler
        #     Z, pi, n_layers, _ = sampler(num_samples)
            
        
        # print("N layers", n_layers)
        # print('Z', Z.shape)
        # exit(0)
        if not reverse:
            # print("Forward")
            for s in range(num_samples):
                out = self._forward(x, Z[s], n_layers)
                act_vec.append(out.unsqueeze(0))
        else:
            # print("Reverse")
            for s in range(num_samples):
                out = self._backward(x, Z[s], n_layers)
                act_vec.append(out.unsqueeze(0))
        act_vec = torch.cat(act_vec, dim=0)
        act_vec = torch.mean(act_vec, dim= 0)
        return act_vec
        # return out

from torch.nn.utils import weight_norm
from torch.autograd import Variable, Function

class GaussianNoise(nn.Module):
    
    def __init__(self, std):
        super(GaussianNoise, self).__init__()
        self.std = std
    
    def forward(self, x):
        zeros_ = torch.zeros(x.size()).cuda()
        n = Variable(torch.normal(zeros_, std=self.std).cuda())
        return x + n
    
class CNNGaussSingle(nn.Module):
    """
    CNN from Mean Teacher paper
    """
    
    def __init__(self, num_classes=10, isL2 = False):
        super(CNNGaussSingle, self).__init__()

        self.isL2 = isL2

        self.gn = GaussianNoise(0.15)
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = nn.Conv2d(3, 128, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1  = nn.Dropout(0.5)
        
        self.conv2a = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2  = nn.Dropout(0.5)
        
        self.conv3a = nn.Conv2d(256, 512, 3, padding=0)
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = nn.Conv2d(512, 256, 1, padding=0)
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = nn.Conv2d(256, 128, 1, padding=0)
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)
        
        self.fc1 =  nn.Linear(128, num_classes)
    
    def forward(self, x, debug=False, last=False, freeze=False):
        if freeze:
            with torch.no_grad:
                x = self.gn(x)
                x = self.activation(self.bn1a(self.conv1a(x)))
                x = self.activation(self.bn1b(self.conv1b(x)))
                x = self.activation(self.bn1c(self.conv1c(x)))
                x = self.mp1(x)
                x = self.drop1(x)
                
                x = self.activation(self.bn2a(self.conv2a(x)))
                x = self.activation(self.bn2b(self.conv2b(x)))
                x = self.activation(self.bn2c(self.conv2c(x)))
                x = self.mp2(x)
                x = self.drop2(x)
                
                x = self.activation(self.bn3a(self.conv3a(x)))
                x = self.activation(self.bn3b(self.conv3b(x)))
                x = self.activation(self.bn3c(self.conv3c(x)))
                x = self.ap3(x)
        
                x = x.view(-1, 128)
                if self.isL2:
                    x = F.normalize(x)
        else:
            
            x = self.gn(x)
            x = self.activation(self.bn1a(self.conv1a(x)))
            x = self.activation(self.bn1b(self.conv1b(x)))
            x = self.activation(self.bn1c(self.conv1c(x)))
            x = self.mp1(x)
            x = self.drop1(x)
            
            x = self.activation(self.bn2a(self.conv2a(x)))
            x = self.activation(self.bn2b(self.conv2b(x)))
            x = self.activation(self.bn2c(self.conv2c(x)))
            x = self.mp2(x)
            x = self.drop2(x)
            
            x = self.activation(self.bn3a(self.conv3a(x)))
            x = self.activation(self.bn3b(self.conv3b(x)))
            x = self.activation(self.bn3c(self.conv3c(x)))
            x = self.ap3(x)
    
            x = x.view(-1, 128)
            if self.isL2:
                x = F.normalize(x)
        if last:
            return self.fc1(x), x
        else:
            return self.fc1(x)
        
class CNNGaussSingle1T(nn.Module):
    """
    CNN from Mean Teacher paper
    """
    
    def __init__(self, num_classes=10, isL2 = False):
        super(CNNGaussSingle1T, self).__init__()

        self.isL2 = isL2

        self.gn = GaussianNoise(0.15)
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = nn.Conv2d(3, 128, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(128)
        self.fc2 =  nn.Linear(128 * 32 * 32, 128)
        self.fc1 =  nn.Linear(128, num_classes)
    
    def forward(self, x, debug=False, last=False, freeze=False):
        if freeze:
            with torch.no_grad:
                x = self.gn(x)
                x = self.activation(self.bn1a(self.conv1a(x)))
                x = x.view(-1, 128*32*32)
                x = self.fc2(x)
                if self.isL2:
                    x = F.normalize(x)
        else:
            x = self.gn(x)
            x = self.activation(self.bn1a(self.conv1a(x)))
            x = x.view(-1, 128*32*32)
            x = self.fc2(x)
            if self.isL2:
                x = F.normalize(x)
        if last:
            return self.fc1(x), x
        else:
            return self.fc1(x)
        
class CNNGaussSingle3T(nn.Module):
    """
    CNN from Mean Teacher paper
    """
    
    def __init__(self, num_classes=10, isL2 = False):
        super(CNNGaussSingle3T, self).__init__()

        self.isL2 = isL2

        self.gn = GaussianNoise(0.15)
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = nn.Conv2d(3, 128, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1  = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128 * 16 * 16, 128)
        
        self.fc1 =  nn.Linear(128, num_classes)
    
    def forward(self, x, debug=False, last=False, freeze=False):
        if freeze:
            with torch.no_grad:
                x = self.gn(x)
                x = self.activation(self.bn1a(self.conv1a(x)))
                x = self.activation(self.bn1b(self.conv1b(x)))
                x = self.activation(self.bn1c(self.conv1c(x)))
                x = self.mp1(x)
                x = self.drop1(x)
                
                x = x.view(-1, 128 * 16 * 16)
                x = self.fc2(x)
                if self.isL2:
                    x = F.normalize(x)
        else:
            
            x = self.gn(x)
            x = self.activation(self.bn1a(self.conv1a(x)))
            x = self.activation(self.bn1b(self.conv1b(x)))
            x = self.activation(self.bn1c(self.conv1c(x)))
            x = self.mp1(x)
            x = self.drop1(x)
            
            x = x.view(-1, 128 * 16 * 16)
            x = self.fc2(x)
            if self.isL2:
                x = F.normalize(x)
        if last:
            return self.fc1(x), x
        else:
            return self.fc1(x)
        
class CNNGaussSingle5T(nn.Module):
    """
    CNN from Mean Teacher paper
    """
    
    def __init__(self, num_classes=10, isL2 = False):
        super(CNNGaussSingle5T, self).__init__()

        self.isL2 = isL2

        self.gn = GaussianNoise(0.15)
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = nn.Conv2d(3, 128, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1  = nn.Dropout(0.5)
        
        self.conv2a = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2  = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256*8*8, 128)
        self.fc1 =  nn.Linear(128, num_classes)
    
    def forward(self, x, debug=False, last=False, freeze=False):
        if freeze:
            with torch.no_grad:
                x = self.gn(x)
                x = self.activation(self.bn1a(self.conv1a(x)))
                x = self.activation(self.bn1b(self.conv1b(x)))
                x = self.activation(self.bn1c(self.conv1c(x)))
                x = self.mp1(x)
                x = self.drop1(x)
                
                x = self.activation(self.bn2a(self.conv2a(x)))
                x = self.activation(self.bn2b(self.conv2b(x)))
                x = self.activation(self.bn2c(self.conv2c(x)))
                x = self.mp2(x)
                x = self.drop2(x)
                # print(x.shape); exit(0)
                
                x = x.view(-1, 256*8*8)
                x = self.fc2(x)
                if self.isL2:
                    x = F.normalize(x)
        else:
            
            x = self.gn(x)
            x = self.activation(self.bn1a(self.conv1a(x)))
            x = self.activation(self.bn1b(self.conv1b(x)))
            x = self.activation(self.bn1c(self.conv1c(x)))
            x = self.mp1(x)
            x = self.drop1(x)
            
            x = self.activation(self.bn2a(self.conv2a(x)))
            x = self.activation(self.bn2b(self.conv2b(x)))
            x = self.activation(self.bn2c(self.conv2c(x)))
            x = self.mp2(x)
            x = self.drop2(x)
            
            # print(x.shape); exit(0)
            x = x.view(-1, 256 * 8 * 8)
            # x = x.view(-1, 128)
            x = self.fc2(x)
            if self.isL2:
                x = F.normalize(x)
        if last:
            return self.fc1(x), x
        else:
            return self.fc1(x)