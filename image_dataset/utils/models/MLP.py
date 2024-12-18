import torch
from torch import nn
from utils.layers.layers import MLPBlock
from utils.models.NetworkStructureSampler import NetworkStructureSampler

class simpleMLP(nn.Module):
    def __init__(self, input_feature_dim, out_feature_dim, args, device):
        super(simpleMLP, self).__init__()
        self.args = args
        self.input_feature_dim = input_feature_dim
        self.out_feature_dim = out_feature_dim
        
        # Maximum number of neurons (M)
        self.max_width = args.max_width
        self.bn = nn.BatchNorm1d(self.max_width)
        self.act_layer_fn = nn.LeakyReLU()
        self.truncation_level = args.truncation_level # K truncation_level
        # self.layers = nn.Linear(self.input_feature_dim, self.max_width)
        # self.out_layer = nn.Linear(self.max_width, self.out_feature_dim)
        self.layers = nn.ModuleList([MLPBlock(self.input_feature_dim, self.max_width)])
        for i in range(1, self.truncation_level):
            self.layers.append(MLPBlock(self.max_width, self.max_width, residual=False))

        self.out_layer = nn.Linear(self.max_width, self.out_feature_dim)
        
    def forward(self, x, num_samples=1, last=False, freeze=False):
        B, _, _, _ = x.shape
        x = x.view(B, -1)
        # x = self.bn(self.act_layer_fn(self.layers(x)))
        
        # out = self.out_layer(x)
        # return out
        if freeze:
            with torch.no_grad:
                for layer in self.layers:
                    x = layer(x)
                
                out = self.out_layer(x)
                if last:
                    return out, x
            
                return out
        else:        
            for layer in self.layers:
                x = layer(x)
            # print("af", x.shape)
            # x = x.mean(dim=(2, 3))
            out = self.out_layer(x)
            # print(out.shape); exit()
            if last:
                return out, x
        
            return out
    
    def get_embedding_dim(self):
        return self.max_width

class AdaptiveMLP(nn.Module):
    def __init__(self, input_feature_dim, out_feature_dim, args, device):
        super(AdaptiveMLP, self).__init__()
        self.args = args
        self.input_feature_dim = input_feature_dim
        self.out_feature_dim = out_feature_dim

        # Maximum number of neurons (M)
        self.max_width = args.max_width

        # Truncation level for variational approximation
        self.truncation_level = args.truncation_level # K truncation_level
        self.num_samples = args.num_samples
        self.device = device

        # define neural network structure sampler with parameters defined in argument parser
        self.structure_sampler = NetworkStructureSampler(args, self.device)

        # Define the architecture of neural network by simply adding K layers at initialization
        # Note: we can also dynamically add new layers based on the inferred depth
        self.layers = nn.ModuleList([MLPBlock(self.input_feature_dim, self.max_width)])
        for i in range(self.truncation_level):
            self.layers.append(MLPBlock(self.max_width, self.max_width, residual=True))

        self.out_layer = nn.Linear(self.max_width, self.out_feature_dim)

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
        # print(x.shape)
        # exit(0)
        # if number of layers sampled during testing is greater than truncation level, we use the network with
        # truncation level.
        if not self.training and threshold > len(self.layers):
            threshold = len(self.layers)

        for layer in range(threshold):
            mask = mask_matrix[:, layer]
            x = self.layers[layer](x, mask)

        out = self.out_layer(x)
        if last:
            return out, x
        return out

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
        B, _, _, _ = x.shape
        x = x.view(B, -1)
        # self.num_samples = num_samples
        # act_vec = []
        # Z, threshold = self.structure_sampler(num_samples)

        # for s in range(num_samples):
        #     out = self._forward(x, Z[s], threshold)
        #     act_vec.append(out.unsqueeze(0))

        # act_vec = torch.cat(act_vec, dim=0)
        # return act_vec
    
        self.num_samples = num_samples
        if freeze:
            with torch.no_grad():
                act_vec = []
            act_emb = []
            Z, threshold = self.structure_sampler(num_samples)

            for s in range(num_samples):
                if last:
                    out, l1 = self._forward(x, Z[s], threshold, last=True)
                    act_vec.append(out.unsqueeze(0))
                    act_emb.append(l1.unsqueeze(0))
                else:
                    out = self._forward(x, Z[s], threshold, last=False)
                    act_vec.append(out.unsqueeze(0))

            if last:
                act_vec = torch.cat(act_vec, dim=0)
                act_emb = torch.cat(act_emb, dim=0)
                return act_vec, act_emb
            else:
                act_vec = torch.cat(act_vec, dim=0)
            # print(act_vec.shape)
                return act_vec
        else:
            act_vec = []
            act_emb = []
            Z, threshold = self.structure_sampler(num_samples)

            for s in range(num_samples):
                if last:
                    out, l1 = self._forward(x, Z[s], threshold, last=True)
                    act_vec.append(out.unsqueeze(0))
                    act_emb.append(l1.unsqueeze(0))
                else:
                    out = self._forward(x, Z[s], threshold, last=False)
                    act_vec.append(out.unsqueeze(0))
                    

            if last:
                act_vec = torch.cat(act_vec, dim=0)
                act_emb = torch.cat(act_emb, dim=0)
                return act_vec, act_emb
            else:
                act_vec = torch.cat(act_vec, dim=0)
            # print(act_vec.shape)
                return act_vec

    def get_E_loglike(self, neg_loglike_fun, act_vec, y):
        """
        Compute the expectation of log likelihood
        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        act_vec : scores from different samples of latent neural architecture Z
        y : real labels

        Returns
        -------
        mean_neg_loglike: Expectation of negative log likelihood of the model based on different structures Z
        """
        y = y.expand(self.num_samples, -1)
        act_vec = act_vec.permute(1, 2, 0)
        y = y.permute(1, 0)
        # act_vec = act_vec.view(-1, 10)
        # y = y.reshape(-1, 1)
        # print(y.shape, act_vec.shape)
        neg_log_likelihood = neg_loglike_fun(act_vec, y)
        # mean_neg_log_likelihood = neg_log_likelihood.mean(0).mean()
        
        return neg_log_likelihood

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
        return self.max_width

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
        return running_sum + 4169 + 128 + 650