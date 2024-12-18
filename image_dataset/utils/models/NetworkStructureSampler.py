import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Beta, RelaxedBernoulli, Bernoulli
from torch.distributions.kl import kl_divergence


class NetworkStructureSampler(nn.Module):
    """
    Samples the network structure Z from Beta-Bernoulli process prior
    Network structure Z represents depth + dropout mask as:
    (a) depth: the number of layers with activated neurons
    (b) binary mask modulating the neuron activations
    """

    def __init__(self, args, device: torch.device):
        super(NetworkStructureSampler, self).__init__()
        self.args = args
        self.device = device

        # epsilon to select the number of layers with activated neurons
        self.ε = args.epsilon

        # maximum number of neurons/channels (M) in each layer
        self.max_width = args.max_width

        # Truncation level for variational approximation
        self.truncation_level = args.truncation_level

        # Temperature for Concrete Bernoulli
        self.τ = torch.tensor(args.temp)

        # Number of samples to estimate expectations
        self.num_samples = args.num_samples

        # Hyper-parameters for prior beta
        self.α = torch.tensor(args.a_prior).float().to(self.device)
        self.β = torch.tensor(args.b_prior).float().to(self.device)

        # Define a prior beta distribution
        self.prior_beta = Beta(self.α, self.β)
    
        # inverse softplus to avoid parameter underflow
        # α = np.log(np.exp(args.a_prior) - 1)
        # β = np.log(np.exp(args.b_prior) - 1)
        
        # Previous Prior we used
        α = np.log(np.exp(np.random.uniform(1.1, 1.1)) - 1)
        β = np.log(np.exp(np.random.uniform(1.0, 1.0)) - 1)
        # print(α, β); exit()

        # Define variational parameters for posterior distribution
        self.a_k = nn.Parameter(torch.Tensor(self.truncation_level).zero_() + α).requires_grad_()
        self.b_k = nn.Parameter(torch.Tensor(self.truncation_level).zero_() + β).requires_grad_()
        print('initialization ', self.a_k, self.b_k)

    def get_variational_params(self):
        a_k = F.softplus(self.a_k) + 0.01
        b_k = F.softplus(self.b_k) + 0.01
        return a_k, b_k
    
    def reset_parameters(self):
        α = np.log(np.exp(np.random.uniform(1.1, 1.1)) - 1)
        β = np.log(np.exp(np.random.uniform(1.0, 1.0)) - 1)
        # print(α, β); exit()

        # Define variational parameters for posterior distribution
        self.a_k = nn.Parameter(torch.Tensor(self.truncation_level).zero_() + α).requires_grad_()
        self.b_k = nn.Parameter(torch.Tensor(self.truncation_level).zero_() + β).requires_grad_()
        print('reset ', self.a_k, self.b_k)

    def get_kl(self):
        """
        Computes the KL divergence between variational beta and prior beta
        """
        a_k, b_k = self.get_variational_params()
        variational_beta = Beta(a_k, b_k)
        kl_beta = kl_divergence(variational_beta, self.prior_beta)
        # print("posterior", a_k, b_k)
        # print("prior", self.α, self.β)
        # print("Individual kl", kl_beta)
        # print("Sum KL", kl_beta.sum())
        return kl_beta.sum()

    def get_threshold(self, Z: torch.Tensor):
        """
        Compute the threshold i.e. layers with activated neurons

        Parameters
        ----------
        Z : binary mask matrix from beta-Bernoulli process

        Returns
        -------
        threshold: number of layers with activated neurons
        """

        # First, count the number of neurons in each layer
        threshold_Z = (Z > self.ε).sum(1)
        # Second, compute the layers with activated neurons
        threshold_array = (threshold_Z > 0).sum(dim=1).cpu().numpy()
        # Third, consider maximum of thresholds from multiple samples
        threshold = max(threshold_array)
        # Previous version
        if threshold == 0:
            threshold = torch.tensor(1)
        return threshold

    def forward(self, num_samples: int = 2, get_pi: bool = False):
        # print("n_samples", num_samples)
        # Define variational beta distribution
        a_k, b_k = self.get_variational_params()
        variational_beta = Beta(a_k, b_k)
        # self.get_kl()
        # sample from variational beta distribution
        # Previous didn't have .view() part
        ν = variational_beta.rsample((num_samples,)).view(num_samples, self.truncation_level)  # S x K
        # print(num_samples, ν, ν.shape); exit()
        # Convert ν to π i.e. activation level of layer
        # Product of ν is equivalent to cumulative sum of log of ν
        π = torch.cumsum(ν.log(), dim=1).exp()
        keep_prob = π.detach().mean(0)
        π = π.unsqueeze(1).expand(-1, self.max_width, -1)  # S x M x K

        # sample binary mask z_l given the activation level π_l of the layer
        if self.training:
            # draw continuous binary sample from concrete Bernoulli distribution to backpropagate the gradient
            concrete_bernoulli_dist = RelaxedBernoulli(probs=π, temperature=self.τ)
            Z = concrete_bernoulli_dist.rsample()
        else:
            # draw discrete sample from Bernoulli distribution
            bernoulli_dist = Bernoulli(probs=π)
            Z = bernoulli_dist.sample()
            # Z = π
        # print(π); exit()

        threshold = self.get_threshold(Z)
        
        if get_pi:
            # return probabilities to plot
            return Z, threshold, keep_prob

        return Z, threshold

class SampleNetworkArchitecture(nn.Module):
    """
    Samples an architecture from Beta-Bernoulli prior
    """

    def __init__(self, num_neurons=400, a_prior=1.1, b_prior=1., num_samples=5, truncation=50, device=None):
        super(SampleNetworkArchitecture, self).__init__()
        self.device = device
        self.num_neurons = num_neurons
        # Hyper-parameters for Prior probabilities
        self.a_prior = torch.tensor(a_prior).float().to(self.device)
        self.b_prior = torch.tensor(b_prior).float().to(self.device)

        # Define a prior beta distribution
        self.beta_prior = Beta(self.a_prior, self.b_prior)

        # Temperature for Bernoulli sampling
        self.temperature = torch.tensor(0.1).to(self.device)
        self.truncation = truncation
        # Number of samples from IBP prior to estimate expectations
        self.num_samples = num_samples

        a_val = np.log(np.exp(np.random.uniform(0.54, 0.54)) - 1)  # inverse softplus
        b_val = np.log(np.exp(np.random.uniform(1.0, 1.0)) - 1)

        # Define variational parameters for posterior distribution
        self.kuma_a = nn.Parameter(torch.Tensor(self.truncation).zero_() + a_val)
        self.kuma_b = nn.Parameter(torch.Tensor(self.truncation).zero_() + b_val)

    def get_var_params(self):
        beta_a = F.softplus(self.kuma_a) + 0.01
        beta_b = F.softplus(self.kuma_b) + 0.01
        return beta_a, beta_b
    
    def reset_parameters(self):
        a_val = np.log(np.exp(np.random.uniform(0.54, 0.54)) - 1)  # inverse softplus
        b_val = np.log(np.exp(np.random.uniform(1.0, 1.0)) - 1)

        # Define variational parameters for posterior distribution
        self.kuma_a = nn.Parameter(torch.Tensor(self.truncation).zero_() + a_val)
        self.kuma_b = nn.Parameter(torch.Tensor(self.truncation).zero_() + b_val)
        

    def get_kl_beta(self):
        """
        Computes the KL divergence between posterior and prior
        Parameters
        ----------
        threshold : Number of layers on sampled network

        Returns
        -------

        """
        beta_a, beta_b = self.get_var_params()
        beta_posterior = Beta(beta_a, beta_b)
        kl_beta = kl_divergence(beta_posterior, self.beta_prior)
        return kl_beta.sum()

    # def get_kl_bernoulli(self):
    #     v_prior = self.beta_prior.rsample((self.truncation,)).view(1, -1)
    #     pi_prior = torch.cumsum(v_prior.log(), dim=1).exp().view(-1)
    #     bernoulli_prior = Bernoulli(probs=pi_prior)
    #
    #     beta_a, beta_b = self.get_var_params()
    #     beta_posterior = Beta(beta_a, beta_b)
    #     v_post = beta_posterior.rsample().view(1, -1)
    #     pi_post = torch.cumsum(v_post.log(), dim=1).exp().view(-1)
    #     bernoulli_post = Bernoulli(probs=pi_post)
    #
    #     kl_bernoulli = kl_divergence(bernoulli_post, bernoulli_prior)
    #     return kl_bernoulli.sum()

    def get_kl(self):
        return self.get_kl_beta() #+ self.get_kl_bernoulli()

    def forward(self, num_samples=5, get_pi=False, get_intermediate_pi=False):
        # Define variational beta distribution
        beta_a, beta_b = self.get_var_params()
        # print(beta_a, beta_b)
        beta_posterior = Beta(beta_a, beta_b)

        # sample from variational beta distribution
        v = beta_posterior.rsample((num_samples, )).view(num_samples, self.truncation)

        # Convert v -> pi i.e. activation level of layer
        pi = torch.cumsum(v.log(), dim=1).exp()
        keep_prob = pi.detach().mean(0)
        pi = pi.unsqueeze(1).expand(-1, self.num_neurons, -1)


        if self.training:
            # sample active neurons given the activation probability of that layer
            bernoulli_dist = RelaxedBernoulli(temperature=self.temperature, probs=pi)
            Z = bernoulli_dist.rsample()

        else:
            # sample active neurons given the activation probability of that layer
            bernoulli_dist = Bernoulli(probs=pi)
            Z = bernoulli_dist.sample()

        # compute threshold
        threshold_Z = (Z > 0.01).sum(1)
        threshold_array = (threshold_Z > 0).sum(dim=1).cpu().numpy()
        threshold = max(threshold_array)
        
        # In case of no active layers
        if threshold == 0:
            threshold = torch.tensor(1)

        self.threshold = threshold
        if get_pi:
            return Z, threshold, pi.detach().mean(0)

        if get_intermediate_pi:
            return pi
        
        return Z, pi, threshold, threshold_array

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.num_neurons) + ' x ' + str(self.truncation) + ')'