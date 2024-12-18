from modules.gated_layers import Gate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import math

class DBBDropout(Gate):
    def __init__(self, num_gates,
            beta_init=1.0, thres=1e-3, rho=math.sqrt(5.0), kl_scale=1.0):
        super(DBBDropout, self).__init__(num_gates)
        self.thres = thres
        self.rho = rho
        self.kl_scale = kl_scale
        self.bn = nn.BatchNorm1d(num_gates)
        self.bn.bias.data.fill_(beta_init)
        self.sigma_uc = nn.Parameter(0.5413*torch.ones(num_gates))
        self.num_active = 0.
        self.counter = 0

    def reset(self):
        self.num_active = 0.
        self.counter = 0

    def forward(self, x, z_in):
        if len(x.size()) == 4:
            h = F.avg_pool2d(x, [x.size(2), x.size(3)])
            h = h.view(h.size(0), -1)
        else:
            h = x
        h = self.bn(h.detach())

        if self.training:
            sigma = F.softplus(self.sigma_uc)
            eps = torch.randn(self.num_gates)
            if torch.cuda.is_available():
                eps = eps.cuda()
            h = h + sigma * eps
            temp = torch.Tensor([0.1])
            if torch.cuda.is_available():
                temp = temp.cuda()
            p_z = RelaxedBernoulli(temp, probs=h.clamp(1e-10, 1-1e-10))
            z = p_z.rsample()
        else:
            z = h.clamp(1e-10, 1-1e-10)

        if len(x.size()) == 4:
            z = z.view(-1, self.num_gates, 1, 1)
            z_in = z_in.view(-1, self.num_gates, 1, 1)
        else:
            z_in = z_in.view(-1, self.num_gates)
        z = z * z_in

        if not self.training:
            num_active = (z > self.thres).float().sum(1).mean(0).item()
            self.num_active = (self.num_active*self.counter + num_active) / \
                    (self.counter + 1)
            self.counter += 1
            z[z <= self.thres] = 0.

        return x * z

    def get_reg(self, base):
        sigma = F.softplus(self.sigma_uc)
        kld = torch.log(self.rho/sigma) + 0.5*(sigma**2 + self.bn.bias**2)/self.rho**2
        return self.kl_scale*kld.sum()
