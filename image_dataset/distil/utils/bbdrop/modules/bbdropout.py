from distil.utils.bbdrop.modules.gated_layers import Gate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import math

class BBDropout(Gate):
    def __init__(self, num_gates,
            alpha=1e-4, a_uc_init=-1.0, thres=1e-1, kl_scale=1.0):
        super(BBDropout, self).__init__(num_gates)
        self.alpha = alpha
        self.thres = thres
        self.num_gates = num_gates
        self.a_uc_init = a_uc_init
        self.kl_scale = kl_scale
        self.reset_params()
        # self.a_uc = nn.Parameter(a_uc_init*torch.ones(num_gates))
        # self.b_uc = nn.Parameter(0.5413*torch.ones(num_gates))
        
    def reset_params(self):
        self.a_uc = nn.Parameter(self.a_uc_init*torch.ones(self.num_gates))
        self.b_uc = nn.Parameter(0.5413*torch.ones(self.num_gates))

    def stop_training(self):
        self.a_uc.requires_grad = False
        self.b_uc.requires_grad = False

    def get_params(self):
        a = F.softplus(self.a_uc.clamp(min=-10.))
        b = F.softplus(self.b_uc.clamp(min=-10., max=50.))
        return a, b

    def sample_pi(self, samples=1):
        a, b = self.get_params()
        # u = torch.rand((samples, self.num_gates)).clamp(1e-6, 1-1e-6)
        # u = u.mean(0)
        u = torch.rand(self.num_gates).clamp(1e-6, 1-1e-6)
        if torch.cuda.is_available():
            u = u.cuda()
        return (1 - u.pow_(1./b)).pow_(1./a)

    def get_Epi(self):
        a, b = self.get_params()
        Epi = b*torch.exp(torch.lgamma(1+1./a) + torch.lgamma(b) \
                - torch.lgamma(1+1./a + b))
        return Epi

    def get_weight(self, x):
        if self.training:
            pi = self.sample_pi()
            temp = torch.Tensor([0.1])
            if torch.cuda.is_available():
                temp = temp.cuda()
            p_z = RelaxedBernoulli(temp, probs=pi)
            z = p_z.rsample(torch.Size([x.size(0)]))
        else:
            Epi = self.get_Epi()
            mask = self.get_mask(Epi=Epi)
            z = Epi * mask
        return z

    def get_mask(self, Epi=None):
        Epi = self.get_Epi() if Epi is None else Epi
        return (Epi >= self.thres).float()

    def get_reg(self, base):
        a, b = self.get_params()
        kld = (1 - self.alpha/a)*(-0.577215664901532 \
                - torch.digamma(b) - 1./b) \
                + torch.log(a*b + 1e-10) - math.log(self.alpha) \
                - (b-1)/b
        kld = self.kl_scale*kld.sum()
        return kld
