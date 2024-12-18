from modules.gated_layers import Gate
import torch
import torch.nn as nn

class VIB(Gate):
    def __init__(self, num_gates,
            init_mag=9, init_var=0.01, thres=0, kl_scale=1.0):
        super(VIB, self).__init__(num_gates)
        self.thres = thres
        self.kl_scale = kl_scale
        self.mu = nn.Parameter(1.+init_var*torch.randn(num_gates))
        self.logvar = nn.Parameter(-init_mag+init_var*torch.randn(num_gates))

    def get_mask(self):
        log_alpha = self.logvar - (self.mu.pow(2)+1e-8).log_()
        return (log_alpha < self.thres).float()

    def get_weight(self, x):
        if self.training:
            sigma = self.logvar.mul(0.5).exp()
            eps = torch.randn(x.size(0), self.num_gates)
            if torch.cuda.is_available():
                eps = eps.cuda()
            z = self.mu + sigma*eps
        else:
            z = self.mu * self.get_mask()
        return z

    def get_reg(self, base):
        kld = .5*(1 + self.mu.pow(2)/(self.logvar.exp()+1e-8)).log().sum()
        kld = self.kl_scale * kld
        return kld
