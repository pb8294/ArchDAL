from modules.gated_layers import Gate
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

limit_a, limit_b, tol = -.1, 1.1, 1e-6

class L0Reg(Gate):
    def __init__(self, num_gates,
            droprate_init=0.5, temp=2./3., weight_decay=1., lamb=1):
        super(L0Reg, self).__init__(num_gates)
        log_a_init = math.log(1 - droprate_init) - math.log(droprate_init)
        self.log_a = Parameter(log_a_init + 1e-2*torch.randn(num_gates))
        self.temp = temp
        self.weight_decay = weight_decay
        self.lamb = lamb

    def clamp(self):
        self.log_a.data.clamp(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        y = torch.sigmoid(logits * self.temp - self.log_a)
        return y.clamp(min=tol, max=1-tol)

    def quantile_concrete(self, x):
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + self.log_a) / self.temp)
        return y * (limit_b - limit_a) + limit_a

    def get_mask(self):
        pi = torch.sigmoid(self.log_a)
        return F.hardtanh(pi*(limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def get_weight(self, x):
        if self.training:
            eps = tol + (1 - tol)*torch.rand(self.num_gates)
            if torch.cuda.is_available():
                eps = eps.cuda()
            z = F.hardtanh(self.quantile_concrete(eps), min_val=0, max_val=1)
        else:
            z = self.get_mask()
        return z

    def get_reg(self, base):
        if len(base.weight.size()) == 4:
            weight = base.weight
            q0 = self.cdf_qz(0)
            logpw_col = torch.sum(.5*self.weight_decay*weight.pow(2)+self.lamb, 3).sum(2).sum(1)
            logpw = torch.sum((1 - q0) * logpw_col)
            bias = getattr(base, 'bias', None)
            logpb = 0 if bias is None else \
                    torch.sum((1-q0) * (.5*self.weight_decay*bias.pow(2)+self.lamb))
        else:
            weight = base.weight.t()
            logpw_col = torch.sum(.5*self.weight_decay*weight.pow(2)+self.lamb, 1)
            logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
            bias = getattr(base, 'bias', None)
            logpb = 0 if bias is None else torch.sum(.5*self.weight_decay*bias.pow(2))
        return logpw + logpb
