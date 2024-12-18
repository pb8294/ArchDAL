from modules.gated_layers import Gate
import torch
from torch.nn import Parameter
import math

def phi(x):
    return 0.5*torch.erfc(-x/math.sqrt(2.0))

erfinv = torch.erfinv

def erfcx(x):
    return (torch.log(torch.erfc(x)) + x*x).exp()

#def erfcx(x):
#    """M. M. Shepherd and J. G. Laframboise,
#       MATHEMATICS OF COMPUTATION 36, 249 (1981)
#    """
#    K = 3.75
#    y = (torch.abs(x)-K) / (torch.abs(x)+K)
#    y2 = 2.0*y
#    (d, dd) = (-0.4e-20, 0.0)
#    (d, dd) = (y2 * d - dd + 0.3e-20, d)
#    (d, dd) = (y2 * d - dd + 0.97e-19, d)
#    (d, dd) = (y2 * d - dd + 0.27e-19, d)
#    (d, dd) = (y2 * d - dd + -0.2187e-17, d)
#    (d, dd) = (y2 * d - dd + -0.2237e-17, d)
#    (d, dd) = (y2 * d - dd + 0.50681e-16, d)
#    (d, dd) = (y2 * d - dd + 0.74182e-16, d)
#    (d, dd) = (y2 * d - dd + -0.1250795e-14, d)
#    (d, dd) = (y2 * d - dd + -0.1864563e-14, d)
#    (d, dd) = (y2 * d - dd + 0.33478119e-13, d)
#    (d, dd) = (y2 * d - dd + 0.32525481e-13, d)
#    (d, dd) = (y2 * d - dd + -0.965469675e-12, d)
#    (d, dd) = (y2 * d - dd + 0.194558685e-12, d)
#    (d, dd) = (y2 * d - dd + 0.28687950109e-10, d)
#    (d, dd) = (y2 * d - dd + -0.63180883409e-10, d)
#    (d, dd) = (y2 * d - dd + -0.775440020883e-09, d)
#    (d, dd) = (y2 * d - dd + 0.4521959811218e-08, d)
#    (d, dd) = (y2 * d - dd + 0.10764999465671e-07, d)
#    (d, dd) = (y2 * d - dd + -0.218864010492344e-06, d)
#    (d, dd) = (y2 * d - dd + 0.774038306619849e-06, d)
#    (d, dd) = (y2 * d - dd + 0.4139027986073010e-05, d)
#    (d, dd) = (y2 * d - dd + -0.69169733025012064e-04, d)
#    (d, dd) = (y2 * d - dd + 0.490775836525808632e-03, d)
#    (d, dd) = (y2 * d - dd + -0.2413163540417608191e-02, d)
#    (d, dd) = (y2 * d - dd + 0.9074997670705265094e-02, d)
#    (d, dd) = (y2 * d - dd + -0.26658668435305752277e-01, d)
#    (d, dd) = (y2 * d - dd + 0.59209939998191890498e-01, d)
#    (d, dd) = (y2 * d - dd + -0.84249133366517915584e-01, d)
#    (d, dd) = (y2 * d - dd + -0.4590054580646477331e-02, d)
#    d = y * d - dd + 0.1177578934567401754080e+01
#    result = d/(1.0+2.0*torch.abs(x))
#    result[torch.isnan(result)] = 1
#    result[torch.isinf(result)] = 1
#
#    negative_mask = (x < 0.0).float()
#    positive_mask = (x >= 0.0).float()
#    negative_result = 2.0*torch.exp(x*x)-result
#    negative_result[torch.isnan(negative_result)] = 1
#    negative_result[torch.isinf(negative_result)] = 1
#    result = negative_mask * negative_result + positive_mask * result
#    return result

def phi_inv(x):
    return math.sqrt(2.0)*erfinv(2.0*x - 1)

class SBP(Gate):
    def __init__(self, num_gates,
            min_log=-20.0, max_log=0.0, thres=1.0, kl_scale=1.0):
        super(SBP, self).__init__(num_gates)
        self.min_log = min_log
        self.max_log = max_log
        self.thres = thres
        self.kl_scale = kl_scale
        self.mu = Parameter(torch.zeros(num_gates))
        self.log_sigma = Parameter(-5*torch.ones(num_gates))

    def _mean_truncated_log_normal(self):
        a, b = self.min_log, self.max_log
        mu = self.mu.clamp(-20.0, 5.0)
        log_sigma = self.log_sigma.clamp(-20.0, 5.0)
        sigma = log_sigma.exp()

        alpha = (a - mu)/sigma
        beta = (b - mu)/sigma
        z = phi(beta) - phi(alpha)
        mean = erfcx((sigma-beta)/math.sqrt(2.0))*torch.exp(b-beta*beta/2)
        mean = mean - erfcx((sigma-alpha)/math.sqrt(2.0))*torch.exp(a-alpha*alpha/2)
        mean = mean/(2*z)
        return mean

    def _snr_truncated_log_normal(self):
        a, b = self.min_log, self.max_log
        mu = self.mu.clamp(-20.0, 5.0)
        log_sigma = self.log_sigma.clamp(-20.0, 5.0)
        sigma = log_sigma.exp()

        alpha = (a - mu)/sigma
        beta = (b - mu)/sigma
        z = phi(beta) - phi(alpha)
        ratio = erfcx((sigma-beta)/math.sqrt(2.0))*torch.exp((b-mu)-beta**2/2.0)
        ratio = ratio - erfcx((sigma-alpha)/math.sqrt(2.0))*torch.exp((a-mu)-alpha**2/2.0)
        denominator = 2*z*erfcx((2.0*sigma-beta)/math.sqrt(2.0))*torch.exp(2.0*(b-mu)-beta**2/2.0)
        denominator = denominator - 2*z*erfcx((2.0*sigma-alpha)/math.sqrt(2.0))\
                                       *torch.exp(2.0*(a-mu)-alpha**2/2.0)
        denominator = denominator - ratio**2
        ratio = ratio/torch.sqrt(denominator)

        return ratio

    def _sample_truncated_normal(self):
        a, b = self.min_log, self.max_log
        mu = self.mu.clamp(-20.0, 5.0)
        log_sigma = self.log_sigma.clamp(-20.0, 5.0)
        sigma = torch.exp(log_sigma)

        alpha = (a - mu)/sigma
        beta = (b - mu)/sigma
        u = torch.rand(self.num_gates)
        if torch.cuda.is_available():
            u = u.cuda()
        gamma = phi(alpha)+u*(phi(beta)-phi(alpha))
        return (phi_inv(gamma.clamp(1e-5, 1-1e-5))*sigma + mu).clamp(a, b).exp()

    def get_mask(self):
        snr = self._snr_truncated_log_normal()
        return (snr > self.thres).float()

    def get_weight(self, x):
        if self.training:
            z = self._sample_truncated_normal()
        else:
            Etheta = self._mean_truncated_log_normal()
            mask = self.get_mask()
            z = Etheta * mask
        return z

    def get_reg(self, base):
        a, b = self.min_log, self.max_log
        mu = self.mu.clamp(-20.0, 5.0)
        log_sigma = self.log_sigma.clamp(-20.0, 5.0)
        sigma = log_sigma.exp()

        alpha = (a - mu)/sigma
        beta = (b - mu)/sigma
        z = phi(beta) - phi(alpha)

        def pdf(x):
            return torch.exp(-x*x/2.0)/math.sqrt(2.0*math.pi)

        kld = -log_sigma-torch.log(z)-(alpha*pdf(alpha)-beta*pdf(beta))/(2.0*z)
        kld += math.log(self.max_log-self.min_log) - math.log(2.0*math.pi*math.e)/2.0
        kld = self.kl_scale * kld.sum()
        return kld
