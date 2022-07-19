# Inverse Gaussian distribution (for PyTorch)
# 08 Jun 2020: Richard Kohar

import math
import torch as pt
from torch.distributions import Normal
from torch.distributions import Uniform

pt.set_grad_enabled(False)

def pdfIG(x, mu, Lambda):
    # Input:
    # x is a tensor
    # mu is a scalar/tensor
    # Lambda is a scalar/tensor
    # 
    # Output:
    # pdf is a tensor
    pdf = pt.sqrt(Lambda/(2.0*math.pi*x**3.0))*pt.exp(-(Lambda*
                 (x-mu)**2.0)/(2.0*mu**2.0*x))
    return pdf

def cdfIG(x, mu, Lambda, device):
    # Input:
    # x is a tensor
    # mu is a scalar/tensor
    # Lambda is a scalar/tensor
    # 
    # Output:
    # cdf is a tensor
    m = Normal(pt.tensor(0.0, device=device), pt.tensor(1.0, device=device))
    
    cdf = m.cdf(pt.sqrt(Lambda/x)*(x/mu - 1.0)) + pt.exp(2.0*
               Lambda/mu)*m.cdf(-pt.sqrt(Lambda/x)*(x/mu + 1.0))
    return cdf

def randomIG(mu, Lambda, device):
    # Input:
    # mu is a scalar/tensor
    # Lambda is a scalar/tensor
    #
    # Output:
    # sampleIG is a tensor
    
    # Generate a normal distribution
    m = Normal(pt.tensor(0.0, device=device), pt.tensor(1.0, device=device))
    nu = m.sample()
    
    y = nu*nu
    
    x = mu + (mu*mu * y / (2.0*Lambda)) - (mu/(2.0*Lambda))*pt.sqrt(4.0*mu*
             Lambda*y + mu*mu * y*y)
    
    u = Uniform(pt.tensor([0.0], device=device), pt.tensor([1.0], device=device))
    z = u.sample()
      
    ind = (z <= (mu/(mu + x))).float()
    
    sampleIG = (ind*x) + ((1.0-ind)*((mu*mu)/x)).squeeze(dim=0)
    
    return sampleIG

def laplaceIG(mu, Lambda, beta):
    # Input:
    # mu is a scalar
    # Lambda is a scalar
    # beta is a scalar
    #
    # Output:
    # I is a scalar/tensor
    I = pt.exp(Lambda/mu*(1.0 - pt.sqrt(1.0 + ((2.0*mu*mu*beta)/Lambda))))
    
    return I