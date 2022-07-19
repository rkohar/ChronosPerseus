# Truncated Normal distribution when b = infty (for PyTorch)
# 28 Feb 2021: Richard Kohar

import math
import torch as pt
from torch.distributions import Normal

pt.set_grad_enabled(False)

def pdfGaussian(x, mu, sigma, device):
    # Input:
    # x is the tensor data
    # mu is the mean
    # sigma is the standard deviation
    # device is the GPU or CPU
    #
    # Output:
    # pdf is the pdf of the data x.
    
    pdf = 1.0/pt.sqrt(2.0*math.pi*(sigma**2.0))*pt.exp(-((x - mu)**2.0)/(2.0*(sigma**2.0)))
    
    return pdf

def laplaceTruncGaussLeft(mu, sigma, beta, a, device):
    # Input:
    # mu is a scalar
    # sigma is a scalar
    # beta is a scalar
    # a is a scalar
    # device is if it's on the GPU or CPU
    
    m = Normal(pt.tensor(0.0, device=device), pt.tensor(1.0, device=device))
    
    I = pt.exp(((sigma**2.0 * beta**2.0) / 2.0) - (mu * beta)) * ((1 - m.cdf(((a-mu)/sigma) + (sigma*beta)))/(1 - m.cdf((a - mu)/sigma))) 
    
    return I

def pdfTruncGauss(x, mu, sigma, a, b, device):
    # Input:
    # x is a tensor
    # mu is a scalar/tensor
    # sigma is a scalar/tensor
    # a is the left truncation point
    # b is the right truncation point
    # device is if it is a GPU or CPU
    #
    # Output:
    # pdf is the pdf of the data x.
    
    assert a < b, 'a is not less than b'  
    
    m = Normal(pt.tensor(0.0, device=device), pt.tensor(1.0, device=device))
        
    pdf = pdfGaussian(x, mu, sigma, device) / (m.cdf((b - mu)/sigma) - m.cdf((a - mu)/sigma))
    
    xind = x > a
    xind2 = x < b
    pdf = pdf * (xind*xind2)
        
    return pdf

# This is for a Gaussian distribution that is truncated on the left hand side and allow
# the right hand side to go to infinity.
def pdfTruncGaussLeft(x, mu, sigma, a, device):
    # Input:
    # x is a tensor
    # mu is a scalar/tensor
    # sigma is a scalar/tensor
    # a is the left truncation point
    # device is if it is a GPU or CPU
    # 
    # Output:
    # pdf is a tensor
    
    m = Normal(pt.tensor(0.0, device=device), pt.tensor(1.0, device=device))
        
    pdf = pdfGaussian(x, mu, sigma, device) / (1.0 - m.cdf((a - mu)/sigma))
    
    xind = x > a
    
    pdf = pdf * xind
    
    return pdf

# This is for a Gaussian distribution that is truncated on the right hand side and allow
# the left hand side to go to infinity.
def pdfTruncGaussRight(x, mu, sigma, b, device):
    # Input:
    # x is a tensor
    # mu is a scalar/tensor
    # sigma is a scalar/tensor
    # b is the right truncation point
    # device is if it is a GPU or CPU
    # 
    # Output:
    # pdf is a tensor
    
    m = Normal(pt.tensor(0.0, device=device), pt.tensor(1.0, device=device))
        
    pdf = pdfGaussian(x, mu, sigma, device) / (m.cdf((b - mu)/sigma) - 1.0)
    
    xind = x < b
    
    pdf = pdf * xind
    
    
    return pdf

def randomTruncGaussLeft(mu, sigma, a, max_rejections=10000):
    # Input:
    # mu is a scalar/tensor
    # sigma is a scalar/tensor
    # a is the left truncation point
    #
    # Output: 
    # sampleTruncGaussLeft is a tensor
    
    # Sample X ~ N(x | mu, sigma^2, )
    
    rejections = 0
    
    m = Normal(mu, sigma)
    
    while True:
        sampleTruncGaussLeft = m.sample()
        if a <= sampleTruncGaussLeft:
            return sampleTruncGaussLeft
        rejections = rejections + 1
        if rejections > max_rejections:
            assert False, 'Too many rejections'

def randomTruncGaussRight(mu, sigma, b, max_rejections=10000):
    # Input:
    # mu is a scalar/tensor
    # sigma is a scalar/tensor
    
    # b is the right truncated point
    #
    # Output: 
    # sampleTruncGaussRight is a tensor
    
    # Sample X ~ N(x | mu, sigma^2, )
    
    rejections = 0
    
    m = Normal(mu, sigma)
    
    while True:
        sampleTruncGaussRight = m.sample()
        if sampleTruncGaussRight <= b:
            return sampleTruncGaussRight
        rejections = rejections + 1
        if rejections > max_rejections:
            assert False, 'Too many rejections'

def randomTruncGauss(mu, sigma, a, b, device, max_rejections=10000):
    # Input:
    # mu is a scalar/tensor
    # sigma is a scalar/tensor
    # a is the left truncation point
    # b is the left truncation point
    #
    # Output: 
    # sampleTruncGaussLeft is a tensor
    
    # Sample X ~ N(x | mu, sigma^2)
    
    rejections = 0
    
    m = Normal(mu, sigma)
    
    while True:
        sampleTruncGauss = m.sample()
        if a <= sampleTruncGauss <= b:
            return sampleTruncGauss
        rejections = rejections + 1
        if rejections > max_rejections:
            assert False, 'Too many rejections'