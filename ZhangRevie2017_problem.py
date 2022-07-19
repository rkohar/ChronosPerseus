# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 00:22:25 2021

@author: Richard Kohar
"""

import torch as pt  # pt for PyTorch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from truncatedGaussian import * # imports my truncated Gaussian functions

from chronosPerseus_solver import solvePOSMDP

pt.utils.backcompat.broadcast_warning.enabled=True
pt.backends.cudnn.deterministic=True
pt.set_grad_enabled(False)

#Setting device on GPU if available, else CPU
#device = pt.device('cpu')
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional information when using cuda
if device.type == 'cuda':
    print(pt.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(pt.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(pt.cuda.memory_cached(0)/1024**3,1), 'GB')

pt.manual_seed(1988) # set the seed for generating random numbers  #Richard 1988

#Parameters for the solver
numBeliefs = 5000  #in paper on July 7th, 10000, 5000 same answers
numIter = 40       #in paper on July 7th, 40, 60 (&5000) slightly higher, 80 much better number & different actions
#The Rapid Gravity Filter Problem from Zhang and Revie (2017) paper
Beta = 0.01

numStates = 4   # 0 = good, 1 = acceptable, 2 = poor, 3 = awful
numActions = 4  # 0 = do nothing, 1 = backwash, 2 = dose chemicals, 3 = replace
numObs = 100    # Discretize the interval [0,1] into numObs evenly spaced numbers
                #in paper on July 7th, 100, 1000 (&5000) gives slightly higher results but not much

#Initial belief
xi0 = pt.zeros(1,numStates, device=device)  # initial belief !! This was not mentioned in the paper
xi0[0,0] = 1.0

#Probability transition matrix P(a, s, s') = P(s' | s, a)
P = pt.zeros(numActions, numStates, numStates, device=device)
P[0] = pt.tensor([[0.1043, 0.7413, 0.1493, 0.0051],    #Zhang(2017)), p. 211
                  [0.0000, 0.1043, 0.7413, 0.1544],
                  [0.0000, 0.0000, 0.1043, 0.8957],
                  [0.0000, 0.0000, 0.0000, 1.0000]])

P[1] = P[0]  #It's strange that a=do nothing and a=backwash has the same transition probability matrix.
P[2] = pt.tensor([[1.0000, 0.0000, 0.0000, 0.0000],
                  [0.5000, 0.5000, 0.0000, 0.0000],
                  [0.2500, 0.7000, 0.0500, 0.0000],
                  [0.2000, 0.5500, 0.2000, 0.0500]])
P[3, :, 0] = 1.0

#Observation probability matrix G(a, s', o) = G(o | a, s')
G = pt.zeros(numActions, numStates, numObs, device=device)
#action[0] = continue  and action[1] = stop

#"The observation function depends only on the RGF's state, not on the action." (Zhang(2017), p. 211)
# This means that the observation depends only the initial state s and not on
# action a or the landing state s'. This is NOT how Chronos Perseus works: it 
# assumes that the observation function gives the probability of the observation o
# given the action a and the landing state s'.

o = pt.linspace(0, 1, steps=numObs+2, device=device)
o = o[1:-1]
o_numpy = o.cpu().numpy()

o_pdf = pt.tensor(beta.pdf(o_numpy, 2.0, 18.0), device=device)   #[o] size is numObs
G[:, 0, :] = o_pdf / o_pdf.sum()                                 # the division by the sum of the vector normalizes it


o_pdf = pt.tensor(beta.pdf(o_numpy, 6.0, 18.0), device=device)
G[:, 1, :] = o_pdf / o_pdf.sum()

o_pdf = pt.tensor(beta.pdf(o_numpy, 18.0, 18.0), device=device)
G[:, 2, :] = o_pdf / o_pdf.sum()

o_pdf = pt.tensor(beta.pdf(o_numpy, 18.0, 6.0), device=device)
G[:, 3, :] = o_pdf / o_pdf.sum()

fixedTimeConstant = pt.zeros(numActions, numStates, numStates, device=device)
fixedTimeConstant[0, :, :] = 78.7433
fixedTimeConstant[1, :, :] = 85.3052
fixedTimeConstant[2, :, :] = 3.0

#Matrix that gives [0] for truncated Gaussian distribution and [1] for deterministic function
timeDist = pt.ones(numActions, numStates, numStates, device=device, dtype=pt.int8)
timeDist[3,:,:] = 0

#The values for the truncated Gaussian distribution
mu = pt.ones(numActions, numStates, numStates, device=device)
mu[3,:,:] = 10.0
sigma = pt.ones(numActions, numStates, numStates, device=device)
sigma[3,:,:] = 1.5
a = 0.0 # a is the left truncation point for the truncated Gaussian distribution

#Reward R(a, s, s') = R(s,a)
#r1 is the lump sum reward
r1 = pt.zeros(numActions, numStates, device=device)
#r1[a,s]
r1[0,:] = 0.0
r1[1,:] = -100.0
r1[2,:] = -200.0
r1[3,:] = -500.0

#r2 is te continuous rate of reward accumulation
r2 = pt.zeros(numActions, numStates, numStates, device=device)
#r2[a,s,s']
r2[0,0,:] = 500.0
r2[0,1,:] = 250.0
r2[0,2,:] = -300.0
r2[0,3,:] = -500.0
r2[1,0,:] = 500.0
r2[1,1,:] = 250.0
r2[1,2,:] = -300.0
r2[1,3,:] = -500.0
r2[2,:,:] = -100.0
r2[3,:,:] = -100.0

M1 =  laplaceTruncGaussLeft(mu, sigma, Beta, a, device=device) * (timeDist == 0) #[a, s, s']
M2 = pt.exp(-Beta * fixedTimeConstant) * (timeDist == 1) #[a, s, s']
M = M1 + M2
    
R = r1 + ((1.0/Beta) * pt.sum((P * r2 * (1.0-M)), dim=2)) #[a, s]

#Initialize the value function
Rmin = -10e6 * pt.ones(numActions,numStates,numStates,device=device)
Rmin,_ = pt.min(Rmin, dim=2)
Rmin,_ = pt.min(Rmin, dim=1)
Rmin,Amin = pt.min(Rmin,dim=0)

alphaMin = Rmin*pt.ones(1,numStates, device=device)
V = alphaMin
Vactions = pt.tensor([Amin], device=device)

class sojournTime:
    '''
    This class can represent any problem whose time distribution is 
    either a fixed discrete time or a truncated gaussian distribution.
    '''

    def __init__(self, timeDist, mu, sigma, a, fixedTimeConstant, device):
        '''
        timeDist : Tensor (a,s,s') indicates if it is discrete (1) or truncated Gaussian (0)
        mu is the mean for the truncated Gaussian distribution
        sigma is the shape parameter for the truncated Gaussian distribution
        a is the left truncation point for the truncated Gaussian distribution
        fixedTimeConstant is the fixed amount of time for a particular (a,s,s') transition
        device is which device this will run on: either CPU or GPU
        '''
        self.timeDist = timeDist
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.fixedTimeConstant = fixedTimeConstant
        self.device = device
        
    def sampleTime(self, action, s1, s2):
        '''
        sampleTime will generate a time sample for a (s,a,s') transition
        from the appropriate probability distribution.
        '''
        if self.timeDist[action, s1, s2] == 0:  #Truncated Gaussian distribution
            time = randomTruncGaussLeft(self.mu[action,s1,s2], self.sigma[action,s1,s2], self.a)
        elif self.timeDist[action, s1, s2] == 1: #Degenerate distribution
            time = fixedTimeConstant[action, s1, s2]
        return time
    
    def pdf(self, x):
        '''
        pdf will return the likelihood of a sojourn time sample, x, for all
        possible (s,a,s') transitions.
        
        '''        
        
        #NaN should not be allowed for an invalid transition. Put an arbitrary 
        #number and this will be masked out by the transition probability = 0.
        
        igPart = pdfTruncGaussLeft(x, self.mu, self.sigma, self.a, self.device) * (self.timeDist == 0)
        dtPart = (x == fixedTimeConstant)  * (self.timeDist == 1) * 1.0
        return igPart + dtPart
    
T = sojournTime(timeDist, mu, sigma, a, fixedTimeConstant, device)

ZhangRevie2017_problem = {
  "P": P,
  "G": G,
  "R": R,
  "V0": V,
  "V0actions": Vactions,
  "sojournTime": T,
  "xi0": xi0,
  "beta" : Beta
}

B, C, V, Vactions, minValue, maxValue = solvePOSMDP(ZhangRevie2017_problem, numBeliefs, numIter)



# This is the code to run the problem and check the sample beliefs that were
# in Zhang and Revie (2017).
import torch as pt
from chronosPerseus_solver import valuefunc

sample_beliefs = pt.zeros(8,4, device=device)
sample_beliefs[0] = pt.tensor([0.9972, 0.0028, 0.0000, 0.0000])
sample_beliefs[1] = pt.tensor([0.9965, 0.0035, 0.0000, 0.0000])
sample_beliefs[2] = pt.tensor([0.8714, 0.1286, 0.0000, 0.0000])
sample_beliefs[3] = pt.tensor([0.8160, 0.1840, 0.0000, 0.0000])
sample_beliefs[4] = pt.tensor([0.0031, 0.6803, 0.3165, 0.0001])
sample_beliefs[5] = pt.tensor([0.0001, 0.0390, 0.9457, 0.0152])
sample_beliefs[6] = pt.tensor([0.0000, 0.0003, 0.8488, 0.1509])
sample_beliefs[7] = pt.tensor([0.0000, 0.0000, 0.0000, 1.0000])
#sample_beliefs[8] = pt.tensor([1.0000, 0.0000, 0.0000, 0.0000])

vb = pt.zeros(sample_beliefs.size(0), device=device)
vb_optimal_value = pt.zeros(sample_beliefs.size(0))
vb_optimal_action = pt.zeros(sample_beliefs.size(0))

vb_optimal_value, vb_optimal_action = valuefunc(V,Vactions,sample_beliefs)

print(vb_optimal_value)
print(vb_optimal_action)
print(numBeliefs, numObs, numIter)