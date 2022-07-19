# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 08:49:47 2020

Solves the bus problem.

@author: Richard Kohar
"""

import torch as pt  # pt for PyTorch
from inverseGaussian import *    # imports the inverse Gaussian functions
import matplotlib.pyplot as plt  # for plotting

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

pt.manual_seed(1988) # set the seed for generating random numbers

#Parameters for the solver
numBeliefs = 5000
numIter = 100

#The Bus Problem
#beta = 0.3         # (continuous) discount. This is matching a discrete time discount gamma of 0.75
#beta = 0.02        # equivalent discrete time discount gamma of 0.98

beta = 0.02

numStates = 15
# (s,i) = (bus stop, traffic intensity level)
# states: [0] (0,1)  [1] (1,1)  [2] (2,1)  [3] (3,1)  [4] (4,1)   (low traffic)
#         [5] (0,2)  [6] (1,2)  [7] (2,2)  [8] (3,2)  [9] (4,2)   (medium traffic)
#        [10] (0,3) [11] (1,3) [12] (2,3) [13] (3,3) [14] (4,3)   (high traffic)

numActions = 2     # actions: [0] continue, [1] stop

numObs = 5         # observations: [0] bus stop 0 (start), [1] bus stop 1, 
                   #               [2] bus stop 2,         [3] bus stop 3,
                   #               [4] bus stop 4 (final)

xi0 = pt.zeros(1,numStates, device=device)  # initial belief
xi0[0,0] = 1.0/3.0
xi0[0,5] = 1.0/3.0
xi0[0,10] = 1.0/3.0

#Probability transition matrix P(a, s, s') = P(s' | s, a)
P = pt.zeros(numActions, numStates, numStates, device=device)
#action [0] = continue                  (s,i) -> (s+1,i)
P[0,0,1] = 1.0        # low traffic
P[0,1,2] = 1.0
P[0,2,3] = 1.0
P[0,3,4] = 1.0
P[0,4,0] = 1.0/3.0
P[0,4,5] = 1.0/3.0
P[0,4,10] = 1.0/3.0
P[0,5,6] = 1.0        # medium traffic
P[0,6,7] = 1.0
P[0,7,8] = 1.0
P[0,8,9] = 1.0
P[0,9,0] = 1.0/3.0
P[0,9,5] = 1.0/3.0
P[0,9,10] = 1.0/3.0   
P[0,10,11] = 1.0      # high traffic
P[0,11,12] = 1.0
P[0,12,13] = 1.0
P[0,13,14] = 1.0
P[0,14,0] = 1.0/3.0
P[0,14,5] = 1.0/3.0
P[0,14,10] = 1.0/3.0

#action [1] = stop                      (s,i) -> (4,i)
P[1,0,4] = 1.0         # low traffic
P[1,1,4] = 1.0
P[1,2,4] = 1.0
P[1,3,4] = 1.0
P[1,4,0] = 1.0/3.0
P[1,4,5] = 1.0/3.0
P[1,4,10] = 1.0/3.0
P[1,5,9] = 1.0         # medium traffic 
P[1,6,9] = 1.0
P[1,7,9] = 1.0
P[1,8,9] = 1.0
P[1,9,0] = 1.0/3.0
P[1,9,5] = 1.0/3.0
P[1,9,10] = 1.0/3.0
P[1,10,14] = 1.0       # high traffic
P[1,11,14] = 1.0
P[1,12,14] = 1.0
P[1,13,14] = 1.0
P[1,14,0] = 1.0/3.0
P[1,14,5] = 1.0/3.0
P[1,14,10] = 1.0/3.0


#Observation probability matrix G(a, s', o) = G(o | a, s')
G = pt.zeros(numActions, numStates, numObs, device=device)
#action[0] = continue  and action[1] = stop
G[:, 0, 0] = 1.0
G[:, 1, 1] = 1.0
G[:, 2, 2] = 1.0
G[:, 3, 3] = 1.0
G[:, 4, 4] = 1.0
G[:, 5, 0] = 1.0
G[:, 6, 1] = 1.0
G[:, 7, 2] = 1.0
G[:, 8, 3] = 1.0
G[:, 9, 4] = 1.0
G[:, 10, 0] = 1.0
G[:, 11, 1] = 1.0
G[:, 12, 2] = 1.0
G[:, 13, 3] = 1.0
G[:, 14, 4] = 1.0


#Sojourn time distribution parameters
#Remember that the IG distribution cannot take mu=0 as input, otherwise it will return
mu = pt.ones(numActions,numStates,numStates, device=device)
#action [0] = continue
#[action, starting state, landing state]
mu[0,0,1] = 5.0    # low traffic
mu[0,1,2] = 5.0
mu[0,2,3] = 5.0
mu[0,3,4] = 5.0
#Cumulative time match 'times'   25, 20, 15, 10

mu[0,5,6] = 5.0    # medium traffic
mu[0,6,7] = 10.0
mu[0,7,8] = 10.0
mu[0,8,9] = 20.0
#Cumulative time match 'times'   45, 40, 30, 20

mu[0,10,11] = 10.0  # high traffic
mu[0,11,12] = 25.0
mu[0,12,13] = 25.0
mu[0,13,14] = 45.0
#Cumulative time match 'times'   105, 95, 70, 45

Lambda = mu*mu*10.0

times = pt.tensor([30.0, 25.0, 20.0, 12.0])

longDecay = 455.0

fixedTimeConstant = pt.zeros(numActions, numStates, numStates, device=device)
fixedTimeConstant[1,0,4] = times[0] #in minutes   #Biking (low traffic intensity)
fixedTimeConstant[1,1,4] = times[1]
fixedTimeConstant[1,2,4] = times[2]
fixedTimeConstant[1,3,4] = times[3]
fixedTimeConstant[1,4,0] = longDecay#with gamma=0.98 decay, this generates a 10^-4 decay factor between episodes
fixedTimeConstant[1,4,5] = longDecay #455
fixedTimeConstant[1,4,10] = longDecay

fixedTimeConstant[1,5,9] = times[0]              #Biking (medium traffic intensity)
fixedTimeConstant[1,6,9] = times[1]
fixedTimeConstant[1,7,9] = times[2]
fixedTimeConstant[1,8,9] = times[3]
fixedTimeConstant[1,9,0] = longDecay
fixedTimeConstant[1,9,5] = longDecay
fixedTimeConstant[1,9,10] = longDecay

fixedTimeConstant[1,10,14] = times[0]            #Biking (high traffic intensity) 
fixedTimeConstant[1,11,14] = times[1]
fixedTimeConstant[1,12,14] = times[2]
fixedTimeConstant[1,13,14] = times[3]
fixedTimeConstant[1,14,0] = longDecay
fixedTimeConstant[1,14,5] = longDecay
fixedTimeConstant[1,14,10] = longDecay

fixedTimeConstant[0,4,0] = longDecay
fixedTimeConstant[0,4,5] = longDecay
fixedTimeConstant[0,4,10] = longDecay

fixedTimeConstant[0,9,0] = longDecay
fixedTimeConstant[0,9,5] = longDecay
fixedTimeConstant[0,9,10] = longDecay

fixedTimeConstant[0,14,0] = longDecay
fixedTimeConstant[0,14,5] = longDecay
fixedTimeConstant[0,14,10] = longDecay

#Matrix that gives [0] for inverse Gaussian distribution and [1] for deterministic function
timeDist = pt.zeros(numActions, numStates, numStates, device=device, dtype=pt.int8)
timeDist[1,:,:] = 1
#These are the last state to first state to generate the long decay
timeDist[0,4,0] = 1
timeDist[0,4,5] = 1
timeDist[0,4,10] = 1
timeDist[0,9,0] = 1
timeDist[0,9,5] = 1
timeDist[0,9,10] = 1
timeDist[0,14,0] = 1
timeDist[0,14,5] = 1
timeDist[0,14,10] = 1


#Reward R(a, s, s') = R(s,a)
#r1 is the lump sum reward
r1 = pt.zeros(numActions, numStates, device=device)
r1[:,4] = 100
r1[:,9] = 100
r1[:,14] = 100


#r2 is the continuous rate of reward accumulation
r2 = pt.zeros(numActions, numStates, numStates, device=device)

#This should be moved into the class/object created?
M1 = laplaceIG(mu, Lambda, beta) * (timeDist == 0) #[a, s, s']
#M = sojournTime.LambdaOf(beta)  return a matrix with [a,s,s'] with the values all precomputed
M2 = pt.exp(-beta * fixedTimeConstant) * (timeDist == 1) #[a, s, s']
M = M1 + M2
    

#Move to the solver
R = r1 + ((1.0/beta) * pt.sum((P * r2 * (1.0-M)), dim=2)) #[a, s]

#Initialize the value function

Rmin = pt.zeros(numActions,numStates,numStates,device=device)
Rmin,_ = pt.min(Rmin, dim=2)
Rmin,_ = pt.min(Rmin, dim=1)
Rmin,Amin = pt.min(Rmin,dim=0)

alphaMin = Rmin*pt.ones(1,numStates, device=device)
V = alphaMin
Vactions = pt.tensor([Amin], device=device)


class sojournTime:
    '''
    This class can represent any problem whose time distribution is 
    either a fixed discrete time or an inverse gaussian distribution.
    '''

    def __init__(self, timeDist, mu, Lambda, fixedTimeConstant, device):
        '''
        timeDist : Tensor (a,s,s') indicates if it is discrete (2) or inverse
        gaussian (1 Lambda is the ..
        mu is the mean for the inverse Gaussian distribution
        Lambda is the shape parameter for the inverse Gaussian distribution
        fixedTimeConstant is the fixed amount of time for a particular (a,s,s') transition
        device is which device this will run on: either CPU or GPU
        '''
        self.timeDist = timeDist
        self.Lambda = Lambda
        self.mu = mu
        self.fixedTimeConstant = fixedTimeConstant
        self.device = device
        
    def sampleTime(self, action, s1, s2):
        '''
        sampleTime will generate a time sample for a (s,a,s') transition
        from the appropriate probability distribution.
        '''
        if self.timeDist[action, s1, s2] == 0:  #Inverse Gaussian distribution
            time = randomIG(self.mu[action,s1,s2], self.Lambda[action,s1,s2], self.device)
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
        
        igPart = pdfIG(x, self.mu, self.Lambda) * (self.timeDist == 0)
        dtPart = (x == fixedTimeConstant)  * (self.timeDist == 1) * 1.0
        return igPart + dtPart
    

T = sojournTime(timeDist, mu, Lambda, fixedTimeConstant, device)

bus_problem = {
  "P": P,
  "G": G,
  "R": R,
  "V0": V,
  "V0actions": Vactions,
  "sojournTime": T,
  "xi0": xi0,
  "beta" : beta
}

#MAIN SCRIPT

B, C, V, Vactions, minValue, maxValue = solvePOSMDP(bus_problem, numBeliefs, numIter)

import pandas as pd
import numpy as np
from chronosPerseus_solver import valuefunc

def create_uniform_belief_grid(subdiv):
    #subdiv = 11 would give it in 0.1 steps. 0 0.1 0.2 0.3 ... 1.0
    coord = pt.zeros(subdiv**3, 3, device=device)

    i = 0
    for s1 in pt.linspace(0,1,subdiv):
        for s2 in pt.linspace(0,1,subdiv):
            for s3 in pt.linspace(0,1,subdiv):
                coord[i] = pt.tensor([s1, s2, s3])
                i = i + 1
    ind = coord.sum(dim=1) == 1
    coord = coord[ind]
    return coord

#This creates the uniform grid for the latex plot.
coord = create_uniform_belief_grid(11)


V_traffic_stop_intensities = pt.zeros(5,3, device=device, dtype=pt.long) #Tensors used as indices must be long tensors
V_traffic_stop_intensities[0] = pt.tensor([0, 5, 10]) #Bus stop 0 (states associated with low, medium, high)
V_traffic_stop_intensities[1] = pt.tensor([1, 6, 11]) #Bus stop 1 (states associated with low, medium, high)
V_traffic_stop_intensities[2] = pt.tensor([2, 7, 12]) #Bus stop 2 (states associated with low, medium, high)
V_traffic_stop_intensities[3] = pt.tensor([3, 8, 13]) #Bus stop 3 (states associated with low, medium, high)
V_traffic_stop_intensities[4] = pt.tensor([4, 9, 14]) #Bus stop 4 (states associated with low, medium, high)

for j in pt.arange(5): #number of stops: 0, 1, 2, 3, 4
    belief_coord = pt.zeros(coord.size(0),15, device=device)
    belief_coord[:,j] = coord[:,0]
    belief_coord[:,j+5] = coord[:,1]
    belief_coord[:,j+10] = coord[:,2]
    vb_optimal_value, vb_optimal_action = valuefunc(V,Vactions,belief_coord)
    print('stop', j)
    print(vb_optimal_value)
    print(vb_optimal_action)
    
    
    count0 = pt.tensor(0, dtype=pt.int16, device=device)
    count1 = pt.tensor(0, dtype=pt.int16, device=device)
    PolicyAction0 = pt.zeros(coord.size(0), 3, device=device)
    PolicyAction1 = pt.zeros(coord.size(0), 3, device=device)
    for i in pt.arange(coord.size(0)):
        max_action = vb_optimal_action[i]
        if (max_action == 0):
            PolicyAction0[count0] = coord[i]
            count0 = count0 + 1
        elif (max_action == 1):
            PolicyAction1[count1] = coord[i]
            count1 = count1 + 1
       
    PolicyAction0 = PolicyAction0[0:count0]
    PolicyAction1 = PolicyAction1[0:count1]

    PolicyAction0_np = PolicyAction0.cpu().numpy()
    PolicyAction0_df = pd.DataFrame(PolicyAction0_np)
    PolicyAction0_df.columns = ["i1", "i2", "i3"]
    PolicyAction0_df.to_csv("PolicyStop{}Action0.csv".format(j))
       
    PolicyAction1_np = PolicyAction1.cpu().numpy()
    PolicyAction1_df = pd.DataFrame(PolicyAction1_np)
    PolicyAction1_df.columns = ["i1", "i2", "i3"]
    PolicyAction1_df.to_csv("PolicyStop{}Action1.csv".format(j))
       
import os
os.system("pdflatex policyplot.tex")

