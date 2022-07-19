# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:31:29 2020

Chronos Perseus: A point-based POSMDP solver

@author: Richard Kohar
"""

import torch as pt  # pt for PyTorch
from inverseGaussian import *    # imports my IG functions

pt.utils.backcompat.broadcast_warning.enabled=True

pt.backends.cudnn.deterministic=True

pt.set_grad_enabled(False)

# CollectBeliefs
def collectBeliefs(P, G, xi0, sojournTime, numStates, numActions, numBeliefs, device):
    # Input:
    # P[a,s,s] or [numActions, numStates, numStates]
    # G[a,s,o] or [numActions, numStates, numObs]
    # xi0[s] or [numStates]
    # sojournTime is a class that represents the time distribution
    #   methods: sampleTime, pdf
    # numStates: number of states
    # numActions: number of actions
    # numBeliefs: number of beliefs to generate
    # device: cpu or gpu
    
    # Output:
    # B set of beliefs
    # C set of sampled sojourn times
    # w the proportion of sojourn times that came from each s,a,s' transition
    # f is the likelihood of each sampled sojourn times
    
    B = pt.zeros(numBeliefs, numStates, device=device)
    C = pt.zeros(numBeliefs, device=device)
    B[0] = xi0
    w = pt.zeros(numActions,numStates,numStates, device=device)
    f = pt.zeros(numBeliefs, numActions, numStates, numStates, device=device) #f(tau_n | s, a, s')  -- [xi, a, s, s]
    a = pt.randint(numActions,(numBeliefs,1), device=device)  #Randomly select an action a (preselect all actions)
    for b in pt.arange(1,numBeliefs):
        xiOld = B[pt.randint(b,(1,))]      #Randomly select a belief from set B
        s1 = pt.multinomial(xiOld, 1).squeeze()              #Generate state s from belief distribution xi
                                                             #s1 is a scalar tensor / torch.Size([])       
        #Action is already randomly generated       
        s2 = pt.multinomial(P[a[b], s1, :], 1).squeeze()     #Generate state s' according to probability transition matrix for P(.|s,a)
                                                             #s2 is a scalar tensor / torch.Size([])
        C[b] = sojournTime.sampleTime(a[b],s1,s2)            #Generate sojourn time
        f[b,:,:,:] = sojournTime.pdf(C[b])        #[tau, a, s, s']  It is [b,a,s,s'] but at the end, we reduce to unique beliefs, but the number of times will remain the same.
        w[a[b], s1, s2] = w[a[b], s1, s2] + 1
        
        #Randomly select an observation o according to P(o | xi, a, tau)
        myG = G[a[b]]  #[1,s',o]  slice along action (don't worry about dim 0, because we'll use that for s)
        myXi = xiOld.t().unsqueeze(dim=1) #[s,1,1]
        myP = P[a[b]].squeeze(dim=0).unsqueeze(dim=2)   #[1,s,s'] -> [s,s'] -> [s,s',1]
        myf = f[b,a[b]].squeeze(dim=0).unsqueeze(dim=2) #[1,s,s'] -> [s,s'] -> [s,s',1]
        temp = myG * myXi * myP * myf                   #[1,s',o] * [s,1,1] * [s,s',1] * [s,s',1] = [s,s',o]
        temp = pt.sum(temp, dim=0)                      #Sum along s, [s,s',o] -> [s',o]
        Pxiaotau = pt.sum(temp,dim=0)                   #Sum along s' [s',o] -> [o] (row vector)
        o = pt.multinomial(Pxiaotau, 1)
        
        #Calculate the new belief
        xiNew = temp[:,o] #[s'] (in column vector)
        B[b] = xiNew.t()/Pxiaotau[o] #[s'].t()/Normalization constant  [1,s]
    w = w/pt.sum(w)     #normalize w
    B = B.unique(dim=0) #reduces the number of beliefs if redundant
    C = C[1:]           #removes the zeroth term (which is not generated)
    f = f[1:]           #removes the zeroth term (which is not generated)
    
    #We use these when running the ZhangRevie2017_problem to ensure that we have the same
    #beliefs to compare results.
    
    #sample_beliefs = pt.zeros(9,4, device=device)
    #sample_beliefs[0] = pt.tensor([0.9972, 0.0028, 0.0000, 0.0000])
    #sample_beliefs[1] = pt.tensor([0.9965, 0.0035, 0.0000, 0.0000])
    #sample_beliefs[2] = pt.tensor([0.8714, 0.1286, 0.0000, 0.0000])
    #sample_beliefs[3] = pt.tensor([0.8160, 0.1840, 0.0000, 0.0000])
    #sample_beliefs[4] = pt.tensor([0.0031, 0.6803, 0.3165, 0.0001])
    #sample_beliefs[5] = pt.tensor([0.0001, 0.0390, 0.9457, 0.0152])
    #sample_beliefs[6] = pt.tensor([0.0000, 0.0003, 0.8488, 0.1509])
    #sample_beliefs[7] = pt.tensor([0.0000, 0.0000, 0.0000, 1.0000])
    #sample_beliefs[8] = pt.tensor([1.0000, 0.0000, 0.0000, 0.0000])
    
    #B = pt.cat((B, sample_beliefs))
    #B = sample_beliefs
    return B,C,w,f

# This returns the calculation for alpha(s | a, tau_n, o), \eref{eq:POSMDP:discreteStatediscreteAction:beliefState:POSMDP2SMDP:backup3}
def computeAlphaATauO(P,G,V,f):
    # Input:
    # P[a,s,s] or [numActions, numStates, numStates]
    # G[a,s,o] or [numActions, numStates, numObs]
    # V[i,s] or [numVec, numState]
    # f[tau,a,s,s'] -- f(tau_n | s, a, s')
    
    # Output:
    # alphaATauO = G * P * f * V
    # alphaATauO [i,f,a,o,s] or [numInV, numInf, numActions, numObs, numStates]
    
    #Temporary format [i, f, a, s, s', o]
    myG = G.unsqueeze(dim=0).unsqueeze(dim=1).unsqueeze(dim=3) #[a,s',o] -> [1,1,a,1,s',o]
    myP = P.unsqueeze(dim=0).unsqueeze(dim=1).unsqueeze(dim=5) #[a,s,s'] -> [1,1,a,s,s',1]
    myf = f.unsqueeze(dim=0).unsqueeze(dim=5) #[f,a,s,s'] -> [1,f,a,s,s',1]
    myV = V.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=5) #[i,s'] -> [i,1,1,1,s',1]
    
    # Essentially, we are doing: G * P * f * V
    
    # This has myG(1,1,a,1,s',o).*myP(1,1,a,s,s',1).*myF(1,f,a,s,s',1).*myV(i,1,1,1,s',1)
    myM =  myG * myP * myf * myV #[i, f, a, s, s', o]
    
    #Summing over all s', thus alphaAO (i,c,a,s,o) and then transpose to get (i,c,a,o,s)
    alphaATauO = myM.sum(dim=4).transpose(3,4)
    return alphaATauO #[i,f,a,o,s] or [numInV, numInf, numActions, numObs, numStates]

# Backing up to an alpha vector. \eref{eq:POSMDP:discreteStatediscreteAction:beliefState:POSMDP2SMDP:backup}
def backup(V,P,G,R,C,w,f,D,xi,flag):
    # Input:
    # V[i,s] or [numVec, numState]
    # P[a,s,s] or [numActions, numStates, numStates]
    # G[a,s,o] or [numActions, numStates, numObs]
    # R[a, s] or [numActions, numStates]
    # C[tau] (the sampled sojourn times)
    # w[a,s,s'] or [numActions, numStates, numStates] (the proportion that (s,a,s') was sampled)
    # f[tau,a,s,s'] -- f(tau_n | s, a, s')
    # D is the discount factor that is precalculated for every time.
    # xi[1,s] or [1, numState] -- a single belief used for the update
    
    # Output:
    # alpha[numStates]
    
    numStates = V.size(1)  #This will get the number of states from the size of the V set.
    
    # \eref{eq:POSMDP:discreteStatediscreteAction:beliefState:POSMDP2SMDP:backup3}
    alphaATauO = computeAlphaATauO(P,G,V,f)  #[i,f,a,o,s] alpha(s | a, tau, o)
    
    # Computing the argmax using beliefs for all s,a,tau,o
    myXi = xi.unsqueeze(0).unsqueeze(0).unsqueeze(0) #[1,s] -> [1,1,1,1,s]
    XiDotAlphaATauO = (alphaATauO * myXi).sum(dim=4) #[i,f,a,o,s] -> [i,f,a,o]  basically xi \cdot alpha, for every alpha in V
                                         
    # Doing the argmax in \eref{eq:POSMDP:discreteStatediscreteAction:beliefState:POSMDP2SMDP:backup3}
    bestalphaind_forallATauO = XiDotAlphaATauO.argmax(dim=0,keepdim=True).unsqueeze(4).repeat(1,1,1,1,numStates)
    # unsqueeze(4) expands the dimension for s
    # repeat [1,1,1,1,numStates] because [i,f,a,o]
    
    bestalphareduce_forallATauO = pt.gather(alphaATauO, index=bestalphaind_forallATauO, dim=0).squeeze(0)
    # Now, we have those best alpha vectors for each time, action, and observation, thus [f,a,o,s].
    # The singleton dimension along alpha vectors is to be removed.
       
    # Computing \eref{eq:POSMDP:discreteStatediscreteAction:beliefState:POSMDP2SMDP:backup2}
    bestalpha_forTau = bestalphareduce_forallATauO.sum(dim=2) #[f,a,o,s] -> [f,a,s]
    myD = D.unsqueeze(1).unsqueeze(1) #[f] -> [f,1,1]
    alphaAXi = R + pt.sum(myD * bestalpha_forTau, dim=0) #[f,a,s] -> [a,s]
    
    # Computing backup \eref{eq:POSMDP:discreteStatediscreteAction:beliefState:POSMDP2SMDP:backup}
    alphaAction = pt.argmax((alphaAXi* xi).sum(dim=1)) #[]
    alpha = alphaAXi[alphaAction].unsqueeze(0) #[alphaAction, :] -> [1,s]
    
    # Why is the action not selected?
    # alphaAXi = R + pt.sum(myD * bestalpha_forTau, dim=0) 
    # You will want to print the alpha and alphaAction because
    # check 
    
    if flag:
        print("alpha", alpha)
        print("alphaAction", alphaAction)
        print("xi", xi)
        print("alphaAXi*xi", alphaAXi*xi) #see why we are not getting action 0?
        print("sum of alphaAXi*xi", (alphaAXi* xi).sum(dim=1))
    
    # Assumption: Zhang could be wrong. we need to break down the backup value
    # calculation.
       
        print("R*xi (immediate)", pt.sum(R*xi, dim=1))
        print("exp future reward", pt.sum(pt.sum(myD * bestalpha_forTau, dim=0)*xi, dim=1))
    
    # These are the two components that are used for determining which action
    # is to be selected. We should see why action 1 is better from this by seeing
    # that the value for action 1 is better than the value for action 0.
        
    return alpha, alphaAction #alpha[numStates]   alphaAction is scalar with 0 to numActions

# Calculates the optimal value function at a particular belief state (\eref{eq:POMDP:V:optimal:xi,alpha}),
# and it's corresponding optimal action (\eref{eq:POMDP:pi:optimal:xi,alpha}).
def valuefunc(V,Vactions,B):
    # Input:
    # V[i,s] or [numVec, numState]
    # Vactions[i] or [numVec]
    # B[b,s] or [numBelief, numState]
    
    # Output:
    # VxB.values, VxBcorrespondingActions   Return the max V*xi [value, action]  
        
    # B * V' = [b,i] -> max [b]
    # b * V' = [1,i] -> max [1]
    
    myV = V.unsqueeze(0) #[i,s] -> [1,i,s]
    myB = B.unsqueeze(1) #[b,s] -> [b,1,s]
    VxB = myV * myB #[b,i,s]
    VxB = pt.sum(VxB, dim=2) #[b,i,s] -> [b,i].  <V,xi> for every xi in B.
    VxB = VxB.max(dim=1)
    VxBcorrespondingActions = Vactions[VxB.indices]
    return VxB.values, VxBcorrespondingActions #Return the max V*xi [value, action]


# Update function
def update(V,Vactions,B,P,G,R,C,w,f,D,device):
    # Input:
    # V is the number of vectors i by number of states s   V[i,s]
    # Vactions
    # B
    
    # Output:
    
    numStates = V.size(1)
    numBeliefs = B.size(0)
    V2 = pt.zeros_like(B) #[b,s] There cannot be more alpha vectors than beliefs.
    V2actions = pt.zeros(numBeliefs, dtype=pt.int16, device=device)    #corresponding action
    B2ind = pt.ones(numBeliefs, device=device)         #1 if belief is not improved; otherwise, 0 if belief is improved.
    
    (Vb, VbActions) = valuefunc(V,Vactions,B)
    k = pt.tensor(0, dtype=pt.int16, device=device)
    
    alphaATauO = computeAlphaATauO(P,G,V,f) #[i,f,a,o,s] alpha(s | a, tau_n, o)
    
    while pt.sum(B2ind) > 0:
        xiInd = pt.multinomial(B2ind, 1)
        xi = B[xiInd]  #Randomly select a belief xi from B2.
                
        #Creates the new alpha vector candidate that could be added to the set V.
        (alpha, alphaAction) = backup(V,P,G,R,C,w,f,D,xi,xiInd==(B.size(0)-9))
        
        XiAlpha = (xi*alpha).sum(dim=1) #[1,s]*[1,s] -> [s]
        
        #No improvement
        if XiAlpha < Vb[xiInd]:
            #Keep the old one
            alpha = Vb[xiInd]
            alphaAction = VbActions[xiInd]
            B2ind[xiInd] = 0   #Enforce the belief removal (numerical rounding instability)

        #Improvement
        VAlphaB = (B*alpha).sum(dim=1)  #[b] compute alpha value for all beliefs
        B2ind = B2ind * (VAlphaB < Vb)  #1 if belief is not improved; otherwise, 0 if belief is improved.  
        V2[k] = alpha
        V2actions[k] = alphaAction
        k = k + 1
    V = V2[0:k]
    Vactions = V2actions[0:k]
    return V, Vactions

#MAIN SCRIPT
def solvePOSMDP(problem, numBeliefs, numIter):
    """
    Solve a given POMDP assuming pytorch tensors on CUDA device (or CPU)

    :param problem: Dictionnnary containing 
        'P' : Tensor of dimensions (a,s,s') of transition probabilities
        'G' : Tensor of dimensions (a,s',o) of observation probabilities
        'r1' : Tensor of dimensions (a,s) of lump sum rewards
        'r2' : Tensor of dimensions (a,s,s') of continuous reward rate
        'mu' : Tensor of dimensions (a,s,s') of mean parameter for sojourn time distribution
        'Lambda' : Tensor of dimensions (a,s,s') of shape parameter for sojourn time distribution
        'xi0' : Tensor of dimensions (1,s) of initial belief
        'beta' : Scalar discount rate

    :param numBeliefs: Number of beliefs to use
    :param numIter: Number of iteration of value iteration
    
    :return: V, VActions
       'V' : Tensor of dimensions (alpha_vector, state) (all the alpha vectors)
       'VActions' : Tensor of dimensions (alpha_vector) (corresponding actions)
    """
    
    # Extract problem
    P = problem["P"]
    G = problem["G"]
    R = problem["R"]
    V = problem["V0"]
    Vactions = problem["V0actions"]
    sojournTime = problem["sojournTime"]  
    xi0 = problem["xi0"]
    beta = problem["beta"]

    #Extract other information
    device = P.device
    numActions, numStates, numObs = G.size()   
     
    #MAIN SCRIPT
    (B, C, w, f) = collectBeliefs(P, G, xi0, sojournTime, numStates, numActions, numBeliefs, device)

    #Precomputing the discount factor for each tau.
    D = (1/C.size(dim=0))*pt.exp(-beta*C)/((w.unsqueeze(dim=0)*f).sum(dim=1).sum(dim=1).sum(dim=1))
    
    minValue = pt.zeros(numIter)
    maxValue = pt.zeros(numIter)

    for j in pt.arange(1,numIter):
        print("Iteration", j)
        (V, Vactions) = update(V,Vactions,B,P,G,R,C,w,f,D,device)
        minValue[j] = V.min()
        maxValue[j] = V.max()
        # print(j, V.size())
    minValue = minValue[1:]
    maxValue = maxValue[1:]
    return B, C, V, Vactions, minValue, maxValue