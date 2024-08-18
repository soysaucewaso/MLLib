import numpy as np

import universalconsts

MOMLAST = 'momentumlast'
MOMBETA = .9

RMSPLAST = 'rmsplast'
RMSPBETA = .999
TSTR = 't'

EP = universalconsts.EP
def momentuminit(graddict,rootlayer):
    mom = []
    while rootlayer is not None:
        mom.append([None,None])
        mom[-1][0] = np.zeros(np.shape(rootlayer.weights))
        mom[-1][1] = np.zeros(np.shape(rootlayer.biases))
        rootlayer = rootlayer.next
    graddict[MOMLAST] = mom


def momentum(graddict, grads, layer: int, weight: bool):
    i = 0 if weight else 1
    # m_t = m_t-1 * beta + (1-beta) * grads
    mlast = graddict[MOMLAST][layer][i]
    beta = MOMBETA
    t = graddict[TSTR]
    m = mlast * beta + (1-beta) * grads
    # m is a matrix with same dimensions as grads

    graddict[MOMLAST][layer][i] = m

    # m is biased towards 0
    denom = 1 - beta ** t
    mhat = m / denom
    return mhat

def rmspinit(graddict, rootlayer):
    # Root mean square propagation
    acc = []
    while rootlayer is not None:
        acc.append([None,None])
        acc[-1][0] = np.zeros(np.shape(rootlayer.weights))
        acc[-1][1] = np.zeros(np.shape(rootlayer.biases))
        rootlayer = rootlayer.next
    graddict[RMSPLAST] = acc


def rmsphelper(graddict, grads, layer: int, weight: bool):
    i = 0 if weight else 1
    # v_t = v_t-1 * beta + (1-beta) * grads^2
    vlast = graddict[RMSPLAST][layer][i]
    beta = RMSPBETA
    t = graddict[TSTR]
    v = vlast * beta + (1 - beta) * np.square(grads)
    # v is a matrix with same dimensions as grads

    graddict[RMSPLAST][layer][i] = v

    # m is biased towards 0
    denom = 1 - beta ** t
    vhat = v / denom
    return vhat

def rmsp(graddict, grads, layer: int, weight: bool):

    vhat = rmsphelper(graddict,grads,layer,weight)
    denom = np.sqrt(vhat+EP)
    return grads / denom

def adaminit(graddict, rootlayer):
    momentuminit(graddict,rootlayer)
    rmspinit(graddict,rootlayer)
def adam(graddict, grads, layer: int, weight: bool):
    mhat = momentum(graddict,grads,layer,weight)
    vhat = rmsphelper(graddict,grads,layer,weight)
    denom = np.sqrt(vhat+EP)
    return mhat/denom


