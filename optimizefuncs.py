import numpy as np


MOMLAST = 'momentumlast'
MOMBETA = .9
TSTR = 't'
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

