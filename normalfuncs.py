import numpy as np
EP = 10**-5

def simplenormal(input):
    return input / np.max(input)

def minmaxnormal(input):
    # range between 0 and 1
    mi = np.min(input)
    ma = np.max(input)
    d = ma - mi
    return np.where(d == 0,1,(input - mi) / d)

def minmaxgrad(input):
    mi = np.min(input)
    ma = np.max(input)
    d = ma - mi
    return np.where(d == 0, 1, 1)
def variance(data,mean=None):
    if mean is None:
        mean = np.average(data,axis=0,keepdims=True)
    error = np.square((data-mean))
    return np.sum(error,axis=0,keepdims=True) / np.shape(data)[0]
def zeromeannormal(input):
    avg = np.average(input,axis=0,keepdims=True)
    var = variance(input,avg)
    # (x - avg) / sqrt(var + ep)
    return (input - avg) / np.sqrt(var+EP)

def zeromeangrad(input):
    g = 1 / np.sqrt(variance(input)+EP)
    return g