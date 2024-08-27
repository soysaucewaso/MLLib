import math

import numpy as np

WEIGHT_INIT_SCALAR = .1

# weight init functions

# templates
def normal(mean,variance,shape):
    return np.random.normal(mean,variance,shape)

def uniform(low,high,shape):
    return np.random.uniform(low,high,shape)


# real init funcs
def henormal(inputSize,outputSize, shape):
    dev = math.sqrt(2/inputSize)
    return normal(0,dev,shape)

def xavieruni(inputSize,outputSize, shape):
    num = 6
    range = math.sqrt(6/(inputSize+outputSize))
    return uniform(-range,range,shape)