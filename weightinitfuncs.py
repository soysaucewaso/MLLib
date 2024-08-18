import math

import numpy as np

# weight init functions

# templates
def normal(mean,variance,inputSize, outputSize):
    return np.random.normal(mean,variance,(inputSize,outputSize))

def uniform(low,high,inputSize,outputSize):
    return np.random.uniform(low,high,(inputSize,outputSize))


# real init funcs
def henormal(inputSize,outputSize):
    dev = math.sqrt(2/inputSize)
    return normal(0,dev,inputSize,outputSize)

def xavieruni(inputSize,outputSize):
    num = 6
    range = math.sqrt(6/(inputSize+outputSize))
    return uniform(-range,range,inputSize,outputSize)