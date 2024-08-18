import numpy as np
def sigmoid(values: [[float]]) -> np.ndarray:
    # logistic function
    # 1 / (1+e^x)
    values = np.clip(values, -300, 300)
    expValues = np.exp(-values)
    return 1 / (1 + expValues)


def siggrad(values: [[float]]):
    s = sigmoid(values)
    return s * (1 - s)

def lrelu(values: [[float]]) -> np.ndarray:
    return np.where(values >= 0,values, values * .01)

def lrelugrad(values: [[float]]) -> np.ndarray:
    return np.where(values >= 0, 1, .01)
def softmax(values: [[float]]):
    D = np.max(values,axis=1).reshape(-1,1)
    # n by neurons
    expValues = np.exp(values - D)
    sum = np.sum(expValues, axis=1).reshape(-1, 1)
    soft = expValues / sum
    return soft

def softgrad(values: [[float]]):
    # n by neurons
    # only consider diagonal for simplicity, and because
    # cross entropy cost eliminates other terms

    # s(1-s)
    s = softmax(values)
    return s * (1-s)
def tanh(values):
    return np.tanh(values)


def tanhgrad(values):
    return 1 - tanh(values) ** 2