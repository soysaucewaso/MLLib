import numpy as np
def bincrossEntropyCost(predicted: [[float]], actual: [[float]]):
    predicted = np.array(predicted)
    actual = np.array(actual)
    # loss = - yln(a) - (1-y)(ln(1-a))
    # cost function for binary classification
    # cost = loss / m
    m = predicted.shape[0]
    try:
        loss = - (actual * np.log(predicted)) - ((1 - actual) * np.log(1 - predicted))
        cost = np.sum(loss, keepdims=True) / (m)
    except:
        loss = None
        cost = np.sum(loss, keepdims=True) / (m)

    # grads
    grads = predicted - actual
    denom = predicted * (1 - predicted)
    grads = grads / denom
    return cost, grads

def crossEntropyCost(predicted: [[float]], actual: [[float]]):
    predicted = np.array(predicted)
    actual = np.array(actual)
    ln = np.log(predicted)
    cost = - np.sum((actual * ln),axis=1,keepdims=True)
    c = np.sum(cost)
    # (a - y) / (a*(1-a)), only for softmax
    # otherwise y / a
    grads = predicted - actual
    denom = predicted * (1 - predicted)
    grads = grads / denom
    return cost, grads

def meanSquareError(predicted: [[float]], actual: [[float]]):
    predicted = np.array(predicted)
    actual = np.array(actual)

    error = np.square(actual - predicted)
    cost = np.sum(error,axis=1,keepdims=True)
    grads = 2 * (predicted - actual)
    return cost, grads