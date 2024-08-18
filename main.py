import math

import numpy as np

LRELUW = .01
EP = 10**-5
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
def simplenormal(input):
    return input / np.max(input)

def minmaxnormal(input):
    mi = np.min(input)
    ma = np.max(input)
    d = ma - mi
    return np.where(d == 0,1,(input - mi) / d)

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

def zeromeannormgrad(input):
    g = 1 / np.sqrt(variance(input)+EP)
    return g
class layer:
    """A layer of a ML model

    Has a list of neurons each with a list of weights and a bias.
    Each neuron transforms an input vector x with the equation:
    y = wx + b
    """
    WEIGHT_INIT_SCALAR = .1
    BIAS_INIT = 0.0
    SIGMOID = [sigmoid, siggrad]
    TANH = [tanh, tanhgrad]
    SOFTMAX = [softmax, softgrad]
    # leaky relu
    LRELU = [lrelu, lrelugrad]
    LINEAR = [lambda x: x, lambda _: 1]
    ZMNORMAL = [zeromeannormal, zeromeannormgrad]
    def __init__(self, amtNeurons: int, activation, prev: 'layer' = None, normal = None):
        """
        :param amtNeurons: size of the layer
        :param activation: activation function used to transform output
        :param prev: previous layer
        weights: each row is a list of weights for each neuron,
        shape: (amtNeurons,prev.amtNeurons)
        biases: one bias for each neurons, shape: (amtNeurons,1)
        activation: a function which maps a [[float]] to another [[float]] of the same size
        """
        self.amtNeurons = amtNeurons
        self.activation = activation

        self.prev: 'layer' = prev
        self.next: 'layer' = None
        self.weights, self.biases = None, None
        self.cache: [[float]] = None

        self.normal = normal

        if prev is not None:
            self.initWeights(prev.amtNeurons)

    def append(self, amtNeurons: int, activation, normal = ZMNORMAL):
        nextLayer = layer(amtNeurons, activation, prev=self)
        nextLayer.initWeights(self.amtNeurons)
        if self.next is not None:
            self.next.prev = nextLayer
            self.next.initWeights(nextLayer.amtNeurons)
            nextLayer.next = self.next
        nextLayer.prev = self
        self.next = nextLayer

        nextLayer.normal = normal
        return nextLayer

    def initWeights(self, inputSize: int):
        if inputSize == 0:
            return
        # randomly init to prevent neurons all getting the same weights
        self.weights = np.random.random([inputSize, self.amtNeurons]) * layer.WEIGHT_INIT_SCALAR
        self.biases = np.full((1, self.amtNeurons),self.BIAS_INIT)

    @staticmethod
    def multiple(amtNeurons: [int], activation: str, prev: 'layer' = None,normalL: list = None,normal1 = None,normalA =ZMNORMAL):
        if len(amtNeurons) == 0:
            return
        i = 0
        if normalL is not None:
            assert(len(normalL) == len(amtNeurons))
            normal1 = normalL[0]
        if prev is None:
            rootL = layer(amtNeurons[i], activation,normal=normal1)
        else:
            rootL = prev.append(amtNeurons[i], activation,normal=normalL[0] if normalL is not None else normalA)
        curr = rootL
        i+=1
        while i < len(amtNeurons):
            curr = rootL.append(amtNeurons[i], activation,normal=normalL[i] if normalL is not None else normalA)
            i += 1
        return rootL, curr


class model:
    BINCROSS = bincrossEntropyCost
    CROSS = crossEntropyCost
    MSR = meanSquareError
    def __init__(self, rootLayer: layer, cost, learningRate: float = .1):
        """
        :param rootLayer: inputLayer
        :param num_iterations: how many times the model trains
        :param learning_rate: how much the models weights change with each iteration
        :param cost: string representing loss function
        initialize the model without setting any vars yet
        cost is a function mapping 2 matrixes A and Y, to an float cost which
        decreases as A and Y's difference shrinks
        """
        self.rootLayer: layer = rootLayer
        self.tailLayer: layer = self.rootLayer
        while self.tailLayer.next:
            self.tailLayer = self.tailLayer.next
        self.learningRate: float = learningRate
        self.cost = cost if cost is not None else bincrossEntropyCost

    def train(self, inputData: [[float]], outputData: [[float]], numIterations: int = 100,normal=minmaxnormal):
        inputData = np.array(inputData)
        outputData = np.array(outputData)
        ex = outputData.shape[0]
        if self.tailLayer.activation == self.tailLayer.SOFTMAX and np.shape(outputData)[1] == 1:
            n = self.tailLayer.amtNeurons
            o = np.zeros((ex,n))
            for i in range(ex):
                o[i, outputData[i][0]] = 1
            outputData = o
        # init
        if self.rootLayer.weights is None:
            self.rootLayer.initWeights(np.shape(inputData)[1])
        assert (np.shape(self.tailLayer.weights)[1] == np.shape(outputData)[1])
        """
        :param inputData: training inputs; shape: (feature size, # of inputs
        :param outputData: training outpus; shape: (label size, # of inputs)
        """
        c = []
        for i in range(numIterations):
            predictions = self.forwardPropagate(self.rootLayer, inputData, True,normal)
            if math.isnan(predictions[0][0]):
                return False
            cost, grads = self.computeCost(predictions, outputData)
            c.append(np.sum(cost))
            self.backwardPropagate(self.tailLayer, grads)
        return c

    def predict(self, inputData: [[float]],normal=minmaxnormal):
        if self.rootLayer.weights is None:
            return
        assert (np.shape(np.shape(inputData)[1] == self.rootLayer.weights)[0])
        predictions = self.forwardPropagate(self.rootLayer, inputData, False,normal)
        return predictions

    def forwardPropagate(self, currLayer: layer, inputData: [[float]], cacheData: bool, normal):
        assert (currLayer)
        assert (np.shape(np.shape(inputData)[1] == currLayer.weights)[0])
        """
        propagates through a layer, by calculating the input to the next layer
        :return: predicted values for y, the final layer
        """
        if currLayer.normal is None:
            normalized = inputData
        else:
            normalized = currLayer.normal[0](inputData)
        # linear transformation
        transformed = np.dot(normalized, currLayer.weights) + currLayer.biases
        outputData = currLayer.activation[0](transformed)
        if cacheData:
            currLayer.cache = (inputData, normalized, outputData)
        if currLayer.next is None:
            return outputData
        else:
            return self.forwardPropagate(currLayer.next, outputData, cacheData, normal)

    def computeCost(self, prediction, outputData):
        """
        calculates
        :param prediction:
        :param outputData:
        :return:
        """
        cost, grads = self.cost(prediction, outputData)
        return cost, grads

    def backwardPropagate(self, currLayer: layer, grads: [[float]]):
        assert (currLayer)
        x = currLayer.cache[0]
        nx = currLayer.cache[1]
        z = currLayer.cache[2]
        actgrads = currLayer.activation[1](z)
        dZ = grads * actgrads
        # Z = WN(X) + B
        # dZ/dW = N(X)
        dW = np.dot(dZ.T, nx).T
        # dZ/dB = 1
        dB = np.sum(dZ, axis=0).reshape((1, -1))

        currLayer.cache = None
        if currLayer.prev is None:
            self.adjustWeights(currLayer, dW, dB)
            return

        # dZ/dN(X) = WT
        # dN(X)/dX = 1x2

        #XW = 1x3
        #dX = 3x2
        #dL/dX = 1x2
        #dZ/dX =

        # dX: Z * B
        WT = np.transpose(currLayer.weights)
        if currLayer.normal is not None:
            # normalgrad is 1 x Feature Size
            # broadcast
            normalgrad = currLayer.normal[1](x)
            WT = WT * normalgrad
        dX = np.dot(dZ, WT)

        self.adjustWeights(currLayer, dW, dB)
        self.backwardPropagate(currLayer.prev, dX)

    def adjustWeights(self, currLayer: layer, wGrads: [[float]], bGrads: [[float]]):
        assert (np.shape(wGrads) == np.shape(currLayer.weights))
        assert (np.shape(bGrads) == np.shape(currLayer.biases))
        currLayer.weights -= wGrads * self.learningRate
        currLayer.biases -= bGrads * self.learningRate
