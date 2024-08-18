import math

import numpy as np
import activationfuncs as a
import costfuncs
import normalfuncs
import weightinitfuncs
import optimizefuncs

LRELUW = .01


#
class layer:
    """A layer of a ML model

    Has a list of neurons each with a list of weights and a bias.
    Each neuron transforms an input vector x with the equation:
    y = wx + b
    """
    WEIGHT_INIT_SCALAR = .1
    BIAS_INIT = 0.0
    SIGMOID = [a.sigmoid, a.siggrad]
    TANH = [a.tanh, a.tanhgrad]
    SOFTMAX = [a.softmax, a.softgrad]
    # leaky relu
    LRELU = [a.lrelu, a.lrelugrad]
    LINEAR = [lambda x: x, lambda _: 1]

    # normalization
    MINMAX = [normalfuncs.minmaxnormal, normalfuncs.minmaxgrad]
    ZMNORMAL = [normalfuncs.zeromeannormal, normalfuncs.zeromeangrad]

    # Weight Init Functions
    HENORM = (weightinitfuncs.henormal,)
    XAVUNI = (weightinitfuncs.xavieruni,)

    def __init__(self, amtNeurons: int, activation, prev: 'layer' = None, normal=None, weightinit=None):
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
        if weightinit is not None:
            self.weightinit = weightinit
        elif activation == self.SIGMOID or activation == self.TANH or activation == self.SOFTMAX:
            self.weightinit = self.XAVUNI
        else:
            self.weightinit = self.HENORM

        self.prev: 'layer' = prev
        self.next: 'layer' = None
        self.weights, self.biases = None, None
        self.cache: [[float]] = None

        self.normal = normal
        if prev is not None:
            self.initWeights(prev.amtNeurons)

    def append(self, amtNeurons: int, activation, normal=ZMNORMAL):
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
        self.weights = self.weightinit[0](inputSize, self.amtNeurons)
        self.biases = np.full((1, self.amtNeurons), self.BIAS_INIT)

    @staticmethod
    def multiple(amtNeurons: [int], activation: str, prev: 'layer' = None, normalL: list = None, normal1=None,
                 normalA=ZMNORMAL):
        if len(amtNeurons) == 0:
            return
        i = 0
        if normalL is not None:
            assert (len(normalL) == len(amtNeurons))
            normal1 = normalL[0]
        if prev is None:
            rootL = layer(amtNeurons[i], activation, normal=normal1)
        else:
            rootL = prev.append(amtNeurons[i], activation, normal=normalL[0] if normalL is not None else normalA)
        curr = rootL
        i += 1
        while i < len(amtNeurons):
            curr = rootL.append(amtNeurons[i], activation, normal=normalL[i] if normalL is not None else normalA)
            i += 1
        return rootL, curr


class model:
    # cost functions
    BINCROSS = costfuncs.bincrossEntropyCost
    CROSS = costfuncs.crossEntropyCost
    MSR = costfuncs.meanSquareError

    # optimizers
    MOMENTUM = [optimizefuncs.momentuminit, optimizefuncs.momentum]

    def __init__(self, rootLayer: layer, cost, learningRate: float = .1, optimizer=MOMENTUM):
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
        self.numLayers = 0 if rootLayer is None else 1
        while self.tailLayer.next:
            self.tailLayer = self.tailLayer.next
            self.numLayers += 1
        self.learningRate: float = learningRate
        self.cost = cost if cost is not None else self.MSR

        self.optimizer = optimizer
        self.graddict = {'t': 1}
        self.optimizer[0](self.graddict,self.rootLayer)

    def train(self, inputData: [[float]], outputData: [[float]], numIterations: int = 100):
        inputData = np.array(inputData)
        outputData = np.array(outputData)
        ex = outputData.shape[0]
        if self.tailLayer.activation == self.tailLayer.SOFTMAX and np.shape(outputData)[1] == 1:
            n = self.tailLayer.amtNeurons
            o = np.zeros((ex, n))
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
            predictions = self.forwardPropagate(self.rootLayer, inputData, True)
            if math.isnan(predictions[0][0]):
                return False
            cost, grads = self.computeCost(predictions, outputData)
            c.append(np.sum(cost))
            self.backwardPropagate(self.tailLayer, grads,self.numLayers - 1)
            self.graddict['t'] += 1
        return c

    def predict(self, inputData: [[float]]):
        if self.rootLayer.weights is None:
            return
        assert (np.shape(np.shape(inputData)[1] == self.rootLayer.weights)[0])
        predictions = self.forwardPropagate(self.rootLayer, inputData, False)
        return predictions

    def forwardPropagate(self, currLayer: layer, inputData: [[float]], cacheData: bool):
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
            return self.forwardPropagate(currLayer.next, outputData, cacheData)

    def computeCost(self, prediction, outputData):
        """
        calculates
        :param prediction:
        :param outputData:
        :return:
        """
        cost, grads = self.cost(prediction, outputData)
        return cost, grads

    def backwardPropagate(self, currLayer: layer, grads: [[float]], i):
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
            self.adjustWeights(currLayer, dW, dB, i)
            return

        # dZ/dN(X) = WT
        # dN(X)/dX = 1x2

        # XW = 1x3
        # dX = 3x2
        # dL/dX = 1x2
        # dZ/dX =

        # dX: Z * B
        WT = np.transpose(currLayer.weights)
        if currLayer.normal is not None:
            # normalgrad is 1 x Feature Size
            # broadcast
            normalgrad = currLayer.normal[1](x)
            WT = WT * normalgrad
        dX = np.dot(dZ, WT)

        self.adjustWeights(currLayer, dW, dB, i)
        self.backwardPropagate(currLayer.prev, dX, i-1)

    def adjustWeights(self, currLayer: layer, wGrads: [[float]], bGrads: [[float]], i):
        assert (np.shape(wGrads) == np.shape(currLayer.weights))
        assert (np.shape(bGrads) == np.shape(currLayer.biases))
        w = self.optimizer[1](self.graddict, wGrads, i, True)
        b = self.optimizer[1](self.graddict, bGrads, i, False)
        currLayer.weights -= w * self.learningRate
        currLayer.biases -= b * self.learningRate
