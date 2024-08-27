import abc
import math

import numpy as np
import activationfuncs as a
import costfuncs
import normalfuncs
import weightinitfuncs
import optimizefuncs
from abc import ABC, abstractmethod
LRELUW = .01






class layer(ABC):
    """A layer of a ML model

    Has a list of neurons each with a list of weights and a bias.
    Each neuron transforms an input vector x with the equation:
    y = wx + b
    """

    #Density
    DENSE = 'dense'
    CONV = 'convolutional'
    BIAS_INIT = 0.0
    # activation functions
    ACT = {}
    SIGMOID = (a.sigmoid, a.siggrad)
    TANH = (a.tanh, a.tanhgrad)
    SOFTMAX = (a.softmax, a.softgrad)
    RELU = (lambda x: np.where(x >= 0, x, 0), lambda x: np.where(x >= 0, 1, 0))
    # leaky relu
    LRELU = (a.lrelu, a.lrelugrad)
    LINEAR = (lambda x: x, lambda _: 1)

    # normalization
    MINMAX = (normalfuncs.minmaxnormal, normalfuncs.minmaxgrad)
    ZMNORMAL = (normalfuncs.zeromeannormal, normalfuncs.zeromeangrad)

    # Weight Init Functions
    HENORM = (weightinitfuncs.henormal,)
    XAVUNI = (weightinitfuncs.xavieruni,)
    def __init__(self, activation = LRELU, prev: 'layer' = None, normal=None, weightinit=None, channels = (1,1)):
        """
        :param amtNeurons: size of the layer
        :param activation: activation function used to transform output
        :param prev: previous layer
        weights: each row is a list of weights for each neuron,
        shape: (amtNeurons,prev.amtNeurons)
        biases: one bias for each neurons, shape: (amtNeurons,1)
        activation: a function which maps a [[float]] to another [[float]] of the same size
        """
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
        self.channels: tuple = channels
        self.outputSize = 0
    #def defineDensityParamsDefault(self):




    def initWeights(self, inputSize: int, outputSize):
        if inputSize == 0:
            return
        wshape, bshape = self.weightShape(inputSize)
        self.weights = self.weightinit[0](inputSize, outputSize, wshape)
        self.biases = np.full(bshape, self.BIAS_INIT)

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

    # overriden methods for density
    @abc.abstractmethod
    def weightShape(self, iSize):
        pass


    @abc.abstractmethod
    def forwardTransform(self, inputData):
        pass

    def forwardPropagate(self, inputData, cacheData: bool):
        """
                propagates through a layer, by calculating the input to the next layer
                :return: predicted values for y, the final layer
                """
        if self.normal is None:
            normalized = inputData
        else:
            normalized = self.normal[0](inputData)
        # forward propagate differently depending on density

        transformed = self.forwardTransform(inputData)

        outputData = self.activation[0](transformed)
        if cacheData:
            self.cache = (inputData, normalized, outputData)
        if self.next is None:
            return outputData
        else:
            return self.next.forwardPropagate(outputData, cacheData)

    @abc.abstractmethod
    def backwardTransform(self, dZ, normalgrad):
        pass

    def backwardPropagate(self, grads: [[float]], i, model):
        x = self.cache[0]
        z = self.cache[2]
        if self.normal is not None:
            # normalgrad is 1 x Feature Size
            # broadcast
            normalgrad = self.normal[1](x)
        else: normalgrad = 1

        actgrads = self.activation[1](z)
        dZ = grads * actgrads
        dW, dB, dX = self.backwardTransform(dZ, normalgrad)
        self.cache = None

        self.adjustWeights(dW, dB, i, model)
        if self.prev is not None:
            self.prev.backwardPropagate(dX, i-1, model)


    def adjustWeights(self, dW, dB, i, model):
        model.adjustWeights(self,dW,dB,i)
        pass


class model:
    # cost functions
    BINCROSS = costfuncs.bincrossEntropyCost
    CROSS = costfuncs.crossEntropyCost
    MSR = costfuncs.meanSquareError

    # optimizers
    MOMENTUM = (optimizefuncs.momentuminit, optimizefuncs.momentum)
    RMSP = (optimizefuncs.rmspinit, optimizefuncs.rmsp)
    ADAM = (optimizefuncs.adaminit, optimizefuncs.adam)
    def __init__(self, rootLayer: layer, cost, learningRate: float = .1, optimizer=ADAM):
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
            n = self.tailLayer.outputSize
            o = np.zeros((ex, n))
            for i in range(ex):
                o[i, outputData[i][0]] = 1
            outputData = o
        # init
        if self.rootLayer.weights is None:
            self.rootLayer.initWeights(np.shape(inputData)[1],self.rootLayer.outputSize)
        assert (np.shape(self.tailLayer.weights)[1] == np.shape(outputData)[1])
        """
        :param inputData: training inputs; shape: (feature size, # of inputs
        :param outputData: training outpus; shape: (label size, # of inputs)
        """
        c = []
        for i in range(numIterations):
            predictions = self.rootLayer.forwardPropagate(inputData, True)
            if math.isnan(predictions[0][0]):
                return False
            cost, grads = self.computeCost(predictions, outputData)
            c.append(np.sum(cost))
            self.tailLayer.backwardPropagate(grads,self.numLayers - 1, self)
            self.graddict['t'] += 1
        return c

    def predict(self, inputData: [[float]]):
        if self.rootLayer.weights is None:
            return
        assert (np.shape(np.shape(inputData)[1] == self.rootLayer.weights)[0])
        predictions = self.rootLayer.forwardPropagate(inputData, False)
        return predictions

    def computeCost(self, prediction, outputData):
        """
        calculates
        :param prediction:
        :param outputData:
        :return:
        """
        cost, grads = self.cost(prediction, outputData)
        return cost, grads


    def adjustWeights(self, currLayer: layer, wGrads: [[float]], bGrads: [[float]], i):
        assert (np.shape(wGrads) == np.shape(currLayer.weights))
        assert (np.shape(bGrads) == np.shape(currLayer.biases))
        w = self.optimizer[1](self.graddict, wGrads, i, True)
        b = self.optimizer[1](self.graddict, bGrads, i, False)
        currLayer.weights -= w * self.learningRate
        currLayer.biases -= b * self.learningRate
