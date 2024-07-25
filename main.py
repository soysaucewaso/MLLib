import numpy as np
def sigmoid(values: [[int]]):
    # logistic function
    # 1 / (1+e^x)
    expValues = np.exp(-values)
    return 1 / (1+expValues)

def siggrad(values: [[int]]):
    s = sigmoid(values)
    return s * (1-s)
def crossEntropyCost(predicted: [[int]], actual : [[int]]):
    #loss = yln(a) + (1-y)(1-ln(a))
    # cost function for binary classification
    # cost = - loss / m
    loss = (actual * np.log(predicted)) + ((1-actual) * np.log(1-predicted))
    m = predicted.shape[1]
    cost = np.sum(loss,keepdims=True) / (-m)
    # grads
    grads = predicted - actual
    denom = predicted * (1-predicted)
    grads = grads / denom
    return cost, grads


class layer:
    """A layer of a ML model

    Has a list of neurons each with a list of weights and a bias.
    Each neuron transforms an input vector x with the equation:
    y = wx + b
    """
    WEIGHT_INIT_SCALAR = .01
    def __init__(self, amtNeurons: int, activation: str,prev:'layer'=None):
        """
        :param amtNeurons: size of the layer
        :param activation: activation function used to transform output
        :param prev: previous layer
        weights: each row is a list of weights for each neuron,
        shape: (amtNeurons,prev.amtNeurons)
        biases: one bias for each neurons, shape: (amtNeurons,1)
        activation: a function which maps a [[int]] to another [[int]] of the same size
        """
        self.amtNeurons = amtNeurons
        self.activation = sigmoid
        self.actgrad = siggrad
        self.prev: 'layer' = prev
        self.next: 'layer' = None
        self.weights, self.biases = None, None
        self.cache: [[int]] = None

        if prev is not None:
            self.initWeights(prev.amtNeurons)
    def append(self, amtNeurons: int, activation: str):
        nextLayer = layer(amtNeurons,activation,prev=self)
        nextLayer.initWeights(amtNeurons)
        if self.next is not None:
            self.next.prev = nextLayer
            self.next.initWeights(nextLayer.amtNeurons)
            nextLayer.next = self.next

        nextLayer.prev = self
        self.next = nextLayer
        return nextLayer


    def initWeights(self, inputSize: int):
        if inputSize == 0:
            return
        # randomly init to prevent neurons all getting the same weights
        self.weights = np.random.randn(self.amtNeurons,inputSize) * layer.WEIGHT_INIT_SCALAR
        self.biases = np.zeros((self.amtNeurons,1))

    @staticmethod
    def multiple(amtNeurons: [int],activation:str,prev: 'layer' = None):
        if len(amtNeurons) == 0:
            return
        i = 0
        if prev is None:
            rootL = layer(amtNeurons[i],activation)
        else:
            rootL = prev.append(amtNeurons[i],activation)
        curr = rootL
        while i < len(amtNeurons):
            curr = rootL.append(amtNeurons[i],activation)
            i += 1
        return rootL, curr



class model:
    def __init__(self, rootLayer: layer, cost: str, numIterations: int = 100, learningRate: float = .1):
        """
        :param rootLayer: inputLayer
        :param num_iterations: how many times the model trains
        :param learning_rate: how much the models weights change with each iteration
        :param cost: string representing loss function
        initialize the model without setting any vars yet
        cost is a function mapping 2 matrixes A and Y, to an int cost which
        decreases as A and Y's difference shrinks
        """
        self.rootLayer: layer = rootLayer
        self.tailLayer: layer = self.rootLayer
        while self.tailLayer.next:
            self.tailLayer = self.tailLayer.next
        self.numIterations: int= numIterations
        self.learningRate: int = learningRate
        self.cost = crossEntropyCost


    def train(self, inputData: [[int]], outputData: [[int]]):

        #init
        self.rootLayer.initWeights(np.shape(inputData)[1])
        assert(np.shape(self.tailLayer.weights)[0] == np.shape(outputData)[0])
        """
        :param inputData: training inputs; shape: (feature size, # of inputs
        :param outputData: training outpus; shape: (label size, # of inputs)
        """
        for i in range(self.numIterations):
            predictions = self.forwardPropagate(self.rootLayer, inputData, True)
            cost, grads = self.computeCost(predictions,outputData)
            self.backwardPropagate(self.tailLayer, grads)

    def predict(self,inputData: [[int]]):
        assert(np.shape(self.rootLayer.weights)[1] == np.shape(inputData)[0])
        predictions = self.forwardPropagate(self.rootLayer,inputData,False)
        return predictions

    def forwardPropagate(self,currLayer: layer, inputData: [[int]], cacheData: bool):
        assert(currLayer)
        assert(np.shape(currLayer.weights)[1] == np.shape(inputData)[0])
        """
        propagates through a layer, by calculating the input to the next layer
        :return: predicted values for y, the final layer
        """
        if cacheData:
            currLayer.cache = inputData
        # linear transformation
        transformed = np.dot(currLayer.weights, inputData)

        outputData = currLayer.activation(transformed)
        # cache some stuff

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
        cost, grads = self.cost(prediction,outputData)
        return cost, grads

    def backwardPropagate(self,currLayer: layer, grads: [[int]]):
        assert(currLayer)
        actgrads = currLayer.actgrad(grads)
        dZ = grads * actgrads
        # dZ/dW = XT
        XT = np.transpose(currLayer.cache)
        dW = np.dot(dZ,XT)
        # dZ/dB = 1
        shape = np.shape(dZ)
        dB = np.sum(dZ,axis=1).reshape((shape[0],1))
        # dZ/dX = WT dot dZ
        WT = np.transpose(currLayer.weights)
        dX = np.dot(WT,grads)

        currLayer.cache = None
        self.adjustWeights(currLayer,dW,dB)
        if currLayer.prev is None:
            return
        self.backwardPropagate(currLayer.prev,dX)

    def adjustWeights(self,currLayer: layer, wGrads: [[int]], bGrads: [[int]]):
        assert(np.shape(wGrads) == np.shape(currLayer.weights))
        assert(np.shape(bGrads) == np.shape(currLayer.biases))
        currLayer.weights -= wGrads * self.learningRate
        currLayer.biases -= bGrads * self.learningRate



