import numpy as np
def sigmoid(values: [[int]]):
    # logistic function
    # 1 / (1+e^x)
    expValues = np.exp(-values)
    return 1 / (1+expValues)

def crossEntropyCost(predicted: [[int]], actual : [[int]]):
    #loss = yln(a) + (1-y)(1-ln(a))
    # cost function for binary classification
    # cost = - loss / m
    loss = (actual * np.log(predicted)) + ((1-actual) * np.log(1-predicted))
    m = predicted.shape[1]
    cost = np.sum(loss,keepdims=True) / (-m)
    return cost
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
        self.prev = prev
        self.next = None

        prevAmt = 0 if prev is None else prev.amtNeurons
        self.weights, self.biases = self.initWeights(prevAmt, amtNeurons, activation)
    def __add__(self, amtNeurons: int, activation: str):
        nextLayer = layer(amtNeurons,activation,prev=self)
        self.next = nextLayer
    @staticmethod
    def initWeights(prevAmt: int, amtNeurons: int, activation: str):
        if prevAmt == 0:
            return None, None
        # randomly init to prevent neurons all getting the same weights
        weights = np.random.randn(amtNeurons,prevAmt) * layer.WEIGHT_INIT_SCALAR
        biases = np.zeros((amtNeurons,1))

class model:
    def __init__(self, rootLayer, cost: str, numIterations: int = 100, learningRate: int = 3):
        """
        :param rootLayer: inputLayer
        :param num_iterations: how many times the model trains
        :param learning_rate: how much the models weights change with each iteration
        :param cost: string representing loss function
        initialize the model without setting any vars yet
        cost is a function mapping 2 matrixes A and Y, to an int cost which
        decreases as A and Y's difference shrinks
        """
        self.rootLayer = rootLayer
        self.numIterations = numIterations
        self.learningRate = learningRate
        self.cost = crossEntropyCost

    def train(self, inputData: [[int]], outputData: [[int]]):
        """
        :param inputData: training inputs; shape: (feature size, # of inputs
        :param outputData: training outpus; shape: (label size, # of inputs)
        """
        for i in range(self.numIterations):
            predictions, tailLayer = self.forwardPropagate(self.rootLayer, inputData)
            cost = self.computeCost(predictions,outputData)
            self.backwardPropagate(tailLayer, cost)

    def forwardPropagate(self,currLayer: layer, inputData: [[int]]):
        """
        propagates through a layer, by calculating the input to the next layer
        :return: predicted values for y, the final layer
        """
        # linear transformation
        transformed = np.dot(currLayer.weights, inputData)

        outputData = currLayer.activation(transformed)
        # cache some stuff

        if currLayer.next is None:
            return outputData, currLayer
        else:
            return self.forwardPropagate(currLayer.next, outputData)

    def computeCost(self, prediction, outputData):
        """
        calculates
        :param prediction:
        :param outputData:
        :return:
        """
        cost = self.cost(prediction,outputData)
        return cost

    def backwardPropagate(self):
        return None