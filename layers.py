# defines how different layer types shape weights, forward propagate, backwards propagate, and adjust weights
import main
import numpy as np
class dense(main.layer):
    def __init__(self, amtNeurons: int, activation=main.layer.LRELU, prev: 'layer' = None, normal=None, weightinit=None,
                 channels=(1, 1)):
        super().__init__(activation, prev, normal, weightinit, channels)
        self.amtNeurons = amtNeurons
        self.outputSize = amtNeurons
        if self.prev is not None:
            self.initWeights(prev.outputSize,amtNeurons)

    def weightShape(self, iSize):
        return (iSize, self.amtNeurons), (1, self.amtNeurons)

    def forwardTransform(self, input):
        assert (np.shape(np.shape(input)[1] == self.weights)[0])
        # linear transformation
        transformed = np.dot(input, self.weights) + self.biases
        return transformed

    def backwardTransform(self, dZ, normalgrad):
        # dZ is the derivative of the loss function with respect to z
        # normalgrad is the derivative of x with respect to n(x)
        nx = self.cache[1]

        # Z = WN(X) + B
        # dZ/dW = N(X)
        dW = np.dot(dZ.T, nx).T
        # dZ/dB = 1
        dB = np.sum(dZ, axis=0).reshape((1, -1))

        if self.prev is None:
            return dW, dB, None

        # dZ/dN(X) = WT
        # dN(X)/dX = 1x2

        # XW = 1x3
        # dX = 3x2
        # dL/dX = 1x2
        # dZ/dX =

        # dX: Z * B
        WT = np.transpose(self.weights)
        WT = WT * normalgrad
        dX = np.dot(dZ, WT)
        return dW, dB, dX

    def appendDense(self, amtNeurons: int, activation=main.layer.LRELU, normal=main.layer.ZMNORMAL):
        nextLayer = dense(amtNeurons, activation, prev=self)
        if self.next is not None:
            self.next.prev = nextLayer
            self.next.initWeights(nextLayer.amtNeurons)
            nextLayer.next = self.next
        nextLayer.prev = self
        self.next = nextLayer
        #nextLayer.channels = (self.channels[1],ochannels)
        nextLayer.normal = normal
        return nextLayer

class conv(main.layer):

    def __init__(self,inputD, kernel = (3, 3), activation = main.layer.LRELU, prev: 'layer' = None, normal = None, weightinit = None,channels = (1, 1),padb=False):
        assert(len(inputD) == len(kernel))
        assert(len(channels) == 2)
        super().__init__(activation, prev, normal, weightinit, channels)
        self.kernel = kernel
        # in a conv layer input and output size are the same
        self.inputD = inputD
        self.inputSize = np.prod(inputD)
        if type(self.inputSize) == list:
            self.inputSize = self.inputSize[0]
        self.padb = padb
        if self.padb:
            self.outputSize = self.inputSize
        else:
            self.outputSize = 1
            for i in range(len(inputD)):
                self.outputSize *= inputD[i] - kernel[i]
        self.initWeights(self.inputSize, self.outputSize)
    def weightShape(self, iSize):
        ktup = tuple(k for k in self.kernel)
        return ktup + (self.channels[0],self.channels[1]), (self.channels[1],)


    def forwardTransform(self, inputData):
        # dot a Batch size x height x width x channel input with a ci x co x kh x kw kernel
        if self.padb:
            inputData = self.padInput(inputData)
        ishape = np.shape(inputData)[1:-1]
        oshape = []
        for i in range(len(ishape)):
            oshape.append(ishape[i]-self.kernel[i]+1)
        # reshape to 1 x 1 x 1 x 1 x co
        result = np.reshape(self.biases,(1,) + tuple(1 for _ in oshape) + (np.shape(self.biases)[-1],))
        # multiply each loc in kernel with submatrix
        # support any dimension of kernel
        kernelindices = [0 for _ in range(len(self.kernel))]
        traversed = False
        while not traversed:
            windex = tuple(kernelindices)
            iindex = (slice(None), ) + tuple(slice(windex[i],windex[i]+oshape[i]) for i in range(len(windex)))
            # broadcast kernel at one r,c
            slab = np.dot(inputData[iindex], self.weights[windex])
            result = result + slab
            for i in range(len(kernelindices)-1,-1,-1):
                kernelindices[i] += 1
                if kernelindices[i] < self.kernel[i]:
                    break
                kernelindices[i] = 0
                if i == 0:
                    traversed = True
        return result


    def padInput(self, inputData):
        padding = self.kernel[0], self.kernel[1]
        padding = (0,0), (padding[0] // 2, (padding[0]-1) // 2), (padding[1] // 2, (padding[1]-1) // 2), (0,0)


        return np.pad(inputData,pad_width=padding,mode='constant',constant_values=0)


    def backwardTransform(self, dZ, normalgrad):
        pass

class pooling(main.layer):







