# defines how different layer types shape weights, forward propagate, backwards propagate, and adjust weights
import main
import numpy as np
from joblib import Parallel, delayed
class dense(main.layer):
    def __init__(self, amtNeurons: int, activation=main.layer.LRELU, prev: 'layer' = None, normal=main.layer.ZMNORMAL,
                 weightinit=None,
                 channels=(1, 1)):
        super().__init__(activation, prev, normal, weightinit, channels, )
        self.amtNeurons = amtNeurons
        self.outputSize = amtNeurons
        if self.prev is not None:
            self.initWeights(prev.outputSize, amtNeurons)

    def weightShape(self, iSize):
        return (iSize, self.amtNeurons), (1, self.amtNeurons)

    def forwardTransform(self, input):
        reshapedInput = np.reshape(input, (np.shape(input)[0], -1))
        assert np.shape(reshapedInput)[1] == np.shape(self.weights)[0]

        # linear transformation
        transformed = np.dot(reshapedInput, self.weights) + self.biases
        self.cache = (None, reshapedInput, None)
        return transformed

    def backwardTransform(self, dZ, normalgrad):
        ngradshape = np.shape(normalgrad)
        if len(ngradshape) > 0:
            normalgrad = np.reshape(normalgrad, (ngradshape[0], -1))
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


class conv(main.layer):
    M1 = 'm1'
    M2 = 'm2'
    def __init__(self, inputD, kernel=(3, 3), stride=(1, 1), activation=main.layer.LRELU, prev: 'layer' = None,
                 normal=main.layer.ZMNORMAL, weightinit=None, channels=(1, 1), padb=False):
        assert (len(inputD) == len(kernel) and len(inputD) == len(stride))
        assert (len(channels) == 2)
        super().__init__(activation, prev, normal, weightinit, channels)
        self.kernel = kernel
        self.stride = stride
        # in a conv layer input and output size are the same
        self.inputD = inputD
        self.inputSize = np.prod(inputD,keepdims=False)
        self.padb = padb
        if self.padb:
            self.outputSize = self.inputSize
        else:
            self.outputSize = 1
            for i in range(len(inputD)):
                # output1 = input - kernel + 1
                # output1 = ((output1 - 1) // stride) + 1
                self.outputSize *= (inputD[i] - kernel[i]) // self.stride[i] + 1
        self.inputSize *= channels[0]
        self.outputSize *= channels[1]
        self.initWeights(self.inputSize, self.outputSize)

        # object pointers used for multiprocessing
        self.pointers = {}

    def weightShape(self, iSize):
        ktup = tuple(k for k in self.kernel)
        return ktup + (self.channels[0], self.channels[1]), (self.channels[1],)


    def convolve(self, m1, m2, m1shape, m2shape, stride):
        def worker(indices):
            m2index = tuple(indices)
            m1index = (slice(None),) + tuple(
                slice(m2index[i], m2index[i] + oshape[i] * stride[i], stride[i]) for i in range(len(m2index)))
            return np.dot(m1[m1index], m2[m2index])
        # m1 is BxD1xDNxC
        # m2 is D1xDNxCxF
        # convolve m1 with m2 as a sliding window
        assert (len(m1shape) == len(m2shape))

        #m1str = f'm1{self.__hash__()}.pkl'
        #dump(m1,m1str)
        #shared_m1 = load(m1str)
        for i in range(len(m1shape)):
            assert (m1shape[i] >= m2shape[i])

        assert (len(stride) == len(m1shape))

        oshape = self.getoshape(m1shape, m2shape, self.stride)

        results = Parallel(n_jobs=-1)(delayed(worker)(indices) for indices in self.nnestedgenerator(m2shape))
        result = np.sum(results,axis=0)

        return result

    def forwardTransform(self, inputData):
        # dot a Batch size x height x width x channel input with a ci x co x kh x kw kernel
        # o(i,j) = sum(x(i+r,j+c)*k(r,c))
        if self.padb:
            inputData = self.padInput(inputData)
        ishape = np.shape(inputData)[1:-1]
        # cache padded data
        self.cache = (None, inputData, None)
        Z = self.convolve(inputData, self.weights, ishape, self.kernel, self.stride) + self.biases
        return Z

    def padInput(self, inputData):
        padding = ((0, 0),) + tuple((k // 2, (k - 1) // 2) for k in self.kernel) + ((0, 0),)

        return np.pad(inputData, pad_width=padding, mode='constant', constant_values=0)

    def backwardTransform(self, dZ, normalgrad):
        # pad normalgrad
        if self.padb:
            normalgrad = self.padInput(normalgrad)
        nx = self.cache[1]

        dB = np.sum(dZ, axis=0)
        while len(np.shape(dB)) > 1:
            dB = np.sum(dB, axis=0)
        nxshape = np.shape(nx)[1:-1]
        dZshape = np.shape(dZ)[1:-1]

        # dW is K1 x K2
        dWshape = np.shape(self.weights)
        dW = np.zeros((dWshape))
        dWindices = self.nnestedforloopinit(self.kernel)
        einsumstr = ''.join([chr(97 + i) for i in range(len(dZshape))])

        # dX = dy0*w0,dy1*w0 + dy0*w1,
        # dX/w0 = dy0*w0,dy1*w0,dyN*w0,0,0
        # y0 = n(x0)w0 + x1w1
        # y1 = x2w0
        # y11 = x22w0
        # dX11/w0 = n'(x0)w0
        dX = 0

        # pad dZ with 0 rows and cols for stride
        fillDzShape = ((np.shape(dZ)[0],) +
                       tuple(((dZshape[i] - 1) * self.stride[i] + 1) for i in range(len(dZshape)))
                       + (np.shape(dZ)[-1],))

        while True:
            Windex = tuple(dWindices)
            Xindex = (slice(None),) + tuple(slice(Windex[i], Windex[i] + dZshape[i]
                                                  * self.stride[i], self.stride[i]) for i in range(len(Windex)))
            # x is batch size, y is channels, z is filters
            dW[Windex] = np.einsum(f'x{einsumstr}y,x{einsumstr}z->yz', dZ, nx[Xindex])

            filldZ = np.zeros(fillDzShape)
            fillIndex = ((slice(None),) + tuple(
                slice(0, dZshape[i] * self.stride[i], self.stride[i]) for i in range(len(dZshape))))
            filldZ[fillIndex] = dZ
            padtuple = ((0, 0),) + tuple(
                (dWindices[i], nxshape[i] - fillDzShape[i + 1] - dWindices[i]) for i in range(len(dWindices))) + (
                       (0, 0),)
            paddZ = np.pad(filldZ, pad_width=padtuple, mode='constant', constant_values=0)
            # dot with weights
            WT = np.transpose(self.weights[Windex])
            dX += np.dot(paddZ, WT) * normalgrad
            if self.nnestedforloopincrement(dWindices, dWshape):
                break
        return dW, dB, dX

    def nnestedforloopinit(self, shape):
        indices = [0 for _ in range(len(shape))]
        return indices

    def nnestedforloopincrement(self, indices, shape):
        for i in range(len(indices) - 1, -1, -1):
            indices[i] += 1
            if indices[i] < shape[i]:
                return False
            indices[i] = 0
            if i == 0:
                return True

    def nnestedgenerator(self, shape):
        indices = self.nnestedforloopinit(shape)
        yield tuple(indices)
        while not self.nnestedforloopincrement(indices, shape):
            yield tuple(indices)

    def getoshape(self, s1, s2, stride):
        # output shape after convolving s1 with s2
        oshape = []
        for i in range(len(s1)):
            oshape.append(((s1[i] - s2[i]) // stride[i]) + 1)
        return oshape
