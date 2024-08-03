import random
import unittest
import collections
collections.Callable = collections.abc.Callable
import main
import numpy as np
import numpy.testing as tst
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
class TestMain(unittest.TestCase):
    def test_sigmoid(self):
        X = np.array([[1,2],[4,-1]])
        res = main.sigmoid(X)
        expected = np.array([[0.7310586,0.8807971],[0.9820138,0.2689414]])
        tst.assert_almost_equal(res,expected)
    def test_cross_entropy_loss(self):
        A = np.array([[0.5002307,  0.49985831, 0.50023963]])
        Y = np.array([[1, 0, 0]])
        cost = main.crossEntropyCost(A,Y)
        expected = 0.6930587610394646
        self.assertAlmostEqual(cost,expected)

    def test_forward_propagate(self):
        l = main.layer(3,'sigmoid')
        m = main.model(l,'crossEntropyCost')
        out, _ = m.forwardPropagate(l,[[.5,.5],[-.5,.5]])
        print(out)

    def test_model(self):
        l = main.layer(3, 'sigmoid')
        m = main.model(l, 'crossEntropyCost')
        m.train(np.array([[-1,0],[0,1]]),np.array([[1, 0,1],[1,0,1]]))
        pred = m.predict([[-.5,.5],[30,2]])
        print(pred)

    def to_ndarray(self,dataset):
        inputdata = np.array([np.array(e[0]).flatten() for e in dataset])
        outputdata = np.reshape([np.array(e[1]) for e in dataset],(-1,1))
        return inputdata, outputdata



    def testWithRandomData(self):
        l = main.layer(4, main.layer.TANH)
        l.append(1,main.layer.SIGMOID)
        m = main.model(l, 'crossEntropyCost', learningRate=.01)
        n = 100
        k = 10
        outputdata = [[random.randint(0,1)] for _ in range(n)]
        inputdata = [[random.random(),(r[0]*-2+1),random.random()] for r in outputdata]
        testo = [[random.randint(0,1)] for _ in range(k)]
        testi = [[random.random(),(r[0]*-2+1),random.random()] for r in testo]
        m.train(inputdata,outputdata,numIterations=20)
        p = m.predict(testi)
        for i in range(k):
            print(f"Input: {testi[i]}Predicted: {p[i]},Actual: {testo[i]}")

    def test_with_dataset(self):
        l = main.layer(1, 'sigmoid')
        m = main.model(l,'crossEntropyCost',learningRate=.00001)
        beans, info = tfds.load("beans",with_info=True, as_supervised=True,split=['train','test'])
        inputdata, outputdata = self.to_ndarray(beans[0])
        inputdata = inputdata / 255.0
        outputdata = [[min(e[0],1)] for e in outputdata]
        testidata, testodata = self.to_ndarray(beans[1])
        p1 = m.predict(testidata)
        m.train(inputdata,outputdata,2)

        p2 = m.predict(testidata)
        for i in range(20):
            print(f"P1: {p1[i]}, P2: {p2[i]}, Actual: {testodata[i]}")


    def test_simple(self):
        l = main.layer(1,main.layer.TANH)
        m = l.append(1,main.layer.TANH)
        m.append(1,main.layer.SIGMOID)
        n= 100
        input = [[random.randint(0,1)] for _ in range(n)]
        output = [i for i in input]
        m = main.model(l,'crossEntropyCost',learningRate=1)
        m.train(input,output)
        testi = [[random.randint(0,1)] for _ in range(10)]
        p = m.predict(testi)
        for i, ival in enumerate(testi):
            print(f"Input: {ival} Actual: {p[i]}")

    def test_small(self):
        l = main.layer(2,main.layer.TANH)
        m = l.append(2,main.layer.SIGMOID)

        #o = m.append(5,main.layer.TANH)
        #m.append(1,main.layer.SIGMOID)
        input = [[0],[1]]
        output = [[0,1],[1,1]]
        m = main.model(l,'crossEntropyCost',learningRate=.3)
        m.train(input,output)
        testi = [[0],[1]]
        p = m.predict(testi)
        for i, ival in enumerate(testi):
            print(f"Input: {ival} Actual: {p[i]}")


if __name__ == '__main':
    unittest.main()