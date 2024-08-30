import math
import random
import unittest
import collections

import layers
import normalfuncs

collections.Callable = collections.abc.Callable
import main
import numpy as np
import numpy.testing as tst
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds



class TestMain(unittest.TestCase):

    def test_zero_mean_normal(self):
        print(1)
    def test_sigmoid(self):
        X = np.array([[1,2],[4,-1]])
        res = main.sigmoid(X)
        expected = np.array([[0.7310586,0.8807971],[0.9820138,0.2689414]])
        tst.assert_almost_equal(res,expected)

    def test_relu(self):
        X = np.array([[1, 2], [4, -1]])
        res = main.lrelu(X)
        g = main.lrelugrad(X)
        print((res,g))

    def test_softmax(self):
        X = np.array([[1,2],[4.0,-1]])
        res = main.softmax(X)
        grad = main.softgrad(X)
        X[1][0]+=.001

        res2 = main.softmax(X)
        print((res2 - res) / .001)

        print(grad)

    def test_entropy_loss(self):
        X = np.array([[1, 2, 3]])
        sm = main.softmax(X)
        c, g = main.crossEntropyCost(sm,[.5, .25, .25])
        grads = g * main.softgrad(X)
        print((c,grads,sm))
    def test_bin_cross_entropy_loss(self):
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



    def testWithRandomData(self):
        l = main.layer(4, main.layer.TANH)
        l.append(2,main.layer.SOFTMAX)
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
        inputdata, outputdata = to_ndarray(beans[0])
        inputdata = inputdata / 255.0
        outputdata = [[min(e[0],1)] for e in outputdata]
        testidata, testodata = self.to_ndarray(beans[1])
        p1 = m.predict(testidata)
        m.train(inputdata,outputdata,2)

        p2 = m.predict(testidata)
        for i in range(20):
            print(f"P1: {p1[i]}, P2: {p2[i]}, Actual: {testodata[i]}")

    def test_dataset2(self):

        l = main.layer(40, main.layer.LRELU)
        h = l.append(40,main.layer.TANH)
        o = h.append(1,main.layer.LINEAR,normal=None )
        m = main.model(l, main.model.MSR,learningRate=.005,optimizer=main.model.ADAM)
        ds, info = tfds.load("diamonds", with_info=True, as_supervised=True, split=['train'])
        inputdata, outputdata = to_ndarray(ds[0])
        inputdata = [[v for _, v in l[0].items()] for l in inputdata]

        tsize = len(inputdata) - 20

        inputdata = inputdata / np.max(inputdata,axis=0,keepdims=True)
        maxoutput = np.max(outputdata,axis=0,keepdims=True)
        testidata, testodata = inputdata[tsize:], outputdata[tsize:]
        outputdata = outputdata / maxoutput
        inputdata, outputdata = inputdata[:tsize], outputdata[:tsize]
        p1 = m.predict(testidata)
        c = m.train(inputdata, outputdata, 100)
        print(c)
        p2 = m.predict(testidata)
        p2 *= maxoutput
        p3 = []
        #for i in range(len(p2)):
        #    p3.append([f"{el:.2%}" for el in p2[i]])
        for i in range(len(p2)):
            print(f"P2: {p2[i]}, Actual: {testodata[i]}")

    def test_simple(self):
        l = main.layer(1,main.layer.TANH)

        m = l.append(1,main.layer.TANH)
        m.append(2,main.layer.SOFTMAX)
        n= 100
        input = [[random.randint(0,1)] for _ in range(n)]
        output = [i for i in input]
        m = main.model(l,main.model.CROSS,learningRate=.01)
        m.train(input,output)
        testi = [[random.randint(0,1)] for _ in range(10)]
        p = m.predict(testi)
        for i, ival in enumerate(testi):
            print(f"Input: {ival} Actual: {p[i]}")

    def test_small(self):
        l = layers.dense(7,main.layer.LRELU)
        m = l.append(layers.dense(7,main.layer.LRELU))
        m.append(layers.dense(2,main.layer.SOFTMAX))
        input = [[0],[1]]
        output = [[0],[1]]
        m = main.model(l,main.model.CROSS   ,learningRate=.3,optimizer=main.model.ADAM)
        m.train(input,output,numIterations=10)
        testi = [[0],[1]]
        p = m.predict(testi)
        for i, ival in enumerate(testi):
            print(f"Input: {ival} Actual: {p[i]}")

    def load_csv(self,path):
        rec = np.recfromcsv(path)
        data = [[p for p in r] for r in rec]
        return data, rec.dtype.names

    def test_fossil_ds(self):
        data, names = self.load_csv('fossils/train_data.csv')
       # test, _ = self.load_csv('fossils/test_data.csv')
        pmap = {b'Normal polarity': 0, b'Reversed polarity' : 1}
        bmap = {False: 0, True: 1}
        rocks = [b'Sandstone',b'Shale',b'Limestone',b'Conglomerate']
        pos = [b'Bottom',b'Middle',b'Top']
        periods = set()
        for d in data:
            periods.add(d[4])
        a = list(periods)
        rmap = {k:i for i, k in enumerate(rocks)}
        posmap = {k:i for i, k in enumerate(pos)}
        geomap = {k:i for i, k in enumerate(a)}
        out = [d[-1] for d in data]
        output = np.reshape(out,(-1,1))
        input = np.array([d[0:4]+[geomap[d[4]]]+[pmap[d[5]]]+[bmap[d[6]]] + [d[7]] + [rmap[d[8]]] + [posmap[d[9]]] + d[10:-1] for d in data])

        testsize = 20
        to = output[:testsize]
        ti = input[:testsize]

        normalparams = {}
        output = normalfuncs.zeromeannormal(output[testsize:],normalparams)
        input = input[testsize:]

        l = main.layer(20,normal=main.layer.ZMNORMAL)
        n = l.append(20)
        r = n.append(20)
        o = r.append(1,main.layer.LINEAR)

        m = main.model(l,main.model.MSR,learningRate=.0001)
        m.train(input,output,numIterations=200)

        pred = m.predict(ti)
        pred = normalfuncs.undozeromean(pred,normalparams)
        s = 0
        for i in range(testsize):
            print(ti[i])
            print(pred[i])
            print(f'Actual: {to[i]}')
            print('')
            s += abs(pred[i] - to[i]) ** 2
        print(s/testsize)
        print(math.sqrt(s)/testsize)

    def test_conv(self):
        l = layers.conv((4,4), kernel=(2,2), stride=(2,1),padb=False,)
        l2 = l.append(layers.dense(1))

        s = l.weightShape(1)[0]
        m = main.model(l,cost=main.model.MSR)
        i = np.reshape([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],[16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]], (2, 4, 4, 1))
        o = np.reshape([[0],[2]],(2,1))
        m.train(i,o)
        print(l.weights)
        print(m.predict(i))


if __name__ == '__main':

    unittest.main()



