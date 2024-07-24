import unittest
import collections
collections.Callable = collections.abc.Callable
import main
import numpy as np
import numpy.testing as tst

class TestMain(unittest.TestCase):
    def test_sigmoid(self):
        X = np.array([[1,2],[4,-1]])
        res = main.sigmoid(X)
        expected = np.array([[0.7310586,0.8807971],[0.9820138,0.2689414]])
        print(res)
        tst.assert_almost_equal(res,expected)
    def test_cross_entropy_loss(self):
        A = np.array([[0.5002307,  0.49985831, 0.50023963]])
        Y = np.array([[1, 0, 0]])
        cost = main.crossEntropyCost(A,Y)
        expected = 0.6930587610394646
        self.assertAlmostEqual(cost,expected)

if __name__ == '__main':
    unittest.main()