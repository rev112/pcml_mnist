import unittest
import functions as fun
from mlp import *
import numpy as np

class FunctionsTestCase(unittest.TestCase):

    def test_random_direction_norm(self):
        for d in xrange(1, 1001, 10):
            dir = fun.get_random_direction(d)
            self.assertAlmostEqual(np.linalg.norm(dir), 1.)


#    def test_random_direction_uniformity(self):
#        """to test this properly, we need a chi-squared test;
#        so better way is to plot and test visually
#        
#        image with 3D random directions is plotted with 
#        fun.testplot_random_directions
#        and results are promising
#        """
#        pass
#        # THIS DOWN IS WRONG
#        d = 5
#        sum = np.zeros(d)
#        for k in xrange(10000):
#            dir = fun.get_random_direction(d)
#            sum += dir
#
#        self.assertLessEqual(np.linalg.norm(sum), 1e-3)


class MLPTestCase(unittest.TestCase):

    def test_total_dimension_getter(self):

        # 1 hidden layer
        hidden_layers = [10]
        input_dim = 5
        mlp = Mlp(hidden_layers, input_dim)

        true_dim_value = 2*(hidden_layers[0]*(input_dim+1)) + hidden_layers[0]+1
        self.assertEqual(mlp.get_weights_dimension(), true_dim_value)

        # 2 hidden layers
        hidden_layers = [40, 20]
        input_dim = 100
        mlp = Mlp(hidden_layers, input_dim)

        true_dim_value = 2*(hidden_layers[0]*(input_dim+1)) \
                + 2*(hidden_layers[1]*(hidden_layers[0]+1)) \
                + hidden_layers[1]+1
        self.assertEqual(mlp.get_weights_dimension(), true_dim_value)


    def test_weight_serialization(self):
        hidden_layers = [40, 20]
        input_dim = 100
        mlp = Mlp(hidden_layers, input_dim)

        x = np.random.normal(size=input_dim)
        y_original = mlp.classify(x)
        
        w = mlp.serialize_weights()
        mlp.deserialize_weights(w)

        y_updated = mlp.classify(x)

        self.assertEqual(y_original, y_updated)


# those suite functions down there are not actually used

def functions_suite():
    return unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)

def mlp_suite():
    return unittest.TestLoader().loadTestsFromTestCase(MLPTestCase)


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run( functions_suite() )
    unittest.TextTestRunner(verbosity=2).run( mlp_suite() )
