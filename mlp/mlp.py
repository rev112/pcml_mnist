#!/usr/bin/env python2

import scipy as s
import functions as func

class Layer:
    """Common parent for layers"""
    # TODO do we need it?
    # options: expose abstract methods forward_step, backward_step etc.


class OutputLayer(Layer):
    """Class for an output layer"""

    def __init__(self, d):
        self.d = d
        self.w = s.ones(d)
        self.b = 1.0
        # For consistency
        self.h = 1

    def forward_step(self, x):
        """Return the value for the last layer (a number, not a vector)"""
        w = self.w
        b = self.b
        d = self.d
        assert len(w) == d, "Invalid size of weight vector (w)"
        assert len(x) == d, "Invalid size of input vector (x)"
        a = w.dot(x) + b
        assert type(a) == s.float64
        return a

    def backward_step(x):
        """Return r error value for this layer"""
        return

    def update(r):
        """Update the parameters for this layer (w,b), given the error"""
        return


class HiddenLayer(Layer):
    """Class for one hidden layer"""

    def __init__(self, neurons_num, d):
        self.h = neurons_num
        self.d = d
        links = 2 * self.h
        self.w = s.matrix(s.zeros( (links, d) ))
        self.b = s.ones(links)

    def forward_step(self, x):
        """Return z_k = g(a_k, a_k+1) values for this layer (as a vector)"""
        w = self.w
        b = self.b
        d = self.d
        assert len(x) == d, "Invalid size of input vector (x)"
        a_q = w.dot(x).A[0] + b
        assert len(a_q) == 2 * self.h, "Invalid size of a_q"

        # Apply transfer function
        # FIXME Is there a better way?
        z = map(lambda (x,y): func.g(x,y), zip(a_q[::2], a_q[1::2]))
        assert len(z) == self.h, "Invalid size of output vector (z)"
        return z

    def backward_step(x):
        """Return r_k error values for this layer"""
        return

    def update(r):
        """Update the parameters for this layer, given the errors"""
        return


class Mlp:
    def __init__(self, hidden_layers_list, d):
        self.d = d
        layers = []
        # Create hidden layers
        for neuron_num in hidden_layers_list:
            hidden_layer = HiddenLayer(neuron_num, d)
            layers.append(hidden_layer)
            d = neuron_num

        # Create output layer
        output_layer= OutputLayer(d = hidden_layers_list[-1])
        layers.append(output_layer)
        self.layers = layers

    def get_layers_num(self):
        return len(self.layers)

    def draw(self):
        print 'input dimension:', self.d
        for l in self.layers[:-1]:
            print 'hidden layer size:', l.h
        output_neurons = self.layers[-1].h
        assert output_neurons == 1, "1 neuron in output layer, no?"
        print 'output layer size:', self.layers[-1].h

    def compute_layers_output(self, x):
        assert len(x) == self.d, "Invalid size of input vector (x)"
        output = x
        for l in self.layers:
            output = l.forward_step(output)
        return output

    def classify(self, x):
        output = compute_layers_output(x)
        output_class = int(s.sign(output))
        # TODO do we need to handle this case?
        assert output_class != 0
        return output_class


if __name__ == "__main__":
    d = 5
    neur_n = 2
    l1 = HiddenLayer(neur_n, d)
    print l1.forward_step([1] * d), "\n"

    l2 = OutputLayer(neur_n)
    print l2.forward_step([1] * neur_n), "\n"

    mlp = Mlp(hidden_layers_list = [1,2], d = 3)
    print "Number of layers, including output layer:", mlp.get_layers_num()
    mlp.draw()
    print mlp.compute_layers_output([2,3,4]), "\n"

