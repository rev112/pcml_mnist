#!/usr/bin/env python2

import scipy as s
import functions as func

class Layer:
    """Common parent for MLP layers"""

    def forward_step(self, x):
        """Return the output for the layer"""
        raise NotImplementedError

    def backward_step(self, x):
        """Return r error value for the layer"""
        raise NotImplementedError

    def update(self):
        """Update layer parameters"""
        raise NotImplementedError


class OutputLayer(Layer):
    """Class for an output layer

        d - size of input vector (integer)
        w - vector of weights (scipy.ndarray of floats)
        b - bias parameter (float)
        h - number of neurons, always 1 for output layer
    """

    def __init__(self, d):
        self.d = d
        self.w = s.ones(d)
        self.b = 1.0
        self.h = 1

    def forward_step(self, x):
        """Return the value for the last layer (a float, not a vector)"""
        w = self.w
        b = self.b
        d = self.d
        assert len(w) == d, "Invalid size of weight vector (w)"
        assert len(x) == d, "Invalid size of input vector (x)"
        a = w.dot(x) + b
        assert type(a) == s.float64
        return a

    def backward_step(self, x, t):
        """Return r error value for this output layer (float)

            x - output of the layer (float), equals to a_(k)
            t - actual class (+1 or -1)
        """
        x = s.float64(x)
        t = int(t)
        assert t in [-1, 1], "Invalid class"
        t_new = (1 + t) / 2
        return func.sig(x) - t_new

    def update(r):
        """Update the parameters for this layer (w,b), given the error"""
        return


class HiddenLayer(Layer):
    """Class for one hidden layer

        d - size of input vector (integer)
        w - matrix of weights (scipy.matrix of floats)
        b - bias parameter (float)
        h - number of neurons (integer)
    """

    def __init__(self, neurons_num, d):
        self.h = neurons_num
        self.d = d
        links = 2 * self.h
        self.w = s.matrix(s.zeros( (links, d) ))
        self.b = s.ones(links)

    def layer_output(self, x):
        """Return the layer output before applying the transfer function"""
        w = self.w
        b = self.b
        d = self.d
        assert len(x) == d, "Invalid size of input vector (x)"
        a_q = w.dot(x).A[0] + b
        assert len(a_q) == 2 * self.h, "Invalid size of a_q"
        return a_q

    def forward_step(self, x):
        """Return z_k = g(a_k, a_k+1) values for this layer (as a vector)"""

        a_q = self.layer_output(x)

        # Apply transfer function
        # FIXME Is there a better way?
        #z = map(lambda (x,y): func.g(x,y), zip(a_q[::2], a_q[1::2]))

        # possible fix (not nice)
        a_q_even = a_q[::2]
        a_q_odd = a_q[1::2]
        z = func.g(a_q_even, a_q_odd)
        assert len(z) == self.h, "Invalid size of output vector (z)"
        return z

    def backward_step(self, x, w, a):
        """Return r_k error values for this layer

          x - error vector from the next layer
          w - weight vector from the next layer
          a - output of this layer (before applying the transfer function!)
        """
        # See 3.3.1 from the course notes

        # 1. Create vector of g'_x and g'_y
        # (g_der_1, g_der_2)
        pairs = zip(a[::2], a[1::2])
        g_vector = map(lambda p: (func.g_der_1(p[0], p[1]),
                                  func.g_der_2(p[0], p[1])),
                       pairs)
        g_vector = s.array(g_vector).flatten()
        assert len(g_vector) == 2 * self.h, "Invalid size of g_vector"

        # 2. Create a diagonal matrix of g'
        g_diag = s.diag(g_vector)
        print g_diag

        # 3. Prepare weight vector (ex: [3,5] -> [3,3,5,5])
        w = map(lambda x: [x, x], w)
        w = s.array(w).flatten()
        assert len(w) == 2 * self.h, "Invalid size of extended weight vector"
        print w

        # 4. Compute the product
        r_temp = g_diag.dot(w.transpose())
        r = r_temp.dot(x)
        assert len(r) == 2 * self.h, "Invalid size of resulting error vector"
        print r

        return

    def update(r):
        """Update the parameters for this layer, given the errors"""
        return


class Mlp:
    """Class that represents the whole network

        d - size of input vector (integer)
        layers - list of Layer object, thhe last one is OutputLayer object
    """

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
        """Return the number of layers in the network, including the ouput layer"""
        return len(self.layers)

    def draw(self):
        """Prints the layout of the network"""
        print 'input dimension:', self.d
        for l in self.layers[:-1]:
            print 'hidden layer size:', l.h
        output_neurons = self.layers[-1].h
        assert output_neurons == 1, "1 neuron in output layer, no?"
        print 'output layer size:', self.layers[-1].h

    def compute_layers_output(self, x):
        """Return the output of the whole network (a_Last, float)"""
        assert len(x) == self.d, "Invalid size of input vector (x)"
        output = x
        for l in self.layers:
            output = l.forward_step(output)
        return output

    def get_input_error(self, lx, lt):
        """Return the value of error function for the whole dataset"""
        assert len(lx) == len(lt), "Data vector and class vector have different dimensions"
        error = 0
        # Size of dataset
        n = len(lx)
        # FIXME for loop used, they will punish us
        # ANSWER: we can't fix this, since compute_layers_output() can't accept
        # multiple inputs at once. It is not meant to work that way. Anyway, I
        # think that we don't even need this function.. we're going to use
        # stochastic gradient descent which always takes only one data per time.
        for i in xrange(n):
            x = lx[i]
            t = lt[i]
            assert len(x) == self.d, "Invalid size of data point (x)"

            # Network output
            a = self.compute_layers_output(x)

            # Update error, see 4.2 in miniproject description
            temp = -t*a
            error += temp + s.log(1 + s.exp(-temp))

        error = error / n
        return error

    def classify(self, x):
        """Classify the input as +1 or -1"""
        output = self.compute_layers_output(x)
        output_class = int(s.sign(output))
        # TODO do we need to handle this case? Not sure right now
        assert output_class != 0, "Impossibru!"
        return output_class


if __name__ == "__main__":
    d = 5
    neur_n = 2

    l2 = OutputLayer(neur_n)
    f_step2 = l2.forward_step([1] * neur_n)
    error = l2.backward_step(f_step2, 1)
    print f_step2, error, "\n"

    l1 = HiddenLayer(neur_n, d)
    f_step1 = l1.layer_output([1] * d)
    errors = l1.backward_step(error, l2.w, f_step1)



    mlp = Mlp(hidden_layers_list = [1,2], d = 3)
    print "Number of layers, including output layer:", mlp.get_layers_num()
    mlp.draw()
    print mlp.compute_layers_output([2,3,4]), "\n"
    print mlp.get_input_error([[2,3,4], [4,5,6]], [1,-1])
    print mlp.classify([2,3,4])

