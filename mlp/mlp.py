#!/usr/bin/env python2

import sys
import scipy as s
import functions as func
import defaults
import debug as dbg
import pickle

class Layer:
    """Common parent for MLP layers
    
        d - size of input vector (integer)
        w - matrix of weights (scipy.matrix of floats)
        b - bias parameter vector (floats)
        h - number of neurons (integer)
    """

    def __init__(self, neurons_num, d):
        # We need previous delta values for momentum term computation
        self.dw_prev = 0
        self.db_prev = 0

        self.h = neurons_num
        self.d = d
        self.w = None # define it in subclass
        self.b = None # define it in subclass
        return

    def set_default_parameters(self, params={}):
        """Read default parameters from 'defaults' module"""
        s = [
                ('l_rate', defaults.LEARNING_RATE_DEFAULT),
                ('m_term', defaults.MOMENTUM_TERM_DEFAULT),
            ]
        for k, v in s:
            params.setdefault(k, v)

    def forward_step(self, x):
        """Return the output for the layer"""
        raise NotImplementedError

    def backward_step(self, x):
        """Return r error value for the layer"""
        raise NotImplementedError

    def compute_gradient(self, x, r):
        """Computes gradient within the layer

            x - layer input (vector)
            r - layer error (vector, computed in backward_step)
            
            returns (dE_dw, dE_db)
        """
        raise NotImplementedError

    def update(self, x, r, params={}):
        """Update the parameters for this layer (w,b), given the error

            x - layer input (vector)
            r - layer error (vector of floats, computed in backward_step)
        """

        self.set_default_parameters(params)
        dE_dw, dE_db = self.compute_gradient(x,r)

        dE_dw_shape = dE_dw.shape
        w_shape = self.w.shape
        assert dE_dw_shape == w_shape, "Inconsistent shapes of weight matrix and the gradient"

        l_rate = params['l_rate']
        m_term = params['m_term']

        # Using momentum term. Set m_term to 0 to disable it
        dw = -l_rate * (1 - m_term) * dE_dw + m_term * self.dw_prev
        self.w += dw
        self.dw_prev = dw

        # TODO Duplicated code...
        db = -l_rate * (1 - m_term) * dE_db + m_term * self.db_prev
        self.b += db
        self.db_prev = db
        return

    def get_weights_len(self):
        """Returns the length of all weights for this layer"""
        (wrows, wcols) = self.w.shape
        lenb = len(self.b)
        return wrows*wcols + lenb

    def serialize_weights(self):
        """Returns ndarray of all weights for this layer"""
        # this returns 'flat' vector, row after row (and then turn it to array)
        w = s.asarray(self.w).reshape(-1)
        return s.concatenate((w, self.b))

    def deserialize_weights(self, w):
        """Takes array w, reshapes it and stores it as internal weights and
        bias parameters
        Note: this is not a serialization of the whole class, this
        reconstruction uses dimensions already stored in object
        """
        assert(len(w) == self.get_weights_len())

        (wrows, wcols) = self.w.shape
        wtotal = wrows * wcols

        self.w = s.asmatrix(w[:wtotal]).reshape(wrows, wcols)
        self.b = w[wtotal:]
        

class OutputLayer(Layer):
    """Class for an output layer

        d - size of input vector (integer)
        w - vector of weights (actually, a scipy.matrix with one row, all floats)
        b - bias parameter (1 float in array)
        h - number of neurons, always 1 for output layer
    """

    def __init__(self, d):
        Layer.__init__(self, 1, d)

        self.w = s.matrix(s.ones(d))
        self.b = s.array([1.0])

    def forward_step(self, x):
        """Return the value for the last layer (a float, not a vector)"""
        w = func.get_first_row(self.w)
        b = self.b
        d = self.d
        assert len(w) == d, "Invalid size of weight vector (w)"
        assert len(x) == d, "Invalid size of input vector (x)"
        a = w.dot(x) + b
        assert type(a) == s.float64 \
                or (type(a) == type(s.array([])) and len(a) == 1)
        return a

    def backward_step(self, a, t):
        """Return r error value for this output layer (float)

            a - output of the layer (float), equals to a_(k)
            t - actual class (+1 or -1)
        """
        a = s.float64(a)
        t = int(t)
        assert t in [-1, 1], "Invalid class"
        t_new = (1 + t) / 2
        return s.array([func.sig(a) - t_new])


    def compute_gradient(self, x, r):
        """Compute gradient for w and b, return as a tuple

            x - layer input (vector)
            r - layer error (vector of one float, computed in backward_step)
        """

       # Gradient for weight vector
        dE_dw = s.matrix(r * s.array(x))

        # Gradient for bias
        assert len(r) == 1
        dE_db = s.array(r)
        dbg.prt('output gradients', dE_db, dE_dw)
        return (dE_dw, dE_db)

class HiddenLayer(Layer):
    """Class for one hidden layer

        d - size of input vector (integer)
        w - matrix of weights (scipy.matrix of floats)
        b - bias parameter vector (floats)
        h - number of neurons (integer)
    """

    def __init__(self, neurons_num, d):
        Layer.__init__(self, neurons_num, d)

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
        a_q_even = a_q[::2]
        a_q_odd = a_q[1::2]
        z = func.g(a_q_even, a_q_odd)
        assert len(z) == self.h, "Invalid size of output vector (z)"
        return z

    def backward_step(self, x, w, a):
        """Return r_k error values for this layer

          x - error vector from the next layer
          w - weight matrix from the next layer
          a - output of this layer (before applying the transfer function!)
        """
        # See 3.3.1 from the course notes

        # TODO Check it!!!
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

        # 3. Prepare weight matrix (ex: [[1,2], [3,4]] -> [[1,1,2,2], [3,3,4,4]])
        w = func.duplicate_columns(w)
        assert w.shape[1] == 2 * self.h, "Invalid size of extended weight vector"

        # 4a. Compute the product
        r_temp = g_diag.dot(w.transpose())
        r = r_temp.dot(x)
        # 4b. Turn r into a vector
        r = s.array(r.tolist()[0])

        assert len(r) == 2 * self.h, "Invalid size of resulting error vector"
        return r

    def compute_gradient(self, x, r):
        """Compute gradient for w and b, return as a tuple

            x - layer input (vector)
            r - layer error (vector, computed in backward_step)
        """
        # Gradient for weight vector
        assert len(r) == 2 * self.h, "Invalid size of error vector"

        # TODO check it!!!
        dE_dw =  (s.matrix(x).transpose() * r).transpose()

        dE_dw_shape = dE_dw.shape
        assert dE_dw_shape == (2 * self.h, self.d), "Invalid shape of weight matrix"

        # Gradient for bias
        dE_db = r
        assert len(dE_db) == 2 * self.h, "Invalid size of error gradient"
        return (dE_dw, dE_db)

class Mlp:
    """Class that represents the whole network

        d - size of input vector (integer)
        layers - list of Layer object, the last one is OutputLayer object

    Parameters:
        params - parameters dictionary
            mterm - momentum term
            learning_rate - learning rate for Mlp (dynamic)
    """

    def __init__(self, hidden_layers_list, d, test_gradients_flag = False,
                    derivative_tolerance = 1e-8):
        self.d = d
        self.set_parameters()
        layers = []
        # Create hidden layers
        for neuron_num in hidden_layers_list:
            hidden_layer = HiddenLayer(neuron_num, d)
            layers.append(hidden_layer)
            d = neuron_num

        # Create output layer
        output_layer = OutputLayer(d = hidden_layers_list[-1])
        layers.append(output_layer)
        self.layers = layers

        self.test_gradient_flag = test_gradient_flag
        self.derivative_tolerance = derivative_tolerance

    def set_parameters(self,
                       m_term = defaults.MOMENTUM_TERM_DEFAULT,
                       l_rate = defaults.LEARNING_RATE_DEFAULT,
                       is_rate_dynamic = defaults.IS_DYNAMIC_LEARNING_RATE_DEFAULT):
        self.params = {}

        assert 0.0 <= m_term and m_term <= 1.0, "Invalid momentum term"
        self.params['m_term'] = m_term * 1.0

        assert 0.0 <= l_rate, "Invalid learning rate"
        self.params['l_rate'] = l_rate * 1.0

        assert is_rate_dynamic in [True, False], "Waaaaat?"
        self.params['is_rate_dynamic'] = is_rate_dynamic

    def get_layers_num(self):
        """Return the number of layers in the network, including the ouput layer"""
        return len(self.layers)

    def get_weights_dimension(self):
        """Returns the number of all the weights and biases"""
        layer_dims = [layer.get_weights_len() for layer in self.layers]
        return s.sum(layer_dims)

    def serialize_weights(self):
        """Returns ndarray of all weights"""
        layers_weights = [layer.serialize_weights() for layer in self.layers]
        return s.concatenate(layers_weights)

    def deserialize_weights(self, w):
        """Takes array w, reshapes it and stores it as internal weights and
        bias parameters
        Note: this is not a serialization of the whole class, this
        reconstruction uses dimensions already stored in object
        """
        assert len(w) == self.get_weights_dimension()

        layer_dims = [layer.get_weights_len() for layer in self.layers]

        # cumulative sum, to index w
        post_dims = s.cumsum(layer_dims).tolist()
        # lower bound to index w, for easier iteration
        pre_dims = [0] + post_dims[:-1]

        layers_weights = [w[a:b] for a, b in zip(pre_dims, post_dims)]
        assert len(layers_weights) == len(self.layers)

        for layer, w_l in zip(self.layers, layers_weights):
            layer.deserialize_weights(w_l)


    def draw(self):
        """Print the layout of the network"""
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

    def get_point_error(self, x, t):
        # Update error, see 4.2 in miniproject description
        assert len(x) == self.d, "Invalid size of data point (x)"
        a = self.compute_layers_output(x)
        # Compute log(1 + e^(-t*a)) as (-t*a + log(1 + e^(t*a)))
        temp = -t*a
        return temp + s.log1p(s.exp(-temp))

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
            error += self.get_point_error(x,t)

        error = error / n
        return error

    def classify(self, x):
        """Classify the input as +1 or -1"""
        output = self.compute_layers_output(x)
        output_class = int(s.sign(output))
        # TODO do we need to handle this case? Not sure right now
        assert output_class != 0, "Impossibru!"
        return output_class

    def update_network(self, x, t):
        """Update parameters of the network for one point; performs one forward
        and one backward pass
        @param x one datapoint - a list or ndarray of length d
        @param t integer denoting datapoint's class (-1 or 1)
        """
        assert len(x) == self.d, "Invalid size of input vector (x)"
        pass_info = []
        l_input = s.array(x)

        ### Forward step

        # Hidden layers
        for l in self.layers[:-1]:
            l_temp = l.layer_output(l_input)   # Layer output without transfer function
            l_output = l.forward_step(l_input) #   ...        with transfer function
            layer_info = {'input': l_input, 'temp': l_temp, 'output': l_output}
            pass_info.append(layer_info)
            l_input = l_output

        # Output layer
        output_layer = self.layers[-1]
        l_output = output_layer.forward_step(l_input) # the output !
        layer_info = {'input': l_input, 'output': l_output}
        pass_info.append(layer_info)

        # pdb.set_trace()
        dbg.prt("Pass info:", pass_info)

        ### Backward step

        # Update output layer
        out_layer = self.layers[-1]
        out_layer_input = pass_info[-1]['input']
        network_output = pass_info[-1]['output']
        out_error = out_layer.backward_step(network_output, t) # backward step
        next_err = out_error
        next_w = out_layer.w
        out_layer.update(out_layer_input, out_error, self.params)

        # Update hidden layers
        hidden_layers = self.layers[:-1]
        pass_info_hidden = pass_info[:-1]
        assert len(hidden_layers) == len(pass_info_hidden), "Inconsistent number of layers"
        hidden_layer_num = len(hidden_layers)

        for i in xrange(hidden_layer_num - 1, -1, -1):
            layer = hidden_layers[i]
            layer_info = pass_info[i]
            dbg.prt('layer', i, ':', layer, layer_info)
            layer_weights = layer.w
            layer_error = layer.backward_step(next_err, next_w, layer_info['temp'])
            layer.update(layer_info['input'], layer_error, self.params)
            next_w = layer_weights
            next_err = layer_error

        self.test_gradient(x, t)

    def test_gradient(self, x, t):
        """A debugging method that performs gradient testing. It will do
        nothing if self.test_gradient_flag is off
        """
        if not self.test_gradient_flag:
            return

        total_dimension = self.get_weights_dimension()
        whole_gradient = s.zeros(total_dimension)

        # fill the gradient
        start_index = 0
        for layer in self.layers:
            length = layer.get_weights_len()
            end_index = start_index + length

            (dE_dw, dE_db) = layer.compute_gradient() 
            dE_dw = s.asarrya(dE_dw).reshape(-1)

            whole_dradient[start_index:end_index] = s.concatenate((dE_dw, dE_db))

            start_index += length


        # test the gradient: first with 30 random 'directions'
        nb_random_directions = 30
        too_large_difference = False
        for k in xrange(nb_random_directions):
            direction = func.get_random_direction(total_dimension)

            derivative_approx = func.computeDirectionalDerivative(x, t, direction, 
                    self.derivative_tolerance)
            derivative_true = whole_gradient.dot(direction)

            if abs(derivative_true - derivative_approx) > 10*self.derivative_tolerance:
                too_large_difference = True
                break

        # if some wrong gradient found, test gradient for every weight separately
        if too_large_difference:
            print >> sys.stderr, "[WARNING]: Gradient of error function might be wrong", \
                    "\n\tTesting for all directions"
            direction = s.zeros(total_dimension)
            for i in xrange(total_dimension):
                direction[i] = 1.0
                if i >= 0:
                    direction[i-1] = 0.0

                derivative_approx = func.computeDirectionalDerivative(x, t, direction, 
                        self.derivative_tolerance)
                derivative_true = whole_gradient.dot(direction)

                difference = abs(derivative_true - derivative_approx)

                if difference > 10*self.derivative_tolerance:
                    # TODO: maybe output something nicer
                    print >> sys.stderr, "\tweight index: %d;\tdifference: %f" % (i, difference)
                

if __name__ == "__main__":
    neur_n = 2

    l2 = OutputLayer(neur_n)
    linput = [1] * neur_n
    f_step2 = l2.forward_step(linput)
    error = l2.backward_step(f_step2, 1)
    print l2.w, l2.b, "\n"
    l2.update(linput, error)
    print l2.w, l2.b, "\n"

    d = 5
    l1 = HiddenLayer(neur_n, d)
    linput = [1] * d
    f_step1 = l1.layer_output(linput)
    errors = l1.backward_step(error, l2.w, f_step1)
    print errors
    l1.update(linput, errors)
    print l1.w

    mlp = Mlp(hidden_layers_list = [4,2], d = 2)
    print "Number of layers, including output layer:", mlp.get_layers_num()
    mlp.draw()
    for i in xrange(40):
        print "\nRun", i
        mlp.update_network([1,1], 1)
        mlp.update_network([-1,-1], -1)

    # just to test the computeGradientApproximation code
    lx = [[0,0], [1,1]]
    lt = [1, 0]
    print "numerical gradient:", func.computeGradientApproximation(mlp, lx, lt)

    pickle.dump(mlp, open('trained_network.dat', 'wb'))


