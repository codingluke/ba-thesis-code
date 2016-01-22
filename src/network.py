"""network3.py
~~~~~~~~~~~~~~

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

"""

#### Libraries
# Standard library
import cPickle
import gzip
import pdb

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

np.random.seed(45675674)

#### Main class used to construct and train networks
class Network():

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param
                       for layer in self.layers
                       for param in layer.params]
        self.x = T.matrix("x", dtype=theano.config.floatX)
        self.y = T.matrix("y", dtype=theano.config.floatX)
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout,
                           self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        self.meta = {}

    def __getstate__(self):
        return (self.layers, self.mini_batch_size,
                self.x, self.y, self.params, self.meta)

    def __setstate__(self, state):
        layers = mini_batch_size = x = y = params = meta = None
        if len(state) == 5:
          layers, mini_batch_size, x, y, params = state
        else:
          layers, mini_batch_size, x, y, params, meta = state
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.x = x
        self.y = y
        self.params = params
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        if meta: self.meta = meta

    def predict(self, data):
        shared_data = theano.shared(
            np.asarray(data, dtype=theano.config.floatX),
            borrow=True)

        i = T.lscalar() # mini-batch index
        predictions = theano.function(
            [i], self.layers[-1].output,
            givens={
                self.x:
                shared_data[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        num_batches = len(data)/self.mini_batch_size
        return [predictions(j) for j in xrange(num_batches)]

    def save(self, filename='model.pkl'):
        f = open(filename, 'wb')
        cPickle.dump(self, f)

    def SGD(self, training_data=None, epochs=10, mini_batch_size=100,
            eta=0.025, validation_data=None, lmbda=0.0, momentum=None,
            patience=40000, patience_increase=2, improvement_threshold=0.995,
            validation_frequency=1, metric_recorder=None, save_dir=None):
        """Train the network using mini-batch stochastic gradient descent."""

        if not validation_data or len(validation_data) < 1:
            raise Exception("no validation data")

        # Save metainfo for later
        self.meta = {
          'mini_batch_size' : mini_batch_size,
          'eta' : eta,
          'lmbda' : lmbda,
          'momentum' : momentum,
          'patience' : patience,
          'patience_increase' : patience_increase,
          'improvement_threshold' : 0.995,
          'validation_frequency' : validation_frequency,
          'n_hidden' : self.layers[1].n_in,
          'n_input' : self.layers[0].n_in,
          'training_data' : len(training_data),
          'validation_data' : len(validation_data),
          'algorithm' : 'RBMSProp'
        }


        # Prepare Theano shared variables with the shape and type of
        # The train, valid batches.
        train_x_zeros, train_y_zeros = training_data.next()
        training_x = tshared(train_x_zeros)
        training_y = tshared(train_y_zeros)
        training_data.reset()
        valid_x_zeros, valid_y_zeros = validation_data.next()
        validation_x = tshared(valid_x_zeros)
        validation_y = tshared(valid_y_zeros)
        validation_data.reset()

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_x) / mini_batch_size
        tota_num_training_batches = num_training_batches * len(training_data)
        num_validation_batches = size(validation_x) / tota_num_training_batches

        # define the (regularized) cost function,
        # symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost=cost, wrt=self.params)

        # RBMSProp
        rho = 0.9
        epsilon = 1e-6
        updates = []
        for p, g in zip(self.params, grads):
            acc = theano.shared(p.get_value() * 0.,
                                 broadcastable=p.broadcastable)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, T.cast(p - eta * g, theano.config.floatX)))

        # With MOMENTUM
        #updates = []
        # m = 0.1
        #for param, grad in zip(self.params, grads):
            #v = theano.shared(param.get_value() * 0.,
                              #broadcastable=param.broadcastable)
            #updates.append((param, param - eta * v))
            #updates.append((v, m * v + (1. - m) * grad))

        # Naive SGD
        #updates = [(param, param-eta*grad)
                   #for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.

        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        # Do the actual training
        best_validation_accuracy = 1.0
        done_looping = False

        val_per_epochs = training_data.actual_full_length() / mini_batch_size
        validation_frequency = int(val_per_epochs/ validation_frequency)

        iteration = 0
        for epoch in xrange(epochs):
            if done_looping: break
            train_itr = 0
            for train_x, train_y in training_data:
                if done_looping: break
                train_itr += 1
                training_x = tshared(train_x)
                training_y = tshared(train_y)
                for minibatch_index in xrange(num_training_batches):

                    iteration += 1
                    cost = train_mb(minibatch_index)

                    if iteration % validation_frequency == 0:
                        valid_acc = []
                        for valid_x, valid_y in validation_data:
                            validation_x = tshared(valid_x)
                            validation_y = tshared(valid_y)
                            valid_acc.append(
                                    [validate_mb_accuracy(j)
                                     for j in xrange(num_validation_batches)])
                        validation_accuracy = np.mean(valid_acc)
                        if metric_recorder:
                            metric_recorder.record(cost=cost,
                                validation_accuracy=validation_accuracy,
                                epoch=epoch, iteration=iteration)

                        print("Epoch {0}: validation accuracy {1}".format(
                            epoch, validation_accuracy))
                        if validation_accuracy <= best_validation_accuracy:
                            print "Best validation accuracy to date."
                            if save_dir:
                                # save model
                                self.meta['iteration'] = iteration
                                self.meta['accuracy'] = validation_accuracy
                                self.meta['cost'] = cost
                                self.save(save_dir + \
                                          "%d_model.pkl" % iteration)
                            # increase patience
                            if (validation_accuracy < best_validation_accuracy*
                                improvement_threshold):
                                patience = max(patience,
                                               int(iteration * patience_increase))
                                print "iter {0}, patience {1}".format(
                                    iteration, patience)
                            best_validation_accuracy = validation_accuracy

                    if patience <= iteration:
                        print "iter %i" % iteration
                        print "patience %i" % patience
                        print 'break'
                        done_looping = True
                        break

            print "iter %i" % iteration
            print "patience %i" % patience

        print("Finished training network.")
        return best_validation_accuracy

#### Define layer types

class AutoencoderLayer():

    def __init__(self, n_in=None, n_hidden=None, w=None, b_hid=None,
                 b_vis=None, activation_fn=sigmoid, p_dropout=0.0,
                 representative_layer=None):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout

        if not w:
            w = theano.shared(
                np.asarray(
                    np.random.uniform(
                        low=-np.sqrt(6. / (n_in + n_hidden)),
                        high=np.sqrt(6. / (n_in + n_hidden)),
                        size=(n_in, n_hidden)
                    ),
                    dtype=theano.config.floatX),
                name='w', borrow=True)

        if not b_vis:
            b_vis = theano.shared(
                np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_in,)),
                           dtype=theano.config.floatX),
                name='bvis', borrow=True)

        if not b_hid:
            b_hid = theano.shared(
                np.asarray(np.random.normal(loc=0.0, scale=1.0,
                                            size=(n_hidden,)),
                           dtype=theano.config.floatX),
                name='bhid', borrow=True)
        b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0,
                                        size=(n_hidden,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)

        self.w = w # shared weights
        self.b = b # bias for normal layer
        self.b_hid = b_hid # hidden bias for AE
        self.b_prime = b_vis # visible bias for AE
        self.w_prime = self.w.T # Hidden weights for AE

        self._params = [self.w, self.b_hid, self.b_prime]
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def get_hidde_values(self, inpt):
        return self.activation_fn(T.dot(inpt, self.w) + self.b_hid)

    def get_reconstructed_input(self, hidden):
        return self.activation_fn(T.dot(hidden, self.w_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level=None, eta=None):
        y = self.get_hidde_values(self.inpt)
        z = self.get_reconstructed_input(y)
        cost = T.nnet.binary_crossentropy(z, self.inpt).mean()
        grads = T.grad(cost=cost, wrt=self._params)

        # RBMSProp
        rho = 0.9
        epsilon = 1e-6
        updates = []
        for p, g in zip(self._params, grads):
            acc = theano.shared(p.get_value() * 0.,
                                 broadcastable=p.broadcastable)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, T.cast(p - eta * g, theano.config.floatX)))

        return (cost, updates)

    def train(self, training_data=None, batch_size=200, eta=0.25, epochs=1):
        index = T.lscalar() # Minibatch index
        x = T.matrix("x") # Inputdata

        self.set_inpt(x, x, batch_size)
        cost, updates = self.get_cost_updates(eta=eta)

        # Prepare Theano shared variables with the shape and type of
        # The train, valid batches.
        train_x_zeros, _ = training_data.next()
        training_x = tshared(train_x_zeros)
        training_data.reset()

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_x) / batch_size

        train_mb = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: training_x[index * batch_size: (index + 1) * batch_size]
            }
        )

        for epoch in xrange(epochs):
            c = []
            for train_x, _ in training_data:
                training_x = tshared(train_x)
                for batch_index in xrange(num_training_batches):
                    c.append(train_mb(batch_index))

            print "Trainig epoch %d, cost %f" % (epoch, np.mean(c))

class FullyConnectedLayer():

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.sqr(self.output - y).mean()

    def cost(self, net):
        # cost = T.sqr(self.output - net.y).mean()
        # cost = T.sqr(self.output_dropout - net.y).mean()
        cost = T.nnet.binary_crossentropy(self.output_dropout, net.y).mean()
        return cost

    def __getstate__(self):
        return (self.n_in, self.n_out, self.activation_fn,
                self.p_dropout, self.inpt, self.output,
                self.y_out, self.inpt_dropout, self.output_dropout,
                self.w, self.b)

    def __setstate__(self, state):
        n_in, n_out, activation_fn, p_dropout, inpt, output, \
        y_out, inpt_dropout, output_dropout, w, b = state
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        self.inpt = inpt
        self.output = output
        self.y_out = y_out
        self.inpt_dropout = inpt_dropout
        self.output_dropout = output_dropout
        self.w = w
        self.b = b
        self.params = [self.w, self.b]

#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data.get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)

def tshared(data):
    dtype = theano.config.floatX
    return theano.shared(np.asarray(data, dtype=dtype),  borrow=True)
