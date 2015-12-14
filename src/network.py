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

#### Load the MNIST data
def load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

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
        self.x = T.matrix("x")
        self.y = T.matrix("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout,
                           self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def __getstate__(self):
        return (self.layers, self.mini_batch_size,
                self.x, self.y, self.params)

    def __setstate__(self, state):
        layers, mini_batch_size, x, y, params = state
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.x = x
        self.y = y
        self.params = params
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

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

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data=None, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        first_file = True

        # Prepare Theano shared variables with the shape and type of
        # The train, valid batches.
        train_x_zeros, train_y_zeros = training_data.zeros()
        training_x = tshared(train_x_zeros)
        training_y = tshared(train_y_zeros)
        valid_x_zeros, valid_y_zeros = validation_data.zeros()
        validation_x = tshared(valid_x_zeros)
        validation_y = tshared(valid_y_zeros)

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_x) / mini_batch_size
        num_validation_batches = size(validation_x) / mini_batch_size

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
            updates.append((p, p - eta * g))

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

        patience = 20000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant

        validation_frequency = 5000
        #validation_frequency = min(num_training_batches * len(training_data),
                                   #patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
        print validation_frequency

        iteration = 0
        for epoch in xrange(epochs):
            if done_looping: break
            for train_x, train_y in training_data:
                training_x = tshared(train_x)
                training_y = tshared(train_y)
                for minibatch_index in xrange(num_training_batches):
                    iteration += 1
                    train_mb(minibatch_index)

                    if (iteration + 1) % validation_frequency == 0:
                        valid_acc = []
                        for valid_x, valid_y in validation_data:
                            validation_x = tshared(valid_x)
                            validation_y = tshared(valid_y)
                            valid_acc.append(
                                    [validate_mb_accuracy(j)
                                     for j in xrange(num_validation_batches)])
                        validation_accuracy = np.mean(valid_acc)

                        print("Epoch {0}: validation accuracy {1}".format(
                            epoch, validation_accuracy))
                        if validation_accuracy <= best_validation_accuracy:
                            print "Best validation accuracy to date."
                            # increase patience
                            if (validation_accuracy < best_validation_accuracy*
                                improvement_threshold):
                                patience = max(patience,
                                               iteration * patience_increase)
                                print "iter {0}, patience {1}".format(
                                    iteration, patience)
                            best_validation_accuracy = validation_accuracy

                    if patience <= iteration:
                        done_looping = True
                        break

            print "iter %i" % iteration
            print "patience %i" % patience

        print("Finished training network.")

#### Define layer types

class ConvPoolLayer():
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
       # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

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
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.sqr(self.output - y).mean()

    def cost(self, net):
        cost = T.sqr(self.output - net.y).mean()
        #cost = T.nnet.binary_crossentropy(self.output, net.y).mean()
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
    return theano.shared(np.asarray(data, dtype=dtype), borrow=True)
