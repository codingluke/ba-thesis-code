"""network3.py
aaaaaaa~~~~~~~

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
from theano.tensor.shared_randomstreams import RandomStreams

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


#### Main class used to construct and train networks
class Network():

    def __init__(self, layers, batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.batch_size = batch_size
        self.params = [param
                       for layer in self.layers
                       for param in layer.params]
        self.x = T.matrix("x", dtype=theano.config.floatX)
        self.y = T.matrix("y", dtype=theano.config.floatX)
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout,
                           self.batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        self.meta = {}
        self.eta = 0.02

    def __getstate__(self):
        return (self.layers, self.batch_size,
                self.x, self.y, self.params, self.meta)

    def __setstate__(self, state):
        layers = batch_size = x = y = params = meta = None
        if len(state) == 5:
          layers, batch_size, x, y, params = state
        else:
          layers, batch_size, x, y, params, meta = state
        self.layers = layers
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.params = params
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        if meta: self.meta = meta

    def predict(self, data):
        ext_size = self.batch_size - (data.shape[0] % self.batch_size)
        ext = np.zeros((ext_size, data.shape[1]))
        shape = (data.shape[0]+ext_size, data.shape[1])
        shared_data = tshared(np.append(data, ext).reshape(shape))
        i = T.lscalar() # mini-batch index
        predict = theano.function([i],
            outputs=self.layers[-1].output,
            givens={ self.x: shared_data[i*self.batch_size: \
                                         (i+1)* self.batch_size] })
        num_batches = shape[0] / self.batch_size
        out = np.asarray([predict(j) for j in xrange(num_batches)])
        del shared_data
        return out.reshape(shape[0],out.shape[2])[:-ext_size]

    def save(self, filename='model.pkl'):
        f = open(filename, 'wb')
        cPickle.dump(self, f)

    def pretrain_autoencoders(self, training_data=None, batch_size=200,
                              eta=0.25, epochs=1, metric_recorder=None,
                              save_dir=None):
        aes = [layer
               for layer in self.layers
               if isinstance(layer, AutoencoderLayer)]
        for index, ae in enumerate(aes):
            ae.train(training_data=training_data, batch_size=batch_size,
                     eta=eta, epochs=epochs, ff_layers=aes[:index],
                     metric_recorder=metric_recorder, level=index)
        if save_dir: self.save(save_dir + "pretrained_model.pkl")

    def rms_prop(self, grads):
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
            updates.append((p, T.cast(p - self.eta * g, theano.config.floatX)))
        return updates

    def naive_sgd(self, grads=None, eta=None, momentum=0.0):
        if momentum == 0.0:
            return [(param, param-self.eta * grad)
                    for param, grad in zip(self.params, grads)]
        else:
            updates = []
            m = momentum
            for param, grad in zip(self.params, grads):
                v = theano.shared(param.get_value() * 0.,
                                  broadcastable=param.broadcastable)
                updates.append((param, param - self.eta * v))
                updates.append((v, m * v + (1. - m) * grad))
            return updates

    def get_layer_string(self):
        return "-".join([layer.to_string() for layer in self.layers])

    def get_layer_dropout_string(self):
        return ", ".join([str(layer.p_dropout) for layer in self.layers])

    def SGD(self, training_data=None, epochs=10, batch_size=100,
            eta=0.025, validation_data=None, lmbda=0.0, momentum=0.0,
            patience=40000, patience_increase=2, improvement_threshold=0.995,
            validation_frequency=1, metric_recorder=None, save_dir=None,
            algorithm='rmsprop', early_stoping=True, eta_min=None):
        """Train the network using mini-batch stochastic gradient descent."""

        if not validation_data or len(validation_data) < 1:
            raise Exception("no validation data")

        # Save metainfo for later
        self.meta = {
          'mini_batch_size' : batch_size,
          'random_mode' : training_data.random_mode,
          'eta' : eta,
          'eta_min' : eta_min,
          'lmbda' : lmbda,
          'momentum' : momentum,
          'patience_increase' : patience_increase,
          'improvement_threshold' : 0.995,
          'validation_frequency' : validation_frequency,
          'dropouts' : self.get_layer_dropout_string(),
          'layers' : self.get_layer_string(),
          'training_data' : training_data.full_lenght(),
          'validation_data' : validation_data.full_lenght(),
          'algorithm' : algorithm
        }

        if metric_recorder:
            metric_recorder.record_training_info(infos=self.meta)

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
        num_training_batches = size(training_x) / batch_size
        tota_num_training_batches = num_training_batches * len(training_data)
        num_validation_batches = size(validation_x) / batch_size

        # define the (regularized) cost function,
        # symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost=cost, wrt=self.params)

        # define update rules
        updates = []
        if algorithm == 'rmsprop':
            updates = self.rms_prop(grads)
        elif algorithm == 'sgd':
            updates = self.naive_sgd(grads, momentum=momentum)

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.batch_size: (i+1)*self.batch_size],
                self.y:
                training_y[i*self.batch_size: (i+1)*self.batch_size]
            })

        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.batch_size: (i+1)*self.batch_size],
                self.y:
                validation_y[i*self.batch_size: (i+1)*self.batch_size]
            })

        # Do the actual training
        best_validation_accuracy = 1.0
        done_looping = False

        val_per_epochs = training_data.actual_full_length() / batch_size
        validation_frequency = int(val_per_epochs/ validation_frequency)
        patience = validation_frequency * 4

        if not eta_min: eta_min = eta
        etas = np.linspace(eta, eta_min, epochs)
        cost = 0
        iteration = 0
        for epoch in xrange(epochs):
            if done_looping: break
            train_it = 0
            self.eta = etas[epoch] # Update eta
            for train_x, train_y in training_data:
                train_it += 1
                if done_looping: break
                training_x.set_value(train_x, borrow=True)
                training_y.set_value(train_y, borrow=True)
                for minibatch_index in xrange(num_training_batches):
                    iteration += 1
                    cost += train_mb(minibatch_index)
                    if iteration % validation_frequency == 0:
                        valid_acc = []
                        for valid_x, valid_y in validation_data:
                            validation_x.set_value(valid_x, borrow=True)
                            validation_y.set_value(valid_y, borrow=True)
                            valid_acc.append(
                                    [validate_mb_accuracy(j)
                                     for j in xrange(num_validation_batches)])
                        validation_accuracy = np.mean(valid_acc)
                        train_cost = cost / iteration
                        if metric_recorder:
                            metric_recorder.record(cost=train_cost,
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

                    if patience <= iteration and early_stoping:
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
                 corruption_level=0.0, rnd=None):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        self.theano_rng = RandomStreams(rnd.randint(2 ** 30))
        self.corruption_level = corruption_level
        self.rnd = rnd

        if not w:
            w = tshared(self.rnd.uniform(
                        low=-np.sqrt(6. / (n_in + n_hidden)),
                        high=np.sqrt(6. / (n_in + n_hidden)),
                        size=(n_in, n_hidden)
                    ), 'w')

        if not b_vis:
            b_vis = tshared(self.rnd.normal(loc=0.0, scale=1.0,
                                            size=(n_in,)),
                            'bvis')

        if not b_hid:
            b_hid = tshared(self.rnd.normal(loc=0.0, scale=1.0,
                                            size=(n_hidden,)),
                            'bhid')

        self.w = w # shared weights
        self.b = b_hid # bias for normal layer
        self.b_prime = b_vis # visible bias for AE
        self.w_prime = self.w.T # Hidden weights for AE

        self._params = [self.w, self.b, self.b_prime]
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, batch_size):
        self.inpt = inpt.reshape((batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.hidden_output = self.get_hidden_values(self.inpt)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((batch_size, self.n_in)), self.p_dropout,
            rnd=self.rnd)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def to_string(self):
        af = 'sgm'
        if self.activation_fn == ReLU: af = 'rlu'
        if self.corruption_level == 0.0:
            return "Ae[%s](%d, %d)" % (af, self.n_in, self.n_hidden)
        else:
            return "dAe[%s, %.03f](%d, %d)" %  \
            (af, self.corruption_level, self.n_in, self.n_hidden)

    def get_corrupted_input(self):
        if self.corruption_level == 0.0: return self.inpt
        return self.theano_rng.binomial(size=self.inpt.shape, n=1,
                                  p=1 - self.corruption_level,
                                  dtype=theano.config.floatX) * self.inpt

    def forward(self, data, batch_size=500):
        shared_data = tshared(data)
        i = T.lscalar() # mini-batch index
        fwd = theano.function(
            [i], self.hidden_output,
            givens={
                self.inpt: shared_data[i*batch_size: \
                                       (i+1)* batch_size]
            })
        num_batches = len(data)/batch_size
        out = np.asarray([fwd(j) for j in xrange(num_batches)])
        del shared_data
        return out.reshape(out.shape[0] * out.shape[1], out.shape[2])


    def get_hidden_values(self, inpt):
        return sigmoid(T.dot(inpt, self.w) + self.b)

    def get_reconstructed_input(self, hidden):
        return sigmoid(T.dot(hidden, self.w_prime) + self.b_prime)

    def get_cost_updates(self, eta=None):
        y = self.get_hidden_values(self.get_corrupted_input())
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

    def train(self, training_data=None, ff_layers=[], metric_recorder=None,
              batch_size=200, eta=0.25, epochs=1, level=0):
        index = T.lscalar() # Minibatch index
        x = T.matrix("x") # Inputdata

        self.set_inpt(x, x, batch_size)
        cost, updates = self.get_cost_updates(eta=eta)

        # Prepare Theano shared variables with the shape and type of
        # The train, valid batches.
        train_x_zeros, _ = training_data.next()
        for l in ff_layers: train_x_zeros = l.forward(train_x_zeros)
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
                for l in ff_layers: train_x = l.forward(train_x)
                training_x.set_value(train_x, borrow=True)
                for batch_index in xrange(num_training_batches):
                    c.append(train_mb(batch_index))

            print "Trainig epoch %d, cost %f" % (epoch, np.mean(c))
            if metric_recorder:
                metric_recorder.record(cost=np.mean(c), epoch=epoch, eta=eta,
                                       type='pretrain_%d' % level)

    def __getstate__(self):
        return(self.n_in, self.n_hidden, self.activation_fn, \
               self.p_dropout, self.inpt, self.output, self.inpt_dropout, \
               self.output_dropout, self.w, self.b, self.w_prime, \
               self.b_prime)

    def __setstate__(self, state):
        n_in, n_hidden, activation_fn,  p_dropout, inpt, output, \
        inpt_dropout, output_dropout, w, b, w_prime, b_prime = state

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        self.inpt = inpt
        self.output = output
        self.inpt_dropout = inpt_dropout
        self.output_dropout = output_dropout
        self.w = w
        self.b = b
        self.w_prime = w_prime
        self.b_prime = b_prime
        self.params = [self.w, self.b]
        self._params = [self.w, self.b, self.b_prime]

class FullyConnectedLayer():

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0,
                 rnd=None):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        self.rnd = rnd

        # Initialize weights and biases
        self.w = tshared(self.rnd.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ), 'w')
        self.b = tshared(self.rnd.normal(loc=0.0, scale=1.0, size=(n_out,)),
                         'b')
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, batch_size):
        self.inpt = inpt.reshape((batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((batch_size, self.n_in)), self.p_dropout,
            rnd=self.rnd)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.sqr(self.output - y).mean()

    def to_string(self):
        return "FC(%d, %d)" % (self.n_in, self.n_out)

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

def dropout_layer(layer, p_dropout, rnd=None):
    srng = shared_randomstreams.RandomStreams(
        rnd.randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)

def tshared(data, name=None):
    dtype = theano.config.floatX
    return theano.shared(np.asarray(data, dtype=dtype), name=name, borrow=True)
