"""network.py
~~~~~~~

A Theano-based program for training and running simple and
deep neural networks.

It supports the layer types fully connected and denoising autoencoder. The latter can also can be stacked.

The training and validation data has to be in form of a
python Iterater, which gives back the X and y as indipendent
numpy arrays.

    for X, y in iterator:
        X.shape -> (int, int)
        X.shape -> (int, int)

The training can be stored by a metric_recorder Class. It
has to have the methods 'record' following signature.

    record(self, job_id=None, cost=None, validation_accuracy=None,
           epoch=None, iteration=None, second=None, type='train',
           eta=None)

and record_training_info with following signature

    record_training_info(self, infos:dict )

This program depends on Michael Nielsen's network3.py
github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network3.py and extends it with many features like stacked denoising autoencoder.
It incorporates ideas from the Theano documentation on
multilayer neural nets and stacked denoising autoencoder (notably,
http://deeplearning.net/tutorial/mlp.htm, and
http://deeplearning.net/tutorial/SdA.html), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).
Future more it uses the implementation of the rmsprop algorithmus
by Alec Radford (github.com/Newmu/Theano-Tutorials/blob/master/4_modern_net.py)

"""

#### Libraries

# Standard library
import cPickle
import pdb

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams
from theano.tensor.shared_randomstreams import RandomStreams

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid

static_rnd = np.random.RandomState()

#### Main class used to construct and train networks
class Network():

    def __init__(self, layers, mbs):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mbs`, mini batch size, to be used during training
        by the gradient descend algorithm.
        """
        self.layers = layers
        self.mbs = mbs
        self.params = [param
                       for layer in self.layers
                       for param in layer.params]
        self.x = T.matrix("x", dtype=theano.config.floatX)
        self.y = T.matrix("y", dtype=theano.config.floatX)
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mbs)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout,
                           self.mbs)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        self.meta = {}

    def __getstate__(self):
        """Returns the attributes, which has to be stored by cPickle"""
        return (self.layers, self.mbs,
                self.x, self.y, self.params, self.meta)

    def __setstate__(self, state):
        """Recovers the state by setting the attributes loaded by cPickle"""
        layers = mbs = x = y = params = meta = None
        if len(state) == 5:
          layers, mbs, x, y, params = state
        else:
          layers, mbs, x, y, params, meta = state
        self.layers = layers
        self.mbs = mbs
        self.x = x
        self.y = y
        self.params = params
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        if meta: self.meta = meta

    def predict(self, data):
        """Predicts the y's for a given set of Xs as data numpy.ndarray"""

        ext_size = self.mbs - (data.shape[0] % self.mbs)
        ext = np.zeros((ext_size, data.shape[1]))
        shape = (data.shape[0]+ext_size, data.shape[1])
        shared_data = tshared(np.append(data, ext).reshape(shape))
        i = T.lscalar() # mini-batch index
        predict = theano.function([i],
            outputs=self.layers[-1].output,
            givens={ self.x: shared_data[i*self.mbs: \
                                         (i+1)* self.mbs] })
        num_batches = shape[0] / self.mbs
        out = np.asarray([predict(j) for j in xrange(num_batches)])
        del shared_data
        return out.reshape(shape[0],out.shape[2])[:-ext_size]

    def save(self, filename='model.pkl'):
        """Saves the model to a file a the given filename path"""
        f = open(filename, 'wb')
        cPickle.dump(self, f)

    def pretrain_autoencoders(self, tdata=None, mbs=200, eta=0.25, epochs=1,
                              metric_recorder=None, save_dir=None):
        """Detects all AutoencoderLayer's and trains them layerwise."""
        aes = [layer
               for layer in self.layers
               if isinstance(layer, AutoencoderLayer)]
        for index, ae in enumerate(aes):
            ae.train(tdata=tdata, mbs=mbs,
                     eta=eta, epochs=epochs, ff_layers=aes[:index],
                     metric_recorder=metric_recorder, level=index)
        if save_dir: self.save(save_dir + "pretrained_model.pkl")

    def rms_prop(self, grads, lr):
        """given the gradients and the learning rate, calculates the update
        rules by the rmsprop algorithm and gives them back."""
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
            updates.append((p, T.cast(p - lr * g, theano.config.floatX)))
        return updates

    def naive_sgd(self, grads=None, lr=None, momentum=0.0):
        """given the gradients and the learning rate, calculates the update
        rules by the stochastic gradient descent algorithm and gives them back.
        Uses momentum when momentum is set."""
        if momentum == 0.0:
            return [(param, param-lr * grad)
                    for param, grad in zip(self.params, grads)]
        else:
            updates = []
            m = momentum
            for param, grad in zip(self.params, grads):
                v = theano.shared(param.get_value() * 0.,
                                  broadcastable=param.broadcastable)
                updates.append((param, param - lr * v))
                updates.append((v, m * v + (1. - m) * grad))
            return updates

    def get_layer_string(self):
        """Returns a string representation of the layers for documentation."""
        return "-".join([layer.to_string() for layer in self.layers])

    def get_layer_dropout_string(self):
        """Returns a sring representation of the dropout configuration
        for documentation"""
        return ", ".join([str(layer.p_dropout) for layer in self.layers])

    def train(self, tdata=None, vdata=None, epochs=10, mbs=100, eta=0.025,
              eta_min=None, lmbda=0.0, momentum=0.0, algorithm='rmsprop',
              patience_increase=2, improvement_threshold=0.995,
              validation_frequency=1, metric_recorder=None, save_dir=None,
              early_stoping=True):
        """Train the network using mini-batch and a given algorithm.

        Keyword arguments:

        tdata     -- train data iterator (BatchImgProcessor)
        vdata     -- validation data iterator (BatchImgProcessor)
        epochs    -- number of epochs to train
        mbs       -- Mini-Batch-Size for training
        eta       -- learning rate max
        eta_min   -- learning rate min, when set it calculates a linspace,
                     from eta to eta_min per epoche.
        lambda    -- l2 reguarisation coeffiziet
        momentum  -- Momentunm for stochastic gradient descent. Ignored
                     when train with rmsprop.
        algorithm -- ('rmsprop' | 'sgd') which algorithm for the update
                     rules to use. ('rmsprop')
        save_dir  -- Dir where to save the model, when better is found.
        improvement_threshold -- How much the better model must be that
                                 the patience gets increased.
        patience_increase  -- How much increase the patience when a better
                              model is found.
                              Ignored when not using early stopping
        validation_frequency -- How oftern per epoch should be validated
        metric_recorder -- Instance of a MetricRecorder to record the training
        early_stoping -- When set it stops befor reaching the last epoch, when
                         the model is not getting better.
        """

        imp_thresh = improvement_threshold
        pat_incr = patience_increase

        if not vdata or len(vdata) < 1:
            raise Exception("no validation data")

        # Save metainfo
        self.meta = {
          'mini_batch_size' : mbs,
          'random_mode' : tdata.random_mode,
          'eta' : eta,
          'eta_min' : eta_min,
          'lmbda' : lmbda,
          'momentum' : momentum,
          'patience_increase' : pat_incr,
          'improvement_threshold' : imp_thresh,
          'validation_frequency' : validation_frequency,
          'dropouts' : self.get_layer_dropout_string(),
          'layers' : self.get_layer_string(),
          'training_data' : tdata.full_lenght(),
          'validation_data' : vdata.full_lenght(),
          'algorithm' : algorithm
        }

        if metric_recorder:
            metric_recorder.record_training_info(infos=self.meta)

        # Prepare Theano shared variables with the shape and type of
        # The train, valid batches.
        train_x_zeros, train_y_zeros = tdata.next()
        training_x = tshared(train_x_zeros)
        training_y = tshared(train_y_zeros)
        tdata.reset()
        valid_x_zeros, valid_y_zeros = vdata.next()
        validation_x = tshared(valid_x_zeros)
        validation_y = tshared(valid_y_zeros)
        vdata.reset()

        # compute number of minibatches for training, validation and testing
        n_train_batches = size(training_x) / mbs
        tota_n_train_batches = n_train_batches * len(tdata)
        n_valid_batches = size(validation_x) / mbs

        # define the (regularized) cost function,
        # symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/n_train_batches
        grads = T.grad(cost=cost, wrt=self.params)

        # define update rules
        lr = T.scalar()
        updates = []
        if algorithm == 'rmsprop':
            updates = self.rms_prop(grads, lr)
        elif algorithm == 'sgd':
            updates = self.naive_sgd(grads, lr, momentum=momentum)

        # define theano functions to train a mini-batch, and to compute the
        # accuracy for validation mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i, lr], cost, updates=updates,
            givens={
                self.x: training_x[i*self.mbs: (i+1)*self.mbs],
                self.y: training_y[i*self.mbs: (i+1)*self.mbs]
            }, allow_input_downcast=True)
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: validation_x[i*self.mbs: (i+1)*self.mbs],
                self.y: validation_y[i*self.mbs: (i+1)*self.mbs]
            })

        # set start meta variables
        best_valid_acc = 1.0
        done_looping = False # for early stopping
        val_per_epochs = tdata.actual_full_length() / mbs
        validation_frequency = int(val_per_epochs/ validation_frequency)
        patience = validation_frequency * 4

        # calc linear eta decrease
        if not eta_min: eta_min = eta
        etas = np.linspace(eta, eta_min, epochs)
        cost = iteration = 0

        # Do the actual training
        for epoch in xrange(epochs): # for all epoches
            if done_looping: break # in calse of early stopping
            eta = etas[epoch] # Update eta
            for train_x, train_y in tdata: # go through all batches
                if done_looping: break # in case of early stopping
                # update theano shared variables
                training_x.set_value(train_x, borrow=True)
                training_y.set_value(train_y, borrow=True)
                for minibatch_index in xrange(n_train_batches):
                    # go through batch in mini batches
                    iteration += 1
                    # actual training
                    cost += train_mb(minibatch_index, eta)

                    # Validation
                    if iteration % validation_frequency == 0:
                        valid_accs = []
                        # Go through the validation data, and get the acc
                        for valid_x, valid_y in vdata:
                            # update theano shared variables
                            validation_x.set_value(valid_x, borrow=True)
                            validation_y.set_value(valid_y, borrow=True)
                            valid_accs.append([validate_mb_accuracy(j)
                                for j in xrange(n_valid_batches)])
                        valid_acc = np.mean(valid_accs) # mean over valid_acc

                        # record when metric_recorder is available
                        if metric_recorder:
                            metric_recorder.record(cost=train_cost,
                                validation_accuracy=valid_acc,
                                epoch=epoch, iteration=iteration)

                        print("Epoch {0}: validation accuracy {1}".format(
                            epoch, valid_acc))

                        # When best model is found
                        if valid_acc <= best_valid_acc:
                            print "Best validation accuracy to date."
                            # save model when save_dir is set
                            if save_dir:
                                self.meta['iteration'] = iteration
                                self.meta['accuracy'] = valid_acc
                                self.meta['cost'] = cost
                                name = "%d_model.pkl" % iteration
                                self.save(save_dir + name)

                            # increase patience
                            if (valid_acc < best_valid_acc * imp_thresh):
                                patience = max(patience, int(iteration *
                                                             pat_incr))
                                print "iter {0}, patience {1}".format(
                                    iteration, patience)
                            best_valid_acc = valid_acc

                    # early stopping when patience is over
                    if patience <= iteration and early_stoping:
                        print "iter %i" % iteration
                        print "patience %i" % patience
                        done_looping = True
                        break

            print "iter %i" % iteration
            print "patience %i" % patience

        print("Finished training network.")
        return best_valid_acc

#### Define layer types

class Layer():
    """ 'Base' Class for layers """

    def set_inpt(self, inpt, inpt_dropout, mbs):
        """Method to set the input variable to connect the layers."""
        return NotImplemented

    def to_string(self):
        """Returns a string representation of the layer"""
        return NotImplemented

    def __getstate__(self):
        """Returns the attributes, which has to be stored by cPickle"""
        return NotImplemented

    def __setstate__(self, state):
        """Recovers the state by setting the attributes loaded by cPickle"""
        return NotImplemented


class AutoencoderLayer(Layer):
    """AutoencoderLayer with teight weitghts for a deep neural network."""

    def __init__(self, n_in=None, n_hidden=None, activation_fn=sigmoid,
                 p_dropout=0.0, corruption_level=0.0, rnd=static_rnd):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        self.theano_rng = RandomStreams(rnd.randint(2 ** 30))
        self.corruption_level = corruption_level
        self.rnd = rnd

        # shared weights
        self.w = tshared(self.rnd.uniform(
                    low=-np.sqrt(6. / (n_in + n_hidden)),
                    high=np.sqrt(6. / (n_in + n_hidden)),
                    size=(n_in, n_hidden)), 'w')
        # bias for normal layer
        self.b = tshared(self.rnd.normal(loc=0.0, scale=1.0,
                                        size=(n_hidden,)), 'bhid')
        # visible bias for AE
        self.b_prime = tshared(self.rnd.normal(loc=0.0, scale=1.0,
                                        size=(n_in,)), 'bvis')
        self.w_prime = self.w.T # Hidden teights weights for AE

        self._params = [self.w, self.b, self.b_prime]
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mbs):
        """Method to set the input variable to connect the layers."""

        self.inpt = inpt.reshape((mbs, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.hidden_output = self.get_hidden_values(self.inpt)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mbs, self.n_in)), self.p_dropout,
            rnd=self.rnd)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def to_string(self):
        """Returns a string representation of the layer"""
        af = 'sgm'
        if self.activation_fn == ReLU: af = 'rlu'
        if self.corruption_level == 0.0:
            return "Ae[%s](%d, %d)" % (af, self.n_in, self.n_hidden)
        else:
            return "dAe[%s, %.03f](%d, %d)" %  \
            (af, self.corruption_level, self.n_in, self.n_hidden)

    def get_corrupted_input(self):
        """Returns a corrupted version of the input for denoising"""
        if self.corruption_level == 0.0: return self.inpt
        return self.theano_rng.binomial(size=self.inpt.shape, n=1,
                                  p=1 - self.corruption_level,
                                  dtype=theano.config.floatX) * self.inpt

    def forward(self, data, mbs=500):
        """Forwards a given data set through the layer. Used for preprocess
        data in a stacked autoencoder"""
        shared_data = tshared(data)
        i = T.lscalar() # mini-batch index
        fwd = theano.function(
            [i], self.hidden_output,
            givens={
                self.inpt: shared_data[i*mbs: \
                                       (i+1)* mbs]
            })
        num_batches = len(data)/mbs
        out = np.asarray([fwd(j) for j in xrange(num_batches)])
        del shared_data
        return out.reshape(out.shape[0] * out.shape[1], out.shape[2])

    def get_hidden_values(self, inpt):
        """Calculates the values for the hidden layer"""
        return sigmoid(T.dot(inpt, self.w) + self.b)

    def get_reconstructed_input(self, hidden):
        """reconstructs the input to the output layer"""
        return sigmoid(T.dot(hidden, self.w_prime) + self.b_prime)

    def get_cost_updates(self, eta=None):
        """Returns costs and updates rules for training with the rmsprop
        algorithm"""
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

    def train(self, tdata=None, ff_layers=[], metric_recorder=None,
              mbs=200, eta=0.25, epochs=1, level=0):
        """Train the layer

        Keyword arguments:

        tdata     -- train data iterator (BatchImgProcessor)
        epochs    -- number of epochs to train
        mbs       -- Mini-Batch-Size for training
        eta       -- learning rate max
        ff_layers -- fastforward layers form autoencoders before,
                     to preprocess the data
        level     -- indicator of which layer is trained.
        metric_recorder -- MetricRecorder instance to record the training
        """
        index = T.lscalar() # Minibatch index
        x = T.matrix("x") # Inputdata

        self.set_inpt(x, x, mbs)
        cost, updates = self.get_cost_updates(eta=eta)

        # Prepare Theano shared variables with the shape and type of
        # The train, valid batches.
        train_x_zeros, _ = tdata.next()
        for l in ff_layers: train_x_zeros = l.forward(train_x_zeros)
        training_x = tshared(train_x_zeros)
        tdata.reset()

        # compute number of minibatches for training, validation and testing
        n_train_batches = size(training_x) / mbs

        train_mb = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: training_x[index * mbs: (index + 1) * mbs]
            }
        )

        for epoch in xrange(epochs):
            c = []
            for train_x, _ in tdata:
                for l in ff_layers: train_x = l.forward(train_x)
                training_x.set_value(train_x, borrow=True)
                for batch_index in xrange(n_train_batches):
                    c.append(train_mb(batch_index))

            print "Trainig epoch %d, cost %f" % (epoch, np.mean(c))
            if metric_recorder:
                metric_recorder.record(cost=np.mean(c), epoch=epoch, eta=eta,
                                       type='pretrain_%d' % level)

    def __getstate__(self):
        """Returns the attributes, which has to be stored by cPickle"""
        return(self.n_in, self.n_hidden, self.activation_fn, \
               self.p_dropout, self.inpt, self.output, self.inpt_dropout, \
               self.output_dropout, self.w, self.b, self.w_prime, \
               self.b_prime)

    def __setstate__(self, state):
        """Recovers the state by setting the attributes loaded by cPickle"""
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
                 rnd=static_rnd):
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

    def set_inpt(self, inpt, inpt_dropout, mbs):
        """Method to set the input variable to connect the layers."""
        self.inpt = inpt.reshape((mbs, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mbs, self.n_in)), self.p_dropout,
            rnd=self.rnd)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)

    def accuracy(self, y):
        """Return the accuracy for the mini-batch as RMSE"""
        return T.mean((y - self.output)**2) ** 0.5

    def to_string(self):
        """Returns a string representation of the layer"""
        return "FC(%d, %d)" % (self.n_in, self.n_out)

    def cost(self, net):
        """Returns the validation cost, here binary_crossentropy"""
        return T.nnet.binary_crossentropy(self.output_dropout, net.y).mean()

    def __getstate__(self):
        """Returns the attributes, which has to be stored by cPickle"""
        return (self.n_in, self.n_out, self.activation_fn,
                self.p_dropout, self.inpt, self.output,
                self.y_out, self.inpt_dropout, self.output_dropout,
                self.w, self.b)

    def __setstate__(self, state):
        """Recovers the state by setting the attributes loaded by cPickle"""
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
    """Returns the size of the dataset `data`."""
    return data.get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout, rnd=static_rnd):
    """Mask the weights for random dropouts"""
    srng = shared_randomstreams.RandomStreams(
        rnd.randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)

def tshared(data, name=None):
    """Creates and returns a mutable theano shared variable"""
    dtype = theano.config.floatX
    return theano.shared(np.asarray(data, dtype=dtype), name=name, borrow=True)
