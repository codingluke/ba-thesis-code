
import sys, os
import numpy as np
import PIL.Image
import cPickle
import pdb
import theano
from timeit import default_timer as timer

lib_path = os.path.abspath(os.path.join('..', '..', 'src'))
sys.path.append(lib_path)

from network import Network, ConvPoolLayer, FullyConnectedLayer, \
                    tanh, ReLU
from preprocessor import BatchImgProcessor

border=2
n_hidden_layer=80
eta=0.01
lmbda=0.0
mini_batch_size = 500

BatchProcessor = BatchImgProcessor.load(
    X_dirpath='../../../data/train/*',
    y_dirpath='../../../data/train_cleaned/',
    batchsize=50000, border=border, limit=5,
    dtype=theano.config.floatX)

training_data = BatchProcessor(modus='train', random=True)
validation_data = BatchProcessor(modus='valid')

n_in = (2*border+1)**2
net = Network([FullyConnectedLayer(n_in=n_in, n_out=n_hidden_layer),
               FullyConnectedLayer(n_in=n_hidden_layer, n_out=1)],
              mini_batch_size)

result = net.SGD(training_data=training_data, epochs=100,
        mini_batch_size=mini_batch_size, eta=eta,
        validation_data=validation_data, lmbda=lmbda,
        momentum=None, patience=20000, patience_increase=2,
        improvement_threshold=0.995, validation_frequency=5000)
