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
from preprocessor import BatchProcessor

def train(job_id, border, n_hidden_layer, eta, lmbda):
    starttime = timer()
    mbs = 500

    training_data = BatchProcessor(
        X_dirpath='../../../data/train/*',
        y_dirpath='../../../data/train_cleaned/',
        batchsize=500000,
        border=border,
        limit=20,
        dtype=theano.config.floatX)

    validation_data = BatchProcessor(
        X_dirpath='../../../data/valid/*',
        y_dirpath='../../../data/train_cleaned/',
        batchsize=500000,
        border=border,
        limit=20,
        dtype=theano.config.floatX)

    n_in = (2*border+1)**2
    net = Network([FullyConnectedLayer(n_in=n_in, n_out=n_hidden_layer),
                   FullyConnectedLayer(n_in=n_hidden_layer, n_out=1)],
                  mbs)

    result = net.train(tdata=training_data, epochs=100,
            mbs=mbs, eta=eta,
            vdata=validation_data, lmbda=lmbda,
            momentum=None, patience_increase=2,
            improvement_threshold=0.995, validation_frequency=5000)

    endtime = timer()

    print 'Time = %f' % (endtime - starttime)
    print 'Result = %f' % result
    #time.sleep(np.random.randint(60))
    return float(result)

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    return train(job_id, params['border'][0],
                 params['n_hidden_layer'][0],
                 params['eta'][0],
                 params['lmbda'][0])
