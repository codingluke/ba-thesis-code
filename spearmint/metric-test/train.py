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
from metric import MetricRecorder

def train(job_id, border, n_hidden_layer, eta, lmbda):
    print "Job ID: %d" % job_id
    metric_recorder = MetricRecorder(config_dir_path='.', job_id=job_id)
    C = {
        'X_dirpath' : '../../../data/train/*',
        'y_dirpath' : '../../../data/train_cleaned/',
        'mini_batch_size' : 500,
        'batchsize' : 5000,
        'limit' : 1,
        'epochs' : 1,
        'patience' : 20000,
        'patience_increase' : 2,
        'improvement_threshold' : 0.995,
        'validation_frequency' : 100
    }
    metric_recorder.add_experiment_metainfo(constants=C)
    metric_recorder.start()

    BatchProcessor = BatchImgProcessor.load(
        X_dirpath='../../../data/train/*',
        y_dirpath='../../../data/train_cleaned/',
        batchsize=C['batchsize'],
        border=border,
        limit=C['limit'],
        train_stepover=8,
        dtype=theano.config.floatX)

    training_data = BatchProcessor(modus='train', random=True)
    validation_data = BatchProcessor(modus='valid')
    print "Training size: %d" % len(training_data)
    print "Validation size: %d" % len(validation_data)

    n_in = (2*border+1)**2
    net = Network([FullyConnectedLayer(n_in=n_in, n_out=n_hidden_layer),
                   FullyConnectedLayer(n_in=n_hidden_layer, n_out=1)],
                  C['mini_batch_size'])

    result = net.SGD(training_data=training_data, epochs=C['epochs'],
                     mini_batch_size=C['mini_batch_size'], eta=eta,
                     validation_data=validation_data, lmbda=lmbda,
                     momentum=None, patience=C['patience'],
                     patience_increase=C['patience_increase'],
                     improvement_threshold=C['improvement_threshold'],
                     validation_frequency=C['validation_frequency'],
                     metric_recorder=metric_recorder)

    print 'Time = %f' % metric_recorder.stop()
    print 'Result = %f' % result
    return float(result)

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    return train(job_id, params['border'][0],
                 params['n_hidden_layer'][0],
                 params['eta'][0],
                 params['lmbda'][0])
