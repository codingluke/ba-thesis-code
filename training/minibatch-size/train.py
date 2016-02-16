import sys, os
import numpy as np
import PIL.Image
import cPickle
import pdb
import theano
from timeit import default_timer as timer

lib_path = os.path.abspath(os.path.join('..', '..', 'src'))
sys.path.append(lib_path)

from network import Network, FullyConnectedLayer, \
                    tanh, ReLU
from preprocessor import BatchImgProcessor
from metric import MetricRecorder

rnd = np.random.RandomState()

def train(job_id, mbs):
    #print "Job ID: %d" % job_id
    eta = 0.01 # 1-7 0.01
    border = 2
    n_hidden_layer = 80
    metric_recorder = MetricRecorder(config_dir_path='.', job_id=job_id)
    C = {
        'X_dirpath' : '../../../data/train_all/*',
        'y_dirpath' : '../../../data/train_cleaned/',
        'batchsize' : 500000,
        'limit' : 20,
        'epochs' : 4,
        'patience' : 70000,
        'patience_increase' : 2,
        'improvement_threshold' : 0.995,
        'validation_frequency' : 20,
        'lmbda' : 0.0,
        'training_size' : None,
        'validation_size' : None,
        'algorithm' : 'RMSProp',
        'mini_batch_size': mbs
    }

    training_data = BatchImgProcessor(
        X_dirpath='../../../data/train_all/*',
        y_dirpath='../../../data/train_cleaned/',
        batchsize=C['batchsize'],
        border=border,
        limit=C['limit'],
        random=True,
        random_mode='fully',
        dtype=theano.config.floatX,
        rnd=rnd)

    validation_data = BatchImgProcessor(
        X_dirpath='../../../data/train/*',
        y_dirpath='../../../data/train_cleaned/',
        batchsize=C['batchsize'],
        random=False,
        border=border,
        limit=C['limit'],
        dtype=theano.config.floatX,
        rnd=rnd)

    C['training_size'] = training_data.size()
    C['validation_size'] = validation_data.size()
    print "Training size: %d" % C['training_size']
    print "Validation size: %d" % C['validation_size']

    metric_recorder.add_experiment_metainfo(constants=C)
    metric_recorder.start()

    n_in = (2*border+1)**2
    net = Network([FullyConnectedLayer(n_in=n_in, n_out=n_hidden_layer,
                    rnd=rnd),
                   FullyConnectedLayer(n_in=n_hidden_layer, n_out=1,
                    rnd=rnd)],
                  C['mini_batch_size'])

    result = net.train(tdata=training_data, epochs=C['epochs'],
                     mbs=C['mini_batch_size'], eta=eta,
                     vdata=validation_data, lmbda=C['lmbda'],
                     momentum=None, patience_increase=C['patience_increase'],
                     improvement_threshold=C['improvement_threshold'],
                     validation_frequency=C['validation_frequency'],
                     metric_recorder=metric_recorder,
                     save_dir='./model/%d_' % metric_recorder.job_id,
                     early_stoping=False)

    print 'Time = %f' % metric_recorder.stop()
    print 'Result = %f' % result
    return float(result)

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d'
    print params
    return train(job_id, params['size'][0])

if __name__ == '__main__':
    #job_id = 12 # job 8 strange... , ab 12 mit metadata im model
    for i in [50, 100, 200, 500, 1000, 1500]:
        main(None, {'size' : [i]})
    #main(7, {'size' : [1500]})
