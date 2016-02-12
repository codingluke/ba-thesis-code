import sys, os
import numpy as np
import PIL.Image
import cPickle
import pdb
import theano
from timeit import default_timer as timer

lib_path = os.path.abspath(os.path.join('..', '..', 'src'))
sys.path.append(lib_path)

from network import Network, FullyConnectedLayer, AutoencoderLayer
from preprocessor import BatchImgProcessor
from metric import MetricRecorder

rnd = np.random.RandomState()

def train(job_id, params):
    print "Job ID: %d" % job_id
    border = 2
    n_hidden_layer = params['hidden']
    metric_recorder = MetricRecorder(config_dir_path='./config.json',
                                     job_id=job_id)
    C = {
        'X_dirpath' : '../../../data/onetext_train_small/*',
        'X_valid_dirpath' : '../../../data/onetext_valid_small/*',
        'y_dirpath' : '../../../data/train_cleaned/',
        'batchsize' : 2000000,
        'limit' : None,
        'epochs' : 15,
        'patience' : 70000,
        'patience_increase' : 2,
        'improvement_threshold' : 0.995,
        'validation_frequency' : 2,
        'lmbda' : 0.0,
        'dropout' : 0.0,
        'training_size' : None,
        'validation_size' : None,
        'algorithm' : 'RMSProp',
        'eta' : float(params['eta'][0]),
        'eta_min': float(params['eta_min'][0]),
        'eta_pre' : float(params['eta_pre'][0]),
        'corruption_level' : float(params['corruption_level'][0]),
        'border' : 2,
        'hidden' : int(params['hidden'][0]),
        'mini_batch_size': 500
    }

    training_data = BatchImgProcessor(
        X_dirpath=C['X_dirpath'],
        y_dirpath=C['y_dirpath'],
        batchsize=C['batchsize'],
        border=C['border'],
        limit=C['limit'],
        random=True,
        random_mode='fully',
        dtype=theano.config.floatX,
        rnd=rnd, modus='full')

    validation_data = BatchImgProcessor(
        X_dirpath=C['X_valid_dirpath'],
        y_dirpath=C['y_dirpath'],
        batchsize=C['batchsize'],
        border=C['border'],
        limit=C['limit'],
        random=False,
        dtype=theano.config.floatX,
        rnd=rnd, modus='full')

    pretrain_data = BatchImgProcessor(
        X_dirpath='../../../data/onetext_pretrain_small/*',
        y_dirpath='../../../data/train_cleaned/',
        batchsize=50000, border=border, limit=None,
        random=True, modus='full', random_mode='fully', rnd=rnd,
        dtype=theano.config.floatX)

    C['training_size'] = training_data.actual_full_length()
    C['validation_size'] = validation_data.actual_full_length()
    print "Training size: %d" % C['training_size']
    print "Validation size: %d" % C['validation_size']

    metric_recorder.add_experiment_metainfo(constants=C)
    metric_recorder.start()

    n_in = (2*border+1)**2
    net = Network([
        AutoencoderLayer(n_in=n_in, n_hidden=C['hidden'], rnd=rnd,
          corruption_level=C['corruption_level']),
        FullyConnectedLayer(n_in=C['hidden'], n_out=1, rnd=rnd)],
        C['mini_batch_size'])

    print '...start pretraining'
    net.pretrain_autoencoders(training_data=pretrain_data,
        mbs=C['mini_batch_size'], eta=C['eta_pre'], epochs=15, metric_recorder=metric_recorder)

    result = net.train(tdata=training_data, epochs=C['epochs'],
                     mbs=C['mini_batch_size'], eta=C['eta'],
                     eta_min=C['eta_min'],
                     vdata=validation_data, lmbda=C['lmbda'],
                     momentum=None,
                     patience_increase=C['patience_increase'],
                     improvement_threshold=C['improvement_threshold'],
                     validation_frequency=C['validation_frequency'],
                     metric_recorder=metric_recorder,
                     save_dir='./models/%d_' % metric_recorder.job_id,
                     early_stoping=False)

    print 'Time = %f' % metric_recorder.stop()
    print 'Result = %f' % result
    return float(result)

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d'
    print params
    return train(job_id, params)

if __name__ == '__main__':
    main(1, {'l2' : [0.01], 'dropout' : [0.01],
                'hidden': [80], 'eta' : [0.04], 'eta_min':[0.001]})
