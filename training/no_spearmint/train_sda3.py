import sys, os
import numpy as np
import PIL.Image
import cPickle
import theano
from timeit import default_timer as timer

lib_path = os.path.abspath(os.path.join('..', '..', 'src'))
sys.path.append(lib_path)

from network import Network, FullyConnectedLayer, AutoencoderLayer, ReLU
from preprocessor import BatchProcessor
from metric import MetricRecorder

rnd = np.random.RandomState()

mr = MetricRecorder(config_dir_path='./sae.json')
C = {
    'X_dirpath' : '../../../data/onetext_train/*',
    'X_valid_dirpath' : '../../../data/onetext_valid/*',
    'X_pretrain_dirpath' : '../../../data/onetext_pretrain/*',
    'y_dirpath' : '../../../data/train_cleaned/',
    'batchsize' : 1000000,
    'limit' : None,
    'epochs' : 15,
    'patience' : 70000,
    'patience_increase' : 2,
    'improvement_threshold' : 0.995,
    'validation_frequency' : 1,
    'lmbda' : 0.0,
    'dropout' : 0.001,
    'training_size' : None,
    'validation_size' : None,
    'pretrain_size' : None,
    'algorithm' : 'rmsprop',
    'eta' : 0.045,
    'eta_min': 0.01,
    'eta_pre' : 0.025,
    'corruption_level' : 0.14,
    'border' : 2,
    'hidden_1' : 199,
    'hidden_2' : 81,
    'hidden_3' : 70,
    'mini_batch_size': 500
}

training_data = BatchProcessor(
    X_dirpath=C['X_dirpath'], y_dirpath=C['y_dirpath'],
    batchsize=C['batchsize'], border=C['border'],
    limit=C['limit'], random=True, random_mode='fully',
    dtype=theano.config.floatX,
    rnd=rnd)

validation_data = BatchProcessor(
    X_dirpath=C['X_valid_dirpath'], y_dirpath=C['y_dirpath'],
    batchsize=C['batchsize'], border=C['border'],
    limit=C['limit'], random=False,
    dtype=theano.config.floatX, rnd=rnd)

pretrain_data = BatchProcessor(
    X_dirpath=C['X_pretrain_dirpath'],
    y_dirpath=C['y_dirpath'],
    batchsize=500000, border=C['border'], limit=None,
    random=True, random_mode='fully', rnd=rnd,
    dtype=theano.config.floatX)

C['training_size'] = training_data.size()
C['validation_size'] = validation_data.size()
C['pretrain_size'] = pretrain_data.size()
print "Training size: %d" % C['training_size']
print "Validation size: %d" % C['validation_size']
print "Pretrain size: %d" % C['pretrain_size']

mr.add_experiment_metainfo(constants=C)
mr.start()

save_dir = "./models/%s_%d_" % (mr.experiment_name, mr.job_id)
pretrain_save_dir = save_dir + "pretrain_"

n_in = (2*C['border']+1)**2
net = Network([
    AutoencoderLayer(n_in=n_in, n_hidden=C['hidden_1'], rnd=rnd,
      corruption_level=C['corruption_level']),
    AutoencoderLayer(n_in=C['hidden_1'], n_hidden=C['hidden_2'], rnd=rnd,
      corruption_level=C['corruption_level'], p_dropout=C['dropout']),
    AutoencoderLayer(n_in=C['hidden_2'], n_hidden=C['hidden_3'], rnd=rnd,
      corruption_level=C['corruption_level']),
    FullyConnectedLayer(n_in=C['hidden_3'], n_out=1, rnd=rnd)],
    C['mini_batch_size'])

print '...start pretraining'
net.pretrain_autoencoders(tdata=pretrain_data,
                          mbs=C['mini_batch_size'], eta=C['eta_pre'], 
                          epochs=10, metric_recorder=mr, 
                          save_dir=pretrain_save_dir)

print '...start training'
result = net.train(tdata=training_data, epochs=C['epochs'],
                 mbs=C['mini_batch_size'], eta=C['eta'],
                 eta_min=C['eta_min'],
                 vdata=validation_data, lmbda=C['lmbda'],
                 momentum=None,
                 patience_increase=C['patience_increase'],
                 improvement_threshold=C['improvement_threshold'],
                 validation_frequency=C['validation_frequency'],
                 metric_recorder=mr, save_dir=save_dir,
                 early_stoping=False)

print 'Time = %f' % mr.stop()
print 'Result = %f' % result
