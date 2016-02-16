# coding: utf-8

# default libs
import numpy as np
import theano
import sys, os
from timeit import default_timer as timer

# add own libs to path
lib_path = os.path.abspath(os.path.join('../../src'))
sys.path.append(lib_path)

# own libs
from network import Network, FullyConnectedLayer
from preprocessor import BatchImgProcessor
from metric import MetricRecorder

rnd = np.random.RandomState()

mr = MetricRecorder(config_dir_path='./simple.json')
mr.start()

border = 2

training_data = BatchImgProcessor(
    X_dirpath='../../../data/onetext_train_small/*',
    y_dirpath='../../../data/train_cleaned/',
    batchsize=2000000, border=border, limit=None,
    random=True, random_mode='fully',
    dtype=theano.config.floatX, rnd=rnd)

validation_data = BatchImgProcessor(
    X_dirpath='../../../data/onetext_valid_small/*',
    y_dirpath='../../../data/train_cleaned/',
    batchsize=2000000, border=border, limit=None,
    random=False, rnd=rnd,
    dtype=theano.config.floatX)

print "Job ID: %d" % mr.job_id
save_dir = "./models/%s_%d_" % (mr.experiment_name, mr.job_id)
print "Save Dir: " + save_dir

start = timer()
mbs = 500
net = Network([
        FullyConnectedLayer(n_in=(2*border+1)**2, n_out=80, rnd=rnd),
        FullyConnectedLayer(n_in=80, n_out=1, rnd=rnd)
    ], mbs)

print '...start training'
cost = net.train(tdata=training_data, epochs=4,
        mbs=mbs, eta=0.1, eta_min=0.01,
        vdata=validation_data, lmbda=0.0,
        momentum=0.95, patience_increase=2,
        improvement_threshold=0.995, validation_frequency=3,
        save_dir=save_dir, metric_recorder=mr,
        algorithm='rmsprop', early_stoping=False)
print "Zeit : %d" % mr.stop()
