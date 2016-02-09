# coding: utf-8

import numpy as np
import load_data as l

import PIL.Image
import cPickle
import pdb
import theano
from timeit import default_timer as timer

from network import Network, FullyConnectedLayer, tanh, ReLU, AutoencoderLayer
from preprocessor import BatchImgProcessor
from metric import MetricRecorder

rnd = np.random.RandomState()

mr = MetricRecorder(config_dir_path='./simple.json')
mr.start()

border = 2

training_data = BatchImgProcessor(
    X_dirpath='../../data/onetext_train_small/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=500000, border=border, limit=None,
    random=True, random_mode='fully', modus='full',
    dtype=theano.config.floatX, rnd=rnd)

validation_data = BatchImgProcessor(
    X_dirpath='../../data/onetext_valid_small/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=500000, border=border, limit=None,
    random=False, modus='full', rnd=rnd,
    dtype=theano.config.floatX)

pretrain_data = BatchImgProcessor(
    X_dirpath='../../data/onetext_pretrain_small/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=50000, border=border, limit=None,
    random=True, modus='full', random_mode='fully', rnd=rnd,
    dtype=theano.config.floatX)

print "Job ID: %d" % mr.job_id
save_dir = "./model/%s_%d_" % (mr.experiment_name, mr.job_id)
print "Save Dir: " + save_dir
print len(pretrain_data)

start = timer()
mbs = 500
net = Network([
        AutoencoderLayer(n_in=(2*border+1)**2, n_hidden=21, 
        rnd=rnd, corruption_level=0.4),
        FullyConnectedLayer(n_in=21, n_out=1, rnd=rnd)
    ], mbs)

print '...start pretraining'
net.pretrain_autoencoders(training_data=pretrain_data,
                    batch_size=mbs, eta=0.01, epochs=4)

print '...start training'
cost = net.SGD(training_data=training_data, epochs=4,
        batch_size=mbs, eta=0.1, eta_min=0.01,
        validation_data=validation_data, lmbda=0.0,
        momentum=0.95, patience_increase=2, 
        improvement_threshold=0.995, validation_frequency=20,
        save_dir=save_dir, metric_recorder=mr,
        algorithm='rmsprop', early_stoping=False)
print "Zeit : %d" % mr.stop()
