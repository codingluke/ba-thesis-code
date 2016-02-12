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


border = 2

training_data = BatchImgProcessor(
    X_dirpath='../../data/onetext_train_small/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=2000000, border=border, limit=None,
    random=True, random_mode='fully', modus='full',
    dtype=theano.config.floatX, rnd=rnd)

validation_data = BatchImgProcessor(
    X_dirpath='../../data/onetext_valid_small/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=2000000, border=border, limit=None,
    random=False, modus='full', rnd=rnd,
    dtype=theano.config.floatX)

pretrain_data = BatchImgProcessor(
    X_dirpath='../../data/onetext_pretrain_small/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=50000, border=border, limit=None,
    random=True, modus='full', random_mode='fully', rnd=rnd,
    dtype=theano.config.floatX)

#for cl in [0.1, 0.2, 0.3, 0.4, 0.5]:
for cl in [0.2]:
    mr = MetricRecorder(config_dir_path='./simple.json')
    mr.start()
    print "Job ID: %d" % mr.job_id
    save_dir = "./model/%s_%d_" % (mr.experiment_name, mr.job_id)
    print "Save Dir: " + save_dir
    print len(pretrain_data)

    start = timer()
    mbs = 500
    net = Network([
            AutoencoderLayer(n_in=(2*border+1)**2, n_hidden=190,
            rnd=rnd, corruption_level=cl),
            FullyConnectedLayer(n_in=190, n_out=1, rnd=rnd)
        ], mbs)

    print '...start pretraining'
    #net.pretrain_autoencoders(tdata=pretrain_data,
    #                    mbs=mbs, eta=0.01, epochs=15)

    print '...start training'
    cost = net.train(tdata=training_data, epochs=15,
            mbs=mbs, eta=0.045, eta_min=0.01,
            vdata=validation_data, lmbda=0.0,
            momentum=0.95, patience_increase=2,
            improvement_threshold=0.995, validation_frequency=1,
            save_dir=save_dir, metric_recorder=mr,
            algorithm='rmsprop', early_stoping=False)
    print "Zeit : %d" % mr.stop()
