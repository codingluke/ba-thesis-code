# coding: utf-8

import numpy as np
import load_data as l
import PIL.Image
import cPickle
import pdb
import theano
from utils import tile_raster_images
from timeit import default_timer as timer

from network import Network, FullyConnectedLayer, \
                    tanh, ReLU, AutoencoderLayer
from preprocessor import BatchImgProcessor
from engine import Cleaner
from metric import MetricRecorder

rnd = np.random.RandomState()

mr = MetricRecorder(config_dir_path='./sae.json')
mr.start()

border = 2

pretrain_data = BatchImgProcessor(
    X_dirpath='../../data/onetext_pretrain_small/*',
    y_dirpath='../../data/onetext_pretrain_small/',
    batchsize=500000,
    border=border,
    limit=None,
    dtype=theano.config.floatX,
    random_mode='fully',
    random=True, modus='full', rnd=rnd)

training_data = BatchImgProcessor(
    X_dirpath='../../data/onetext_train_small/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=1000000,
    border=border,
    limit=None,
    dtype=theano.config.floatX,
    random_mode='fully',
    modus='full', random=True, rnd=rnd)

validation_data = BatchImgProcessor(
    X_dirpath='../../data/onetext_valid_small/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=1000000,
    border=border,
    limit=None,
    dtype=theano.config.floatX,
    modus='full', random=False, rnd=rnd)

# training_data = BA2(modus='full', random=True)
# validation_data = BA3(modus='full', random=False)

print "Pretrain size: %d" % len(training_data)
print "Training size: %d" % len(training_data)
print "Validation size: %d" % len(validation_data)

n_in = (2*border+1)**2

mbs = 500

net = Network([
        AutoencoderLayer(n_in=n_in, n_hidden=190, corruption_level=0.14,
          rnd=rnd),
        AutoencoderLayer(n_in=190, n_hidden=81, corruption_level=0.14,
          rnd=rnd),
        FullyConnectedLayer(n_in=81, n_out=1, rnd=rnd),
    ], mbs)

# image = PIL.Image.fromarray(tile_raster_images(
        # X=net.layers[0].w.get_value(borrow=True).T,
        # img_shape=(5, 5), tile_shape=(10, 10),
        # tile_spacing=(2, 2)))
# image.show()

save_dir = "./model/%s_%d_" % (mr.experiment_name, mr.job_id)
print "Savedir : " + save_dir

print '...start pretraining'
#net.pretrain_autoencoders(
#    tdata=pretrain_data,
#    mbs=mbs,
#    eta=0.03, epochs=15,
#    metric_recorder=mr, save_dir=save_dir)

training_data.reset()
print '...start training'
net.train(tdata=training_data, epochs=15,
        mbs=mbs, eta=0.045, eta_min=0.01,
        vdata=validation_data, lmbda=0.0,
        momentum=0.0, patience_increase=2,
        improvement_threshold=0.995, validation_frequency=1,
        save_dir=save_dir, metric_recorder=mr, algorithm='rmsprop')

print "Zeit : %d" % mr.stop()
