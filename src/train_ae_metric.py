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

mr = MetricRecorder(config_dir_path='./sae.json')
mr.start()

border = 2

BA1 = BatchImgProcessor.load(
    X_dirpath='../../data/train_cleaned/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=5000,
    border=border,
    limit=1,
    train_stepover=8,
    dtype=theano.config.floatX)
pretrain_data = BA1(modus='train', random=True)

BatchProcessor = BatchImgProcessor.load(
    X_dirpath='../../data/train/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=5000,
    border=border,
    limit=1,
    train_stepover=8,
    dtype=theano.config.floatX)
training_data = BatchProcessor(modus='train', random=True)
validation_data = BatchProcessor(modus='valid')
print "Training size: %d" % len(training_data)
print "Validation size: %d" % len(validation_data)

n_in = (2*border+1)**2

mini_batch_size = 200

net = Network([
        AutoencoderLayer(n_in=n_in, n_hidden=80, corruption_level=0.2),
        AutoencoderLayer(n_in=80, n_hidden=50, corruption_level=0.4),
        FullyConnectedLayer(n_in=50, n_out=1),
    ], mini_batch_size)

# image = PIL.Image.fromarray(tile_raster_images(
        # X=net.layers[0].w.get_value(borrow=True).T,
        # img_shape=(5, 5), tile_shape=(10, 10),
        # tile_spacing=(2, 2)))
# image.show()

save_dir = "./model/%s_%d_" % (mr.experiment_name, mr.job_id)
print "Savedir : " + save_dir

print '...start pretraining'
net.pretrain_autoencoders(
    training_data=pretrain_data,
    batch_size=mini_batch_size,
    eta=0.01, epochs=2,
    metric_recorder=mr, save_dir=save_dir)

training_data.reset()
print '...start training'
net.SGD(training_data=training_data, epochs=2,
        batch_size=mini_batch_size, eta=0.01,
        validation_data=validation_data, lmbda=0.0,
        momentum=None, patience=20000, patience_increase=2,
        improvement_threshold=0.995, validation_frequency=2,
        save_dir=save_dir, metric_recorder=mr)

print "Zeit : %d" % mr.stop()
