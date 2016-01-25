# coding: utf-8

import numpy as np
import load_data as l
import PIL.Image
import cPickle
import pdb
import theano
#from utils import tile_raster_images
from timeit import default_timer as timer

from network import Network, FullyConnectedLayer, \
                    tanh, ReLU, AutoencoderLayer
from preprocessor import BatchImgProcessor
from engine import CleaningEngine

border = 3

BA1 = BatchImgProcessor.load(
    X_dirpath='../../data/train_cleaned/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=500000,
    border=border,
    limit=30,
    train_stepover=8,
    dtype=theano.config.floatX)
pretrain_data = BA1(modus='train', random=True)

BatchProcessor = BatchImgProcessor.load(
    X_dirpath='../../data/train/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=500000,
    border=border,
    limit=30,
    train_stepover=8,
    dtype=theano.config.floatX)
training_data = BatchProcessor(modus='train', random=True)
validation_data = BatchProcessor(modus='valid')
print "Training size: %d" % len(training_data)
print "Validation size: %d" % len(validation_data)

n_in = (2*border+1)**2

start = timer()
mini_batch_size = 200

net = Network([
        AutoencoderLayer(n_in=n_in, n_hidden=n_in-5),
        AutoencoderLayer(n_in=n_in-5, n_hidden=n_in-10),
        FullyConnectedLayer(n_in=n_in-10, n_out=60),
        FullyConnectedLayer(n_in=60, n_out=1),
    ], mini_batch_size)

net.pretrain_autoencoders(training_data=pretrain_data,
                    batch_size=mini_batch_size,
                    eta=0.025, epochs=10)

#image = PIL.Image.fromarray(tile_raster_images(
#        X=net.layers[0].w.get_value(borrow=True).T,
#        img_shape=(5, 5), tile_shape=(10, 10),
#        tile_spacing=(2, 2)))
#image.show()

training_data.reset()
print '...start training'
net.SGD(training_data=training_data, epochs=10,
        mini_batch_size=mini_batch_size, eta=0.01,
        validation_data=validation_data, lmbda=0.0,
        momentum=None, patience=20000, patience_increase=2,
        improvement_threshold=0.995, validation_frequency=2,
        save_dir='./model/ae4_')
end = timer()
print "Zeit : %d" % (end-start)
