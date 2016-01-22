# coding: utf-8

import numpy as np
import load_data as l
import PIL.Image
import cPickle
import pdb
import theano
from utils import tile_raster_images
from timeit import default_timer as timer

from network import Network, ConvPoolLayer, FullyConnectedLayer, \
                    tanh, ReLU, AutoencoderLayer
from preprocessor import BatchImgProcessor

border = 5

BatchProcessor = BatchImgProcessor.load(
    X_dirpath='../../data/train/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=5000,
    border=border,
    limit=15,
    train_stepover=8,
    dtype=theano.config.floatX)
training_data = BatchProcessor(modus='train', random=True)
validation_data = BatchProcessor(modus='valid')
print "Training size: %d" % len(training_data)
print "Validation size: %d" % len(validation_data)

n_in = (2*border+1)**2

start = timer()
mini_batch_size = 500

ae = AutoencoderLayer(n_in=n_in, n_hidden=n_in-10)
ae.train(training_data=training_data, batch_size=200, eta=0.25, epochs=1)

image = PIL.Image.fromarray(tile_raster_images(
        X=ae.w.get_value(borrow=True).T,
        img_shape=(11, 11), tile_shape=(100, 100),
        tile_spacing=(1, 1)))
image.show()


# net = Network([
        # FullyConnectedLayer(n_in=n_in, n_out=80),
        # FullyConnectedLayer(n_in=80, n_out=1),
    # ], mini_batch_size)

# print '...start training'
# net.SGD(training_data=training_data, epochs=100,
        # mini_batch_size=mini_batch_size, eta=0.01,
        # validation_data=validation_data, lmbda=0.0,
        # momentum=None, patience=20000, patience_increase=2,
        # improvement_threshold=0.995, validation_frequency=2,
        # save_dir='./model/meta_')
# end = timer()
# print "Zeit : %d" % (end-start)
