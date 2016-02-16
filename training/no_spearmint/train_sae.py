# coding: utf-8

import numpy as np
import PIL.Image
import theano
from timeit import default_timer as timer
#from utils import tile_raster_images

# add own libs to path
lib_path = os.path.abspath(os.path.join('../../src'))
sys.path.append(lib_path)

# own libs
from network import Network, FullyConnectedLayer, AutoencoderLayer
from preprocessor import BatchProcessor
from metric import MetricRecorder

border = 2
mr = MetricRecorder(config_dir_path='./sea.json')

pretrain_data = BatchProcessor(
    X_dirpath='../../../data/train_cleaned/*',
    y_dirpath='../../../data/train_cleaned/',
    batchsize=500000,
    border=border,
    limit=30,
    random=True, random_mode='fully',
    dtype=theano.config.floatX)

training_data = BatchProcessor(
    X_dirpath='../../../data/train/*',
    y_dirpath='../../../data/train_cleaned/',
    batchsize=500000,
    border=border,
    random=True, random_mode='fully',
    dtype=theano.config.floatX)

validation_data = BatchProcessor(
    X_dirpath='../../../data/valid/*',
    y_dirpath='../../../data/train_cleaned/',
    batchsize=500000,
    border=border,
    dtype=theano.config.floatX)

print "Pretrain size: %d" % len(pretrain_data)
print "Training size: %d" % len(training_data)
print "Validation size: %d" % len(validation_data)

n_in = (2*border+1)**2

mr.start()
mbs = 500

net = Network([
        AutoencoderLayer(n_in=n_in, n_hidden=80, corruption_level=0.2),
        AutoencoderLayer(n_in=80, n_hidden=50, corruption_level=0.2),
        AutoencoderLayer(n_in=50, n_hidden=20, corruption_level=0.2),
        FullyConnectedLayer(n_in=20, n_out=1),
    ], mbs)

print '...start pretraining'
net.pretrain_autoencoders(tdata=pretrain_data,
                          mbs=mbs, metric_recorder=mr,
                          save_dir='./models/sea_test_pretrain_',
                          eta=0.01, epochs=10)

#image = PIL.Image.fromarray(tile_raster_images(
#        X=net.layers[0].w.get_value(borrow=True).T,
#        img_shape=(5, 5), tile_shape=(10, 10),
#        tile_spacing=(2, 2)))
#image.show()

training_data.reset()
print '...start training'
cost = net.SGD(tdata=training_data, epochs=10,
        mbs=mbs, eta=0.025, eta_min=0.01,
        vdata=validation_data, lmbda=0.0,
        momentum=None, patience_increase=2,
        improvement_threshold=0.995, validation_frequency=2,
        save_dir='./models/sae_test_', algorithm='rmsprop',
        metric_recorder=mr)

print "Kosten: %f" % float(cost)
print "Zeit : %d" % mr.stop()
