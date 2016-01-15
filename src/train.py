# coding: utf-8

import numpy as np
import load_data as l
import PIL.Image
import cPickle
import pdb
import theano

from network import Network, ConvPoolLayer, FullyConnectedLayer, \
                    tanh, ReLU
from preprocessor import BatchImgProcessor

border = 3

#training_data = (l.DataIter('./data/dump/b5_c500000_l50_train_x*'),
                 #l.DataIter('./data/dump/b5_c500000_l50_train_y*'))
#validation_data = (l.DataIter('./data/dump/b5_c500000_l50_valid_x*'),
                   #l.DataIter('./data/dump/b5_c500000_l50_valid_y*'))

#training_data = l.DataIter(x_path='./data/dump/b5_c500000_l50_train_x*',
                           #y_path='./data/dump/b5_c500000_l50_train_y*')
#validation_data = l.DataIter(x_path='./data/dump/b5_c500000_l50_valid_x*',
                             #y_path='./data/dump/b5_c500000_l50_valid_y*')

# training_data = l.DataIter(x_path='./data/dump/b3b3_c1000000_l144_train_x*',
                           # y_path='./data/dump/b3b3_c1000000_l144_train_y*')
# validation_data = l.DataIter(x_path='./data/dump/b3b3_c1000000_l144_valid_x*',
                             # y_path='./data/dump/b3b3_c1000000_l144_valid_y*')

BatchProcessor = BatchImgProcessor.load(
    X_dirpath='../../data/train/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=100000,
    border=border,
    limit=None,
    train_stepover=8,
    dtype=theano.config.floatX)
training_data = BatchProcessor(modus='train', random=True)
validation_data = BatchProcessor(modus='valid')

#print training_data[0].files
#n_in = 121
n_in = 49

mini_batch_size = 500
net = Network([
#        ConvPoolLayer(image_shape=(mini_batch_size, 1, 258, 540),
#                      filter_shape=(20, 1, 5, 5),
#                      poolsize=(2, 2),
#                      activation_fn=ReLU),
#        ConvPoolLayer(image_shape=(mini_batch_size, 20, 127, 268),
#                      filter_shape=(40, 20, 5, 5),
#                      poolsize=(2, 2),
#                      activation_fn=ReLU),
        #FullyConnectedLayer(n_in=40*61*132, n_out=10000, activation_fn=ReLU),
        #FullyConnectedLayer(n_in=n_in, n_out=100, activation_fn=tanh),
        FullyConnectedLayer(n_in=n_in, n_out=100),
        FullyConnectedLayer(n_in=100, n_out=1),
        #SoftmaxLayer(n_in=100, n_out=1)
        #FullyConnectedLayer(n_in=100, n_out=1, activation_fn=tanh)
    ], mini_batch_size)

#print '...start training'
net.SGD(training_data, 100, mini_batch_size, 0.025,
        validation_data, test_data=None, lmbda=0.0)

f = open('model_b3_l144.pkl', 'wb')
cPickle.dump(net, f)
f = open('model_b3_l144.pkl', 'rb')
net = cPickle.load(f)

test_x = np.array(l.x_from_image('../../data/test/1.png', border=border))
y = net.predict(test_x)
y2 = np.vstack(np.array(y).flatten())
orig = np.resize(y2, (258, 540)) * 255
img = PIL.Image.fromarray(orig)
img.show()
img.convert('L').save('1_cleaned.png', 'PNG')

test_x = np.array(l.x_from_image('../../data/test/4.png', border=border))
y = net.predict(test_x)
y2 = np.vstack(np.array(y).flatten())
orig = np.resize(y2, (258, 540)) * 255
img = PIL.Image.fromarray(orig)
img.show()
img.convert('L').save('4_cleaned.png', 'PNG')

test_x = np.array(l.x_from_image('../../data/test/7.png', border=border))
y = net.predict(test_x)
y2 = np.vstack(np.array(y).flatten())
orig = np.resize(y2, (258, 540)) * 255
img = PIL.Image.fromarray(orig)
img.show()
img.convert('L').save('7_cleaned.png', 'PNG')

test_x = np.array(l.x_from_image('../../data/test/10.png', border=border))
y = net.predict(test_x)
y2 = np.vstack(np.array(y).flatten())
orig = np.resize(y2, (258, 540)) * 255
img = PIL.Image.fromarray(orig)
img.show()
img.convert('L').save('10_cleaned.png', 'PNG')
