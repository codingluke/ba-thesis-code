# coding: utf-8

import numpy as np
import load_data as l
import PIL.Image
import cPickle
import pdb
import theano
from timeit import default_timer as timer

from network import Network, FullyConnectedLayer, tanh, ReLU
from preprocessor import BatchImgProcessor

border = 2

BatchProcessor = BatchImgProcessor.load(
    X_dirpath='../../data/train/*',
    y_dirpath='../../data/train_cleaned/',
    batchsize=50000,
    border=border,
    limit=5,
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
        FullyConnectedLayer(n_in=n_in, n_out=200, activation_fn=ReLU,
          p_dropout=0.1),
        FullyConnectedLayer(n_in=200, n_out=100, activation_fn=ReLU,
          p_dropout=0.1),
        FullyConnectedLayer(n_in=100, n_out=50, activation_fn=ReLU,
          p_dropout=0.1),
        FullyConnectedLayer(n_in=50, n_out=1, p_dropout=0.1)
    ], mini_batch_size)

print '...start training'
net.SGD(training_data=training_data, epochs=3,
        batch_size=mini_batch_size, eta=0.025,
        validation_data=validation_data, lmbda=0.0,
        momentum=0.95, patience=20000, patience_increase=2,
        improvement_threshold=0.995, validation_frequency=2,
        save_dir='./model/meta_')
end = timer()
print "Zeit : %d" % (end-start)

f = open('model_b4_l144_bs20000000.pkl', 'wb')
cPickle.dump(net, f)
f = open('model_b4_l144_bs20000000.pkl', 'rb')
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
