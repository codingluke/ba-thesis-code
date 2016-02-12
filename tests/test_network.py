
import unittest
import numpy as np
import PIL.Image
import pdb
import os
from timeit import default_timer as timer
from itertools import izip
import sys
import theano

lib_path = os.path.abspath(os.path.join('src'))
sys.path.append(lib_path)

from network import Network, FullyConnectedLayer, AutoencoderLayer
from preprocessor import BatchImgProcessor
import config

rnd = np.random.RandomState(2348923)

class TestNetwork(unittest.TestCase):

    def setUp(self):
        border = 2
        self.n_in = (2*border+1)**2
        self.pretrain = BatchImgProcessor(
          X_dirpath=config.data_dir_path + 'train_cleaned/*',
          y_dirpath=config.data_dir_path + 'train_cleaned/',
          batchsize=5000,
          border=border,
          limit=1,
          train_stepover=8,
          dtype=theano.config.floatX,
          modus='full', random=True,
          rnd=rnd)

    def test_to_string(self):
        net = Network([
          AutoencoderLayer(n_in=25, n_hidden=22, rnd=rnd),
          AutoencoderLayer(n_in=22, n_hidden=19,
                           corruption_level=0.1, p_dropout=0.1, rnd=rnd),
          FullyConnectedLayer(n_in=19, n_out=1, rnd=rnd),
        ], 200)
        layers = net.get_layer_string()
        dropouts = net.get_layer_dropout_string()
        test_s = "Ae[sgm](25, 22)-dAe[sgm, 0.100](22, 19)-FC(19, 1)"
        self.assertEqual(layers, test_s)

    # @unittest.skipUnless(config.slow, 'slow test')
    def test_train(self):
        border = 2
        mbs = 500
        n_in = (2*border+1)**2
        tdata = BatchImgProcessor(
          X_dirpath=config.data_dir_path + 'train/*',
          y_dirpath=config.data_dir_path + 'train_cleaned/',
          batchsize=5000, border=border,
          limit=1, dtype=theano.config.floatX,
          modus='full', random=True, random_mode='fully',
          rnd=rnd)
        vdata = BatchImgProcessor(
          X_dirpath=config.data_dir_path + 'train/*',
          y_dirpath=config.data_dir_path + 'train_cleaned/',
          batchsize=5000, border=border,
          limit=1, dtype=theano.config.floatX,
          modus='full', random=False, rnd=rnd)
        net = Network([
                FullyConnectedLayer(n_in=25, n_out=19, rnd=rnd),
                FullyConnectedLayer(n_in=19, n_out=1, rnd=rnd),
              ], mbs)
        cost = net.train(tdata=tdata, epochs=1, mbs=mbs, eta=0.1,
                         eta_min=0.01, vdata=vdata, lmbda=0.0,
                         momentum=0.95, patience_increase=2,
                         improvement_threshold=0.995,
                         validation_frequency=1, algorithm='rmsprop',
                         early_stoping=False)
        self.assertTrue(float(cost) < 1.0)

    # @unittest.skipUnless(config.slow, 'slow test')
    def test_pretrain(self):
        n_in = self.n_in
        mini_batch_size = 200
        net = Network([
          AutoencoderLayer(n_in=n_in, n_hidden=n_in-3, rnd=rnd),
          AutoencoderLayer(n_in=n_in-3, n_hidden=n_in-6,
                           corruption_level=0.1, rnd=rnd),
          FullyConnectedLayer(n_in=n_in-6, n_out=1, rnd=rnd),
        ], mini_batch_size)

        net.pretrain_autoencoders(
            tdata=self.pretrain,
            mbs=mini_batch_size,
            eta=0.025, epochs=1)
        pass
