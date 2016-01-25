
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

class TestNetwork(unittest.TestCase):

    def setUp(self):
        border = 2
        self.n_in = (2*border+1)**2
        BA1 = BatchImgProcessor.load(
          X_dirpath='../data/train_cleaned/*',
          y_dirpath='../data/train_cleaned/',
          batchsize=5000,
          border=border,
          limit=1,
          train_stepover=8,
          dtype=theano.config.floatX)
        self.pretrain_data = BA1(modus='full', random=True)

    def test_pretrain(self):
        n_in = self.n_in
        mini_batch_size = 200
        net = Network([
          AutoencoderLayer(n_in=n_in, n_hidden=n_in-3),
          AutoencoderLayer(n_in=n_in-3, n_hidden=n_in-6),
          AutoencoderLayer(n_in=n_in-6, n_hidden=n_in-10),
          FullyConnectedLayer(n_in=n_in-10, n_out=1),
        ], mini_batch_size)

        # net.pretrain_autoencoders(
            # training_data=self.pretrain_data,
            # batch_size=mini_batch_size,
            # eta=0.025, epochs=10)

        pass
