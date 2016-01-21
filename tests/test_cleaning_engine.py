import unittest
import numpy as np
import PIL.Image
import pdb
import os
from timeit import default_timer as timer
from itertools import izip
import sys

lib_path = os.path.abspath(os.path.join('src'))
sys.path.append(lib_path)

import engine
import config

class TestCleaningEngine(unittest.TestCase):

  def test_clean(self):
      PIL.Image.open('../data/test/10.png').show()
      e = engine.CleaningEngine('./src/model/11000_model.pkl', border=2)
      e.clean_and_show('../data/test/10.png')
