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
        img = '../data/test/130.png'
        PIL.Image.open(img).show()

        e = engine.CleaningEngine('./tests/data/models/72000_model.pkl')
        e.clean_and_show(img)

        e = engine.CleaningEngine('./tests/data/models/162000_model.pkl')
        e.clean_and_show(img)

        e = engine.CleaningEngine('./tests/data/models/342000_model.pkl')
        e.clean_and_show(img)

    def test_metadata(self):
        e = engine.CleaningEngine('./tests/data/models/meta_2200_model.pkl')
        e2 = engine.CleaningEngine('./tests/data/models/72000_model.pkl')
        metadata = e.metadata()
        no_metadata = e2.metadata()
        self.assertTrue(metadata)
        self.assertFalse(no_metadata)

