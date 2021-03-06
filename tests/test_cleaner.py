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

from cleaner import Cleaner, BatchCleaner
import config

class TestCleaner(unittest.TestCase):

    @unittest.skipUnless(config.slow, 'slow test')
    def test_clean_and_save(self):
        bCleaner = BatchCleaner(dirty_dir=config.data_dir_path + 'test/',
           model_path='./tests/data/models/ae3_213750_model.pkl')
        outpath = config.data_dir_path + 'test_cleaned'
        bCleaner.clean_and_save(output_dir=outpath)
        pass

    @unittest.skipUnless(config.slow, 'slow test')
    def test_clean_for_submission(self):
        bCleaner = BatchCleaner(dirty_dir=config.data_dir_path + 'test/',
           model_path='./tests/data/models/ae3_213750_model.pkl')
        outpath = config.data_dir_path + 'test_cleaned'
        bCleaner.clean_for_submission(output_dir=outpath)
        pass

class TestCleaner(unittest.TestCase):

    # @unittest.skipUnless(config.slow, 'slow test')
    def test_clean(self):
        img = config.data_dir_path + 'test/10.png'
        PIL.Image.open(img).show()

        e = Cleaner('./tests/data/models/13_72000_model.pkl')
        e.clean_and_show(img)

        e = Cleaner('./tests/data/models/ae3_213750_model.pkl')
        e.clean_and_show(img)
        pass

    def test_to_submission_format(self):
        e = Cleaner('./tests/data/models/ae3_213750_model.pkl')
        img = '../data/test/10.png'
        img, id = e.clean(img)
        csv = e.to_submission_format(img, id)
        row = csv[300].split(',')
        self.assertEqual(row[0], '%s_%d_%d' % (id, 1, 301))
        self.assertTrue(float(row[1]) <= 1.0)

    def test_metadata(self):
        e = Cleaner('./tests/data/models/meta_2200_model.pkl')
        e2 = Cleaner('./tests/data/models/72000_model.pkl')
        metadata = e.metadata()
        no_metadata = e2.metadata()
        self.assertTrue(metadata)
        self.assertFalse(no_metadata)

