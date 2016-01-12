import unittest
import numpy as np
import PIL.Image
import pdb
import os
from timeit import default_timer as timer

from src.preprocessor import ImgPreprocessor, BatchImgProcessor
from src.load_data import DataIter

class TestBatchImgProcessor(unittest.TestCase):

  def setUp(self):
    self.batch_processor = BatchImgProcessor(X_dirpath='../data/train/*',
                                        y_dirpath='../data/train_cleaned/',
                                        batchsize=1000000,
                                        border=3,
                                        limit=None,
                                        train_size=0.8)

  def test_iter2(self):
    # start = timer()
    # data = DataIter(x_path='../preparation/tryone/data/dump/b3b3_c1000000_l144_*_x*',
                           # y_path='../preparation/tryone/data/dump/b3b3_c1000000_l144_*_y*')
    # i = 0
    # for train_x, train_y in data:
      # i += 1
    # end = timer()
    # print end - start
    # print i
    pass

  def test_iteration(self):
    print len(self.batch_processor)
    # start = timer()
    # i = 0
    # for ds in self.batch_processor:
      # i += 1
    # end = timer()
    # print end - start
    # print i
    pass

class TestPreprocessor(unittest.TestCase):

  def setUp(self):
    self.preprocessor = ImgPreprocessor(X_imgpath='./tests/data/train/2.png',
                                        y_dirpath='./tests/data/test/',
                                        train_size=0.8)

  def test_patchsize_according_bordersize(self):
    # The patch has to have (2*border+1)**2 pixels
    border = 3
    self.preprocessor.border = border
    ds = self.preprocessor.get_dataset()
    self.assertEqual(len(ds[0][0]), (2*border+1)**2)

    border = 4
    self.preprocessor.border = border
    ds = self.preprocessor.get_dataset()
    self.assertEqual(len(ds[0][0]), (2*border+1)**2)

  def test_patches_amounth(self):
    shape = np.array(PIL.Image.open(self.preprocessor.X_imgpath)).shape
    num_pixels = shape[0] * shape[1]
    num_entires = len(self.preprocessor.get_dataset())
    self.assertEqual(num_entires, num_pixels)

  def test_center_pixels(self):
    # Prepare test image
    img_array = np.array(PIL.Image.open(self.preprocessor.X_imgpath)).flatten()
    img_array = img_array / 255.

    # Generate Patches
    border = 3
    self.preprocessor.border = border
    X, y = zip(*self.preprocessor.get_dataset())
    center_index = (len(X[0]) - 1) / 2

    # The first patch's center pixel must be the first pixel of the image
    first_pixel = img_array[0]
    first_center_pixel = X[0][center_index]
    self.assertEqual(first_center_pixel, first_pixel)

    # The next pixel nighbour of the first patch's center pixel
    # must be the second pixel of the image
    next_pixel = img_array[1]
    next_center_pixel = X[0][center_index + 1]
    self.assertEqual(next_center_pixel, next_pixel)

    # The last patch's center pixel must be the last pixel in the image
    last_pixel = img_array[-1]
    last_center_pixel = X[-1][center_index]
    self.assertEqual(last_center_pixel, last_pixel)

    # The previous pixel nighbour of the last patch's center pixel
    # must be the second last pixel of the image
    second_last_pixel = img_array[-2]
    second_last_center_pixel = X[-1][center_index - 1]
    self.assertEqual(second_last_center_pixel, second_last_pixel)

  def test_y_pixels(self):
    name = os.path.basename(self.preprocessor.X_imgpath)
    y_imgpath = os.path.join(self.preprocessor.y_dirpath, name)
    img_array = np.array(PIL.Image.open(y_imgpath)).flatten() / 255.
    border = 1
    self.preprocessor.border = border
    X, y = zip(*self.preprocessor.get_dataset())
    self.assertEqual(y[0], img_array[0])
    self.assertEqual(y[len(y) / 2], img_array[len(y) / 2])
    self.assertEqual(y[-1], img_array[-1])


