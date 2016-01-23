import unittest
import numpy as np
import PIL.Image
import pdb
import os
from timeit import default_timer as timer
from itertools import izip

from src.preprocessor import ImgPreprocessor, BatchImgProcessor, TrainRange
from src.load_data import DataIter
import config

class TestBatchImgProcessor(unittest.TestCase):

  def setUp(self):
    self.BatchProcessor = BatchImgProcessor.load(
        X_dirpath='./tests/data/train/*',
        y_dirpath='./tests/data/test/',
        batchsize=100,
        border=1,
        limit=None,
        train_stepover=8)
    self.full_batch = self.BatchProcessor()
    self.train_batch = self.BatchProcessor(modus='train')
    self.valid_batch = self.BatchProcessor(modus='valid')

  def tearDown(self):
    del self.full_batch
    del self.train_batch
    del self.valid_batch
    del self.BatchProcessor

  def test_size(self):
    BP = BatchImgProcessor.load(
        X_dirpath='../data/train/*',
        y_dirpath='../data/train_cleaned/',
        batchsize=1000000,
        border=3,
        limit=None,
        train_stepover=8)

  def test_iterating(self):
    test1 = [(X,y) for X, y in self.train_batch]
    test2 = [(X,y) for X, y in self.train_batch]
    test3 = [(X,y) for X, y in self.train_batch]
    test4 = [(X,y) for X, y in self.train_batch]
    test5 = [(X,y) for X, y in self.train_batch]

  @unittest.skipUnless(config.slow, 'slow test')
  def test_bench(self):
    BP = BatchImgProcessor.load(
        X_dirpath='../data/train/*',
        y_dirpath='../data/train_cleaned/',
        batchsize=50000,
        border=3,
        limit=2,
        train_stepover=8)
    full_slow = BP(random=False, slow=True)
    full_slow_random = BP(random=True, slow=True)
    full_fast = BP(random=False, slow=False)
    full_fast_random = BP(random=True, slow=False)

    start = timer()
    for X, y in full_slow: None
    end = timer()
    print "slow:\t\t %d" % (end - start)

    start = timer()
    for X, y in full_slow_random: None
    end = timer()
    print "slow rand:\t\t %d" % (end - start)

    start = timer()
    for X, y in full_fast: None
    end = timer()
    print "fast:\t\t %d" % (end - start)

    start = timer()
    for X, y in full_fast_random: None
    end = timer()
    print "fast rand:\t\t %d" % (end - start)

  def test_len(self):
    # Pixels manually calculated according test images
    pixels = 50*34*2 / self.full_batch.batchsize
    self.assertEqual(len(self.full_batch), pixels)

  def test_consitency(self):
    ds = self.full_batch.next()
    self.full_batch.reset()
    ds2 = self.full_batch.next()
    center_index = (len(ds[0][0]) - 1) / 2
    self.assertEqual(ds[0][0][center_index], ds2[0][0][center_index])

  def test_randomness(self):
    self.full_batch.random = True
    X, y = self.full_batch.next()
    self.full_batch.reset()
    X2, y2 = self.full_batch.next()
    center_index = (len(X[0]) - 1) / 2
    self.assertNotEqual(X[0][center_index], X2[0][center_index])
    self.assertNotEqual(X[-1][center_index], X2[-1][center_index])

  def test_train_range_intersec(self):
    valid_range = [x for x in xrange(3,258+6-3,8)]
    train_range = [x for x in TrainRange(3,258+6-3,8)]
    intersec = np.intersect1d(valid_range, train_range)
    total_len = len(valid_range) + len(train_range)
    self.assertEqual(len(intersec), 0)
    self.assertEqual(total_len, 258)

    valid_range = [x for x in xrange(0,258,7)]
    train_range = [x for x in TrainRange(0,258,7)]
    intersec = np.intersect1d(valid_range, train_range)
    total_len = len(valid_range) + len(train_range)
    self.assertEqual(len(intersec), 0)
    self.assertEqual(total_len, 258)

  def test_batch_lengths(self):
    validl = len(self.valid_batch)
    validl2 = len([x for x in self.valid_batch])
    self.assertEqual(validl, validl2)

    trainl = len(self.train_batch)
    trainl2 = len([x for x in self.train_batch])
    self.assertEqual(trainl, trainl2)

    fulll = len(self.full_batch)
    fulll2 = len([x for x in self.full_batch])
    self.assertEqual(fulll, fulll2)

  def test_batch_sizes(self):
    eq = True
    for X, y in self.valid_batch:
      eq = (X.shape[0] == self.BatchProcessor.batchsize)
      if not eq: break
    self.assertEqual(eq, True)

    eq = True
    for X, y in self.train_batch:
      eq = (X.shape[0] == self.BatchProcessor.batchsize)
      if not eq: break
    self.assertEqual(eq, True)

  # def test_valid_train_difference(self):
    # count = 0
    # for Xvs, yvs in self.valid_batch:
        # for Xts, yts in self.train_batch:
            # for Xv in Xvs:
                # for Xt in Xts:
                    # eq = np.array_equal(Xv, Xt)
                    # if eq: count += 1; break
                # if eq: break
            # if eq: break
        # if eq: break
    # self.assertEqual(count, 0)

class TestPreprocessor(unittest.TestCase):

  def setUp(self):
    self.preprocessor = ImgPreprocessor(
        X_imgpath='./tests/data/train/noise.png',
        y_dirpath='./tests/data/test/',
        border=1,
        train_stepover=8)

  def test_sliding_window(self):
    ds1 = self.preprocessor._get_X_fast()
    ds2 = self.preprocessor._get_X_fast(modus='train')
    ds3 = self.preprocessor._get_X_fast(modus='valid')

    y1 = self.preprocessor._get_y_fast()
    y2 = self.preprocessor._get_y()
    pass

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

  # def test_valid_train_difference(self):
    # full_set = self.preprocessor.get_dataset()
    # train_set = self.preprocessor.get_dataset(modus='train')
    # valid_set = self.preprocessor.get_dataset(modus='valid')
    # count = 0
    # for Xv, yv in valid_set:
      # for Xt, yt in train_set:
        # eq = np.array_equal(Xv, Xt)
        # if eq: count += 1; break
      # if eq: break
    # self.assertEqual(count, 0)

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


