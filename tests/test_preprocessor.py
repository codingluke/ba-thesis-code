import unittest
import numpy as np
import PIL.Image
import pdb
import os
from timeit import default_timer as timer
from itertools import izip

from src.preprocessor import ImgPreprocessor, BatchImgProcessor
import config

rnd = np.random.RandomState(232342)

class TestBatchImgProcessor(unittest.TestCase):

    def setUp(self):
        self.train_batch = BatchImgProcessor(
            X_dirpath='./tests/data/train/*',
            y_dirpath='./tests/data/test/',
            batchsize=100,
            border=1,
            limit=None,
            rnd=rnd)

        self.valid_batch = BatchImgProcessor(
            X_dirpath='../tests/data/valid/*',
            y_dirpath='./tests/data/test/',
            batchsize=100,
            border=1,
            limit=None,
            rnd=rnd)

    def tearDown(self):
        del self.train_batch, self.valid_batch

    def test_iterating(self):
        test1 = [(X,y) for X, y in self.train_batch]
        test2 = [(X,y) for X, y in self.train_batch]
        test3 = [(X,y) for X, y in self.train_batch]
        test4 = [(X,y) for X, y in self.train_batch]
        test5 = [(X,y) for X, y in self.train_batch]

    @unittest.skipUnless(config.slow, 'slow test')
    def test_bench(self):
        bp = BatchImgProcessor(
            X_dirpath='../data/train/*',
            y_dirpath='../data/train_cleaned/',
            batchsize=4000000,
            border=2,
            limit=None,
            rnd=rnd)

        start = timer()
        bp.random = False
        bp.slow = True
        for X, y in bp: None
        end = timer()
        print "slow:\t\t %d" % (end - start)

        start = timer()
        bp.random = True
        bp.slow = True
        for X, y in bp: None
        end = timer()
        print "slow rand:\t\t %d" % (end - start)

        start = timer()
        bp.random = False
        bp.slow = False
        for X, y in bp: None
        end = timer()
        print "fast:\t\t %d" % (end - start)

        start = timer()
        bp.random = True
        bp.slow = False
        for X, y in bp: None
        end = timer()
        print "fast rand:\t\t %d" % (end - start)

        start = timer()
        bp.random = True
        bp.random_mode = 'fully'
        bp.slow = False
        for X, y in bp: None
        end = timer()
        print "fully rand:\t\t %d" % (end - start)

    def test_iterating_fully_random(self):
        self.train_batch.random_mode = 'fully'
        fullset = [i for i in self.train_batch]
        fullset2 = [i for i in self.train_batch]

    def test_len(self):
        # Pixels manually calculated according test images
        pixels = 50*34*3 / self.train_batch.batchsize
        self.assertEqual(len(self.train_batch), pixels)

    def test_consitency(self):
        ds = self.train_batch.next()
        self.train_batch.reset()
        ds2 = self.train_batch.next()
        center_index = (len(ds[0][0]) - 1) / 2
        self.assertEqual(ds[0][0][center_index], ds2[0][0][center_index])

    def test_randomness(self):
        self.train_batch.random = True
        X, y = self.train_batch.next()
        self.train_batch.reset()
        X2, y2 = self.train_batch.next()
        center_index = (len(X[0]) - 1) / 2
        self.assertNotEqual(X[0][center_index], X2[0][center_index])
        self.assertNotEqual(X[-1][center_index], X2[-1][center_index])

        X3, y3 = self.train_batch.next()
        center_index = (len(X[0]) - 1) / 2
        self.assertNotEqual(X[0][center_index], X3[0][center_index])
        self.assertNotEqual(X[-1][center_index], X3[-1][center_index])

    def test_batch_lengths(self):
        validl = len(self.valid_batch)
        validl2 = len([x for x in self.valid_batch])
        self.assertEqual(validl, validl2)

        trainl = len(self.train_batch)
        trainl2 = len([x for x in self.train_batch])
        self.assertEqual(trainl, trainl2)

    def test_next_fully_random(self):
        set = self.train_batch.next_fully_random()
        set2 = self.train_batch.next()

        self.assertEqual(type(set), type(set2))
        self.assertEqual(set[0].shape, set2[0].shape)
        self.assertEqual(set[1].shape, set2[1].shape)

    def test_batch_sizes(self):
        eq = True
        for X, y in self.valid_batch:
          eq = (X.shape[0] == self.valid_batch.batchsize)
          if not eq: break
        self.assertEqual(eq, True)

        eq = True
        for X, y in self.train_batch:
          eq = (X.shape[0] == self.train_batch.batchsize)
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
            X_imgpath='./tests/data/valid/noise.png',
            y_dirpath='./tests/data/test/',
            border=1, rnd=rnd)

    def test_sliding_window(self):
        x1 = self.preprocessor.get_X_fast()
        x2 = self.preprocessor.get_X()
        y1 = self.preprocessor.get_y_fast()
        y2 = self.preprocessor.get_y()
        pass

    def test_get_random_patch(self):
        patch = self.preprocessor.get_random_patch()
        patch2 = self.preprocessor.get_random_patch()
        patch3 = self.preprocessor.get_random_patch()

        patch = patch[0].flatten()
        patch2 = patch2[0].flatten()
        patch3 = patch3[0].flatten()

        center_index = (len(patch) - 1) / 2
        self.assertNotEqual(patch[center_index], patch2[center_index])
        self.assertNotEqual(patch[center_index], patch3[center_index])
        self.assertNotEqual(patch2[center_index], patch3[center_index])

    def test_fully_random_consistency(self):
        self.preprocessor.load_images('./tests/data/train/gradient.png',
            './tests/data/train/')
        x, y = self.preprocessor.get_random_patch()
        x = x.flatten()
        center_index = (len(x) - 1) / 2
        self.assertEqual(x[center_index], y)

    def test_get_fast_x_consistency(self):
        self.preprocessor.load_images('./tests/data/train/gradient.png',
            './tests/data/train/')
        x = self.preprocessor.get_X_fast()[:2]
        y = self.preprocessor.get_y_fast()[:2]
        center_index = (len(x[0]) - 1) / 2
        self.assertEqual(x[0][center_index], y[0][0])
        self.assertEqual(x[1][center_index], y[1][0])

    def test_patchsize_according_bordersize(self):
        # The patch has to have (2*border+1)**2 pixels
        for border in [3, 4]:
            self.preprocessor.set_border(border)
            ds = self.preprocessor.get_dataset()
            self.assertEqual(len(ds[0][0]), (2*border+1)**2)

    def test_patches_amounth(self):
        shape = np.array(PIL.Image.open(self.preprocessor.X_imgpath)).shape
        num_pixels = shape[0] * shape[1]
        num_entires = len(self.preprocessor.get_dataset())
        self.assertEqual(num_entires, num_pixels)

    def test_center_pixels(self):
        # Prepare test image
        img_path = self.preprocessor.X_imgpath
        img_array = np.array(PIL.Image.open(img_path)).flatten()
        img_array = img_array / 255.

        # Generate Patches
        border = 3
        self.preprocessor.set_border(border)
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
