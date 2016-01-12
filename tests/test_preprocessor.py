import unittest
import numpy as np
import PIL.Image
import pdb

from src.preprocessor import ImgPreprocessor

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
