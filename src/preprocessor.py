#!/usr/bin/env python

import numpy as np
import os.path
import PIL.Image
import PIL.ImageOps
import sys
import glob
from itertools import izip

np.random.seed(seed=59842093)

class ImgProcessor(object):

  @staticmethod
  def image_from_vec(vec, shape):
      orig = np.reshape(vec, shape) * 255.
      return PIL.Image.fromarray(orig)

class ImgPreprocessor(object):

  def __init__(self, X_imgpath=None, y_dirpath=None, border=3, train_size=0.8):
    assert X_imgpath != None and isinstance(X_imgpath, str)
    assert y_dirpath != None and isinstance(y_dirpath, str)
    self.X_imgpath = X_imgpath
    self.y_dirpath = y_dirpath
    self.border = border

  def get_dataset(self):
    return zip(self._get_X(), self._get_y())

  def get_random_dataset(self):
    return np.random.shuffle(self.get_dataset())

  def get_train_dataset(self):
      None

  def get_valid_dataset(self):
      None

  def _get_X(self):
    img = PIL.Image.open(self.X_imgpath)
    return self._get_pixel_patches(img)

  def _get_y(self):
    name = os.path.basename(self.X_imgpath)
    y_imgpath = os.path.join(self.y_dirpath, name)
    img = PIL.Image.open(y_imgpath)
    ary = np.array(img) / 255.
    height, width = ary.shape
    return [ary[x, y].flatten()
            for x in xrange(height)
            for y in xrange(width)]

  def _get_pixel_patches(self, img):
    """Return an array of patch-images for each pixel of the image.
    The size of each patch is (self.border + 1, self.border + 1).
    To get patch-images for the boarder pixels, the images is expanded.
    The patch-images are flattened and the pixel value is converted to
    percentage. 0 => black 1 => white.
    """
    b = self.border
    ext = np.asarray(PIL.ImageOps.expand(img, border=b, fill=0)) / 255.
    height, width = ext.shape
    return [ext[x-b:x+b+1, y-b:y+b+1].flatten()
            for x in xrange(b, height-b)
            for y in xrange(b, width-b)]
