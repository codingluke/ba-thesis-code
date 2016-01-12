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

class BatchImgProcessor(object):

    def __init__(self, X_dirpath=None, y_dirpath=None, border=3,
            train_size=0.8, limit=None, batchsize=None):
        self.X_dirpath = X_dirpath
        self.y_dirpath = y_dirpath
        self.border = border
        self.train_size = train_size
        self.batchsize = batchsize
        self.limit = limit
        self.i = 0
        self.i_preprocessor = 0
        self.preprocessors = [ImgPreprocessor(X_imgpath=img,
                                              y_dirpath=y_dirpath,
                                              train_size=train_size)
                              for img in glob.glob(X_dirpath)[:limit]]
        self.buffer = []

    def __len__(self):
        buffer = 0
        for p in self.preprocessors:
            buffer += len(p)
        return (buffer / self.batchsize)

    def __iter__(self):
        return self

    def next(self):
        """Gives back the next file data as numpy array. Raise StopIteration
        at the end, so it can be used as iterator."""
        if self.i > len(self):
            self.i = 0
            self.i_preprocessor = 0
            raise StopIteration
        else:
            self.i += 1
            if len(self.buffer) <= self.batchsize:
                for p in self.preprocessors[self.i_preprocessor:]:
                    self.buffer.extend(p.get_dataset())
                    self.i_preprocessor += 1
                    if len(self.buffer) >= self.batchsize:
                        break
            batch = self.buffer[:self.batchsize]
            self.buffer = self.buffer[self.batchsize:]
            return batch

class ImgPreprocessor(object):

    def __init__(self, X_imgpath=None, y_dirpath=None, border=3,
                 train_size=0.8):
        assert X_imgpath != None and isinstance(X_imgpath, str)
        assert y_dirpath != None and isinstance(y_dirpath, str)

        self.X_img, self.y_img = self._load_images(X_imgpath, y_dirpath)
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

    def __len__(self):
        h,w = np.array(self.X_img).shape
        return h * w

    def _load_images(self, X_imgpath, y_dirpath):
        name = os.path.basename(X_imgpath)
        y_imgpath = os.path.join(y_dirpath, name)
        X_img = PIL.Image.open(X_imgpath)
        y_img = PIL.Image.open(y_imgpath)
        return X_img, y_img

    def _get_y(self):
        ary = np.array(self.y_img) / 255.
        height, width = ary.shape
        return [ary[x, y].flatten()
                for x in xrange(height)
                for y in xrange(width)]

    def _get_X(self):
        """Return an array of patch-images for each pixel of the image.
        The size of each patch is (self.border + 1, self.border + 1).
        To get patch-images for the boarder pixels, the images is expanded.
        The patch-images are flattened and the pixel value is converted to
        percentage. 0 => black 1 => white.
        """
        b = self.border
        ext = np.asarray(PIL.ImageOps.expand(self.X_img, border=b, fill=0))
        ext = ext / 255.
        height, width = ext.shape
        return [ext[x-b:x+b+1, y-b:y+b+1].flatten()
                for x in xrange(b, height-b)
                for y in xrange(b, width-b)]

    def _get_train_X(self):
        b = self.border
        ext = np.asarray(PIL.ImageOps.expand(self.X_img, border=b, fill=0))
        ext = ext / 255.
        height, width = ext.shape
        return [ext[x-b:x+b+1, y-b:y+b+1].flatten()
                for x in xrange(b, height-b)
                for y in xrange(b, width-b)]

