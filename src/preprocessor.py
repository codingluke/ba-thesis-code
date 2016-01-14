#!/usr/bin/env python

import numpy as np
import os.path
import PIL.Image
import PIL.ImageOps
import sys
import glob
from itertools import izip
import pdb

np.random.seed(seed=59842093)

class ImgProcessor(object):

  @staticmethod
  def image_from_vec(vec, shape):
      orig = np.reshape(vec, shape) * 255.
      return PIL.Image.fromarray(orig)

class BatchImgProcessor(object):

    @classmethod
    def load(cls, X_dirpath=None, y_dirpath=None, border=3,
            train_stepover=8, limit=None, batchsize=None, dtype='float64'):
        cls.X_dirpath = X_dirpath
        cls.y_dirpath = y_dirpath
        cls.border = border
        cls.train_stepover = train_stepover
        cls.batchsize = batchsize
        cls.limit = limit
        cls.dtype = dtype
        cls.preprocessors = [ImgPreprocessor(X_imgpath=img,
                                             y_dirpath=y_dirpath,
                                             train_stepover=train_stepover)
                             for img in glob.glob(X_dirpath)[:limit]]
        return cls

    def __init__(self, modus='full', random=False):
        self.to_modus(modus)
        self.random = random
        self.i = 0
        self.i_preprocessor = 0
        self.buffer = []

    def to_modus(self, modus='full'):
        self.modus = modus

    def to_train_modus(self):
        self.to_modus('train')

    def to_full_modus(self):
        self.to_modus('full')

    def to_valid_modus(self):
        self.to_modus('valid')

    def __len__(self):
        buffer = rounding = 0
        for p in self.preprocessors:
            buffer += p.length(modus=self.modus)
        return (buffer / self.batchsize)

    def __iter__(self):
        return self

    def reset(self):
        self.i = 0
        self.i_preprocessor = 0
        self.buffer = []

    def next(self):
        """Gives back the next file data as numpy array. Raise StopIteration
        at the end, so it can be used as iterator."""
        if self.i > len(self) - 1:
            self.reset()
            if self.random:
                np.random.shuffle(self.preprocessors)
            raise StopIteration
        else:
            self.i += 1
            if len(self.buffer) <= self.batchsize:
                for p in self.preprocessors[self.i_preprocessor:]:
                    self.buffer.extend(p.get_dataset(modus=self.modus))
                    self.i_preprocessor += 1
                    if len(self.buffer) >= self.batchsize:
                        break
            if self.random:
                np.random.shuffle(self.buffer)
            if len(self.buffer[:self.batchsize]) == self.batchsize:
              X, y = izip(*self.buffer[:self.batchsize])
              self.buffer = self.buffer[self.batchsize:]
              return np.asarray(X, dtype=self.dtype),  \
                     np.asarray(y, dtype=self.dtype)
            else:
              raise StopIteration

class ImgPreprocessor(object):

    def __init__(self, X_imgpath=None, y_dirpath=None, border=3,
                 train_stepover=8):
        assert X_imgpath != None and isinstance(X_imgpath, str)
        assert y_dirpath != None and isinstance(y_dirpath, str)

        self.X_img, self.y_img = self._load_images(X_imgpath, y_dirpath)
        self.X_imgpath = X_imgpath
        self.y_dirpath = y_dirpath
        self.border = border
        self.stepover = train_stepover

    def get_dataset(self, modus=None):
        return zip(self._get_X(modus=modus), self._get_y(modus=modus))

    def length(self, modus=None):
        h, w = np.array(self.X_img).shape
        if not modus or modus == 'full':
            return h * w
        elif modus == 'train':
            x, y = self._get_range((h,w))
            lenx = len([i for i in x])
            leny = len([i for i in y])
            return (lenx * leny)
        elif modus == 'valid':
            x, y = self._get_range((h,w))
            lenx = len([i for i in x])
            leny = len([i for i in y])
            return (lenx * leny)

    def _load_images(self, X_imgpath, y_dirpath):
        name = os.path.basename(X_imgpath)
        y_imgpath = os.path.join(y_dirpath, name)
        X_img = PIL.Image.open(X_imgpath)
        y_img = PIL.Image.open(y_imgpath)
        return X_img, y_img

    def _get_y(self, modus=None):
        ary = np.array(self.y_img) / 255.
        height_range, width_range = self._get_range(ary.shape, modus=modus)
        return np.asarray([ary[x, y].flatten()
                for x in height_range
                for y in width_range])

    def _get_X(self, modus=None):
        """Return an array of patch-images for each pixel of the image.
        The size of each patch is (self.border + 1, self.border + 1).
        To get patch-images for the boarder pixels, the images is expanded.
        The patch-images are flattened and the pixel value is converted to
        percentage. 0 => black 1 => white.
        """
        b = self.border
        ext = np.asarray(PIL.ImageOps.expand(self.X_img, border=b, fill=0))
        ext = ext / 255.
        height_range, width_range = self._get_range(ext.shape,
                                                    border=b,
                                                    modus=modus)
        return np.asarray([ext[x-b:x+b+1, y-b:y+b+1].flatten()
                for x in height_range
                for y in width_range])

    def _get_range(self, shape, border=0, modus=None):
        height, width = shape
        if not modus or modus == 'full':
            return xrange(border, height-border), \
                   xrange(border, width-border)
        elif modus == 'train':
            return TrainRange(border, height-border, self.stepover), \
                   TrainRange(border, width-border)
        elif modus == 'valid':
            return xrange(border, height-border, self.stepover), \
                   xrange(border, width-border)

class TrainRange(object):

    def __init__(self, start, end, stepover=None):
        self.start = start
        self.end = end
        self.stepover = stepover
        self.act = start

    def __iter__(self):
        return self

    def next(self):
        if self.act >= self.end:
            self.act = self.start
            raise StopIteration
        else:
            self.act += 1
            if self.stepover and \
               (((self.act - 1) - self.start) % self.stepover) == 0:
                self.act += 1
            return self.act - 1
