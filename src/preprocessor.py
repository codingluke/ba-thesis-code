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

    def __init__(self, X_dirpath=None, y_dirpath=None, border=3,
            train_stepover=8, limit=None, batchsize=None,
            modus='full', random=False):
        self.X_dirpath = X_dirpath
        self.y_dirpath = y_dirpath
        self.border = border
        self.train_stepover = train_stepover
        self.batchsize = batchsize
        self.modus = modus
        self.random = random
        self.limit = limit
        self.i = 0
        self.i_preprocessor = 0
        self.preprocessors = [ImgPreprocessor(X_imgpath=img,
                                              y_dirpath=y_dirpath,
                                              train_stepover=train_stepover,
                                              modus=self.modus)
                              for img in glob.glob(X_dirpath)[:limit]]
        self.buffer = []

    def to_modus(self, modus='full'):
        self.modus = modus
        for p in self.preprocessors:
            p.modus = self.modus

    def to_train_modus(self):
        self.to_modus('train')

    def to_full_modus(self):
        self.to_modus('full')

    def to_valid_modus(self):
        self.to_modus('valid')

    def __len__(self):
        buffer = rounding = 0
        for p in self.preprocessors:
            buffer += len(p)
        if (buffer % self.batchsize) != 0:
            rounding = 1
        return (buffer / self.batchsize) + rounding

    def __iter__(self):
        return self

    def reset(self):
        self.i = 0
        self.i_preprocessor = 0
        self.buffer = []

    def next(self):
        """Gives back the next file data as numpy array. Raise StopIteration
        at the end, so it can be used as iterator."""
        if self.i > len(self):
            self.reset()
            if self.random:
                np.random.shuffle(self.preprocessors)
            raise StopIteration
        else:
            self.i += 1
            if len(self.buffer) <= self.batchsize:
                for p in self.preprocessors[self.i_preprocessor:]:
                    self.buffer.extend(p.get_dataset())
                    self.i_preprocessor += 1
                    if len(self.buffer) >= self.batchsize:
                        break
            if self.random:
                np.random.shuffle(self.buffer)
            batch = self.buffer[:self.batchsize]
            self.buffer = self.buffer[self.batchsize:]
            return batch

class ImgPreprocessor(object):

    def __init__(self, X_imgpath=None, y_dirpath=None, border=3,
                 train_stepover=8, modus='full'):
        assert X_imgpath != None and isinstance(X_imgpath, str)
        assert y_dirpath != None and isinstance(y_dirpath, str)

        self.X_img, self.y_img = self._load_images(X_imgpath, y_dirpath)
        self.X_imgpath = X_imgpath
        self.y_dirpath = y_dirpath
        self.modus = modus
        self.border = border
        self.stepover = train_stepover

    def get_dataset(self):
        return zip(self._get_X(), self._get_y())

    def __len__(self):
        h, w = np.array(self.X_img).shape
        if self.modus == 'full':
            return h * w
        elif self.modus == 'train':
            x, y = self._get_range((h,w))
            lenx = len([i for i in x])
            leny = len([i for i in y])
            return (lenx * leny)
        elif self.modus == 'valid':
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

    def _get_y(self):
        ary = np.array(self.y_img) / 255.
        height_range, width_range = self._get_range(ary.shape)
        return [ary[x, y].flatten()
                for x in height_range
                for y in width_range]

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
        height_range, width_range = self._get_range(ext.shape, b)
        return [ext[x-b:x+b+1, y-b:y+b+1].flatten()
                for x in height_range
                for y in width_range]

    def _get_range(self, shape, border=0):
        height, width = shape
        if self.modus == 'full':
            return xrange(border, height-border), \
                   xrange(border, width-border)
        elif self.modus == 'train':
            return TrainRange(border, height-border, self.stepover), \
                   TrainRange(border, width-border)
        elif self.modus == 'valid':
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
