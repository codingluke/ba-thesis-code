#!/usr/bin/env python

import numpy as np
import os.path
import PIL.Image
import PIL.ImageOps
import sys
import glob
import pdb

from itertools import izip
from numpy.lib.stride_tricks import as_strided as ast

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
                                             train_stepover=train_stepover,
					     border=border)
                             for img in glob.glob(X_dirpath)[:limit]]
        return cls

    def __init__(self, modus='full', random=False, slow=False):
        self.slow = slow
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
            buffer += p.length(modus=self.modus, slow=self.slow)
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
                    self.buffer.extend(p.get_dataset(modus=self.modus,
                                                     slow=self.slow))
                    self.i_preprocessor += 1
                    if len(self.buffer) >= self.batchsize:
                        break
            if len(self.buffer[:self.batchsize]) == self.batchsize:
                batch = np.asarray(self.buffer[:self.batchsize])
                if self.random: np.random.shuffle(batch)
                # if self.random: batch = self.fs(batch)
                X, y = izip(*batch)
                self.buffer = self.buffer[self.batchsize:]
                return np.asarray(X, dtype=self.dtype),  \
                       np.asarray(y, dtype=self.dtype)
            else:
              raise StopIteration

    def fs(self, series):
        split = np.array(np.split(series, self.batchsize / 4))
        np.random.shuffle(split)
        return np.concatenate(split)

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

    def get_dataset(self, modus=None, slow=False):
        if slow:
            return zip(self._get_X(modus=modus), self._get_y(modus=modus))
        else:
            return zip(self._get_X_fast(modus=modus),
                       self._get_y_fast(modus=modus))

    def length(self, modus=None, slow=False):
        if slow:
            return self.length_slow(modus=modus)
        pixels = self.X_img.size[0] * self.X_img.size[1]
        if modus == None or modus == 'full':
            return pixels
        else:
            if modus == 'train':
                return pixels * 0.8
            if modus == 'valid':
                return pixels * (1 - 0.8)

    def length_slow(self, modus=None):
        pixels = self.X_img.size[0] * self.X_img.size[1]
        if modus == None or modus == 'full':
            return pixels
        else:
            x, y = self._get_range(self.X_img.size, modus=modus)
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

    def _get_y_fast(self, modus=None):
        ary = np.array(self.y_img) / 255.
        pixels = self.X_img.size[0] * self.X_img.size[1]
        patches = ary.flatten().reshape(pixels, 1)
        if modus == None or modus == 'full':
            return patches
        if modus == 'train':
            return patches[:pixels*0.8]
        if modus == 'valid':
            return patches[pixels*0.8:]

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

    def _get_X_fast(self, modus=None):
        b = self.border
        ext = np.asarray(PIL.ImageOps.expand(self.X_img, border=b, fill=0))
        pixels = self.X_img.size[0] * self.X_img.size[1]
        ext = ext / 255.
        size = 2*b+1
        patches = sliding_window(ext, (size, size),
                                 (1, 1)).reshape(pixels, size**2)
        if modus == None or modus == 'full':
            return patches
        if modus == 'train':
            return patches[:pixels*0.8]
        if modus == 'valid':
            return patches[pixels*0.8:]

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


def sliding_window(a, ws, ss=None):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)


    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape, strides = newstrides)
    return strided


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')
