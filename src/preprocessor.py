#!/usr/bin/env python
"""preprocessor.py
Includes classes to preprocess images (Processor) and iterate over them (BatchProcessor), so that they can be used to train neural networks
described in the paper "Enhancement and Cleaning of Handwritten Data by
Using Neural Networks" http://dx.doi.org/10.1007/11492429_46.
"""

import numpy as np
import os.path
import PIL.Image, PIL.ImageOps
import glob

from itertools import izip
from numpy.lib.stride_tricks import as_strided as ast

class BatchProcessor(object):

    def __init__(self, random=False, slow=False, X_dirpath=None,
                 y_dirpath=None, border=3, limit=None, batchsize=None,
                 dtype='float32', random_mode='normal', rnd=None):
        """A Python iteratior class, uses Processor classes for iterating
        over multiple images, so that not all data has to be loaded in memory.

        Keyword arguments:

        X_dirpath    -- Path to the dirty images.
        y_dirpath    -- Path to the clean images.
        border       -- Size of the border with neighbour pixels. (3)
        limit        -- Limit images to load (None)
        batchisize   -- Size of a batch for one iteration.
        dtype        -- numpy data type for returning values. ('float32')
        random       -- When True, the dataset is shuffled after a
                        completed iteration. (False)
        random_mode  -- When set to 'fully', it uses the fully random
                        mode ('normal')
        slow         -- When True, the dataset is generated unoptimized, not
                        recommended. (False)
        rnd          -- Numpy RandomState (None)
        """

        self.X_dirpath = X_dirpath
        self.y_dirpath = y_dirpath
        self.border = border
        self.batchsize = batchsize
        self.limit = limit
        self.dtype = dtype
        self.preprocessors = [Processor(X_imgpath=img, y_dirpath=y_dirpath,
                                        border=border, rnd=rnd)
                             for img in glob.glob(X_dirpath)[:limit]]
        self.slow = slow
        self.random = random
        self.i = 0
        self.i_preprocessor = 0
        self.buffer = []
        self.random_mode = random_mode
        self.rnd = rnd

    def __len__(self):
        """Returns the size of available batches"""

        return int(self.dataset_size() / self.batchsize)

    def dataset_size(self):
        """Returns the size of avaliable sub images from all images"""

        length = 0
        for p in self.preprocessors:
            length += p.length(slow=self.slow)
        return length

    def num_lost_datasets(self):
        """Returns the amount of datasets lost, because of the batchsize."""

        return self.dataset_size() - self.size()

    def size(self):
        """Returns the size of actual available sub images in all batches"""

        return len(self) * self.batchsize

    def __iter__(self):
        """Iterator function for the class, returns itself"""

        return self

    def reset(self):
        """Resets the iterator"""

        self.i = 0
        self.i_preprocessor = 0
        self.buffer = []
        if self.random: self.rnd.shuffle(self.preprocessors)

    def next(self):
        """Returns the next batch as numpy array. Raise StopIteration
        at the end, so it can be used as iterator."""
        if self.i > len(self) - 1:
            self.reset()
            raise StopIteration
        else:
            self.i += 1
            if self.random and self.random_mode == 'fully':
                return self.next_fully_random()
            if len(self.buffer) <= self.batchsize:
                for p in self.preprocessors[self.i_preprocessor:]:
                    self.buffer.extend(p.get_dataset(slow=self.slow))
                    self.i_preprocessor += 1
                    if len(self.buffer) >= self.batchsize: break
            batch = np.asarray(self.buffer[:self.batchsize])
            if self.random: self.rnd.shuffle(batch)
            X, y = izip(*batch)
            self.buffer = self.buffer[self.batchsize:]
            return np.asarray(X, dtype=self.dtype),  \
                   np.asarray(y, dtype=self.dtype)

    def next_fully_random(self):
        """Returns a batch of fully random picked sub images by randomly
        chosen images."""
        high = len(self.preprocessors) - 1
        X, Y = [], []
        for i in xrange(self.batchsize):
            rnd = high
            if rnd > 0: rnd = self.rnd.randint(0, high)
            patch = self.preprocessors[rnd].get_random_patch()
            X.append(patch[0])
            Y.append(patch[1])
        l = (2*self.border+1)**2
        return np.asarray(X, dtype=self.dtype).reshape(self.batchsize, l), \
               np.asarray(Y, dtype=self.dtype).reshape(self.batchsize, 1)

class Processor(object):
    """The Processor produces for every Pixel in the dirty image a
    sub image including the pixel and a border of neighbours with
    a variable size."""

    def __init__(self, X_imgpath=None, y_dirpath=None, border=3, rnd=None):
        assert X_imgpath != None and isinstance(X_imgpath, str)

        self.border = border
        self.X_img, self.y_img, self.pixels = \
            self.load_images(X_imgpath, y_dirpath)
        self.X_imgpath = X_imgpath
        self.y_dirpath = y_dirpath
        self.rnd = rnd

    def get_dataset(self, slow=False):
        """Returns the dataset including the input (X) and target (y).
        When the parameter `slow` is True, it uses the unoptimized, slower
        implementation. It just exists for benchmark perpuses.
        """
        if slow: return zip(self.get_X(), self.get_y())
        else: return zip(self.get_X_fast(), self.get_y_fast())

    def length(self, slow=False):
        """Returns the length of the dataset, in other words, the
        amount of pixels"""

        return self.pixels

    def load_images(self, X_imgpath, y_dirpath):
        """Loads the X and y images. This means it calculates their sizes,
        and converts them to numpy arrays. The X images is extended by the
        border size, with black pixels."""

        name = os.path.basename(X_imgpath)
        X_img = PIL.Image.open(X_imgpath)
        pixels = X_img.size[0] * X_img.size[1]
        X_img = np.asarray(PIL.ImageOps.expand(X_img,
                border=self.border, fill=0)) / 255.
        y_img = None
        if y_dirpath:
            y_imgpath = os.path.join(y_dirpath, name)
            y_img = np.array(PIL.Image.open(y_imgpath)) / 255.
        self.X_img = X_img
        self.y_img = y_img
        self.pixels = pixels
        return X_img, y_img, pixels

    def get_random_patch(self):
        """Returns a random subimage and its corresponding target pixel"""

        b = self.border
        height, width = self.X_img.shape
        _x = self.rnd.randint(b, height-b)
        _y = self.rnd.randint(b, width-b)
        return self.X_img[_x-b:_x+b+1, _y-b:_y+b+1], \
               self.y_img[_x-b,_y-b]

    def get_y(self):
        """Returns all target pixels in sorted order"""
        height_range, width_range = self.__get_range(self.y_img.shape)
        return np.asarray([self.y_img[x, y].flatten()
                           for x in height_range
                           for y in width_range])

    def get_y_fast(self):
        """Returns all target pixels in sorted order, fast"""

        return self.y_img.flatten().reshape(self.pixels, 1)

    def get_X(self):
        """Return an array of patch-images for each pixel of the image.
        The size of each patch is (self.border + 1, self.border + 1).
        The patch-images are flattened and the pixel value is converted to
        percentage. 0 => black 1 => white.
        """

        b = self.border
        height_range, width_range = self.__get_range(self.X_img.shape,
                                                    border=b)
        return np.asarray([self.X_img[x-b:x+b+1, y-b:y+b+1].flatten()
                for x in height_range
                for y in width_range])

    def get_X_fast(self):
        """Return an array of patch-images for each pixel of the image.
        The size of each patch is (self.border + 1, self.border + 1).
        The patch-images are flattened and the pixel value is converted to
        percentage. 0 => black 1 => white, fast.
        """

        size = 2*self.border+1
        return sliding_window(self.X_img, (size, size),
                              (1, 1)).reshape(self.pixels, size**2)

    def set_border(self, border):
        """Sets a new border. For this the images have to be reloaded."""

        self.border = border
        self.X_img, self.y_img, self.pixels = \
            self.load_images(self.X_imgpath, self.y_dirpath)

    def __get_range(self, shape, border=0):
        """Returns the range for iterating over the pixels in the X images.
        Its needed caus of the border. Borderpixels have to be ignored."""

        height, width = shape
        return xrange(border, height-border), \
               xrange(border, width-border)


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
