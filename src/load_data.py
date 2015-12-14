#!/usr/bin/env python
import argparse
import cPickle
import numpy as np
import os.path
import PIL.Image
import PIL.ImageOps
import sys
import glob
import theano.tensor as T
import theano
from itertools import izip

np.random.seed(seed=59842093)

def pixel_patches(img, border):
    """Return an array of patch-images for each pixel of the image.
    The size of each patch is (border + 1, border + 1).
    To get patch-images for the boarder pixels, the images is expanded.
    The patch-images are flattened and the pixel value is converted to
    percentage. 0 => black 1 => white.
    """
    ext = np.asarray(PIL.ImageOps.expand(img, border=border, fill=0)) / 255.
    #ext =
    #ext = enlarged(np.asarray(img), border) / 255.
    h, w = ext.shape
    return [ext[x-border:x+border+1, y-border:y+border+1].flatten()
            for x in xrange(border, h-border)
            for y in xrange(border, w-border)]

def image_from_vec(vec, shape):
    orig = np.reshape(vec, shape) * 255.
    img = PIL.Image.fromarray(orig)
    return img

def load_imgs(path, limit=None, border=2):
    return [(img, cleaned_path(img))
            for img in glob.glob(path)[:limit]]

def load_dataset(path, limit=None, border=2):
    x, y = [], []
    for img in glob.glob(path)[:limit]:
        x.extend(x_from_image(img, border))
        y.extend(y_from_image(img))
    return zip(x, y)

def split_dataset(ds, percentage_train=0.8):
    np.random.shuffle(ds)
    train = int(len(ds) * percentage_train)
    valid = len(ds) * ((1. - percentage_train) / 2.)
    valid = int(np.ceil(valid))
    return (ds[:train], # train data
            ds[train:train + valid], # validation data
            ds[train + valid:]) # test data

def shared(dataset):
    """Place the data into shared variables. This allows Theano to copy
    the data to the GPU, if one is availale.
    """
    x, y = zip(*dataset)
    print 'shared enzipped'
    shared_x = theano.tensor._shared(
        np.asarray(x, dtype=theano.config.floatX), borrow=True)
    print 'shared_x'
    shared_y = theano.tensor._shared(
        np.asarray(y, dtype=theano.config.floatX), borrow=True)
    print 'shared_y'
    return shared_x, shared_y

def cleaned_path(path):
    name = os.path.basename(path)
    return os.path.join(os.path.dirname(path), '../train_cleaned', name)

def x_from_image(path, border):
    img = PIL.Image.open(path)
    return pixel_patches(img, border)

def y_from_image(source):
    img = PIL.Image.open(cleaned_path(source))
    ary = np.array(img) / 255.
    height, width = ary.shape
    return [ary[x, y].flatten()
            for x in xrange(height)
            for y in xrange(width)]

def load_model(path):
    with open(path) as f:
        return cPickle.load(f)

def process_datachunk(path, border=2, limit=100, chunksize=50000,
                      storepath=None):
    i, x, y = 0, [], []
    for img in glob.glob(path)[:limit]:
        x.extend(x_from_image(img, border))
        y.extend(y_from_image(img))
        if len(x) >= chunksize:
            size = len(x) - (len(x) % chunksize)
            num = size / chunksize
            x_chunks = np.split(np.asarray(x[:size]), num)
            y_chunks = np.split(np.asarray(y[:size]), num)
            x, y = x[size:], y[size:] # let the rest pass
            for xt, yt in izip(x_chunks, y_chunks):
                split_and_save_chuncks(zip(xt, yt), storepath=storepath, i=i)
                i += 1

def split_and_save_chuncks(ds, percentage_train=0.8, storepath=None, i=0):
    np.random.shuffle(ds)
    train = int(len(ds) * percentage_train)
    valid = len(ds) * ((1. - percentage_train) / 2.)
    valid = int(np.ceil(valid))

    dump = zip(*ds[:train])
    save_nparray(dump[0], storepath + "_train_x_" + str(i) + ".npy")
    save_nparray(dump[1], storepath + "_train_y_" + str(i) + ".npy")

    dump = zip(*ds[train:train + valid])
    save_nparray(dump[0], storepath + "_valid_x_" + str(i) + ".npy")
    save_nparray(dump[1], storepath + "_valid_y_" + str(i) + ".npy")

    dump = zip(*ds[train + valid:])
    save_nparray(dump[0], storepath + "_test_x_" + str(i) + ".npy")
    save_nparray(dump[1], storepath + "_test_y_" + str(i) + ".npy")

def save_nparray(a, path, dtype='float64'):
    with open(path, 'w') as f:
        np.save(f, np.asarray(a, dtype=dtype))

class DataIter(object):
    def __init__(self, x_path=None, y_path=None, limit=None):
        """Iterator for numpy datafiles. The files are sorted by the
        filename. Enables to preprocess large datasets and load them in an
        iterative manner. So The dataset can be bigger than the available
        memory.

        :param x_path (str) : File path to the x (features) numpy data.
                              Can have asterix (*) for multiple file
                              projection.
        :param y_path (str) : File path to the y (label) numpy data.
                              Can have asterix (*) for multiple file
                              projection.
        :param limit (int)  : Limit the files to be loaded. Good for testing
                              when not all the data is needed.
        """
        self.i = 0
        self.x_files = sorted(glob.glob(x_path))[:limit]
        self.y_files = sorted(glob.glob(y_path))[:limit]
        self.end = len(self.x_files) - 1
        self.__check()

    def __iter__(self):
        return self

    def next(self):
        """Gives back the next file data as numpy array. Raise StopIteration
        at the end, so it can be used as iterator."""
        if self.i > self.end:
            self.i = 0
            raise StopIteration
        else:
            self.i += 1
            x_path = self.x_files[self.i - 1]
            y_path = self.y_files[self.i - 1]
            return self.__load_file(x_path), self.__load_file(y_path)

    def zeros(self):
        """Returns numpy.zeros arrays for x and y data with the shape and
        size of the files to iterate over."""
        first_x = self.__load_file(self.x_files[0])
        first_y = self.__load_file(self.y_files[0])
        #zeros_x = np.zeros(first_x.shape, dtype=first_x.dtype)
        #zeros_y = np.zeros(first_y.shape, dtype=first_y.dtype)
        return first_x, first_y

    def __len__(self):
        return len(self.x_files)

    def __check(self):
        if len(self.x_files) is not len(self.y_files):
            err = "X-Files and Y-Files must be in same amonth."
            raise ValueError(err)
        first_x = self.__load_file(self.x_files[0])
        first_y = self.__load_file(self.y_files[0])
        if first_x.shape[0] != first_y.shape[0]:
            err = "The x and y data-chunks must have same length"
            raise ValueError(err)

    def __load_file(self, path):
        with open(path) as f:
            return np.load(f)

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('datapath', help='Path to the data.')
    parser.add_argument('savepath', help='Path to store the processed data.')
    parser.add_argument('-b', type=int, default=2, help='Border for X images',
                        dest='border')
    parser.add_argument('-cs', default=5, type=int, dest='chunksize',
                        help='How many images per file')
    parser.add_argument('-l', default=10, type=int, dest='limit',
                        help='how many images to load')
    args = parser.parse_args()

    storepath = args.savepath + "b" + str(args.border) + "_c" + \
               str(args.chunksize) + "_l" + str(args.limit)
    process_datachunk(args.datapath, border=args.border, limit=args.limit,
                      chunksize=args.chunksize, storepath=storepath)

    return 0

if __name__ == '__main__':
    sys.exit(main())

