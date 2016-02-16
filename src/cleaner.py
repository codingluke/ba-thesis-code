# coding: utf-8
"""cleaner.py
~~~~~~~
A python program dependend on network.py and preprocessor.py for
automatically clean single or many images with a trained and
saved neural network.
"""

# standard libraries
import cPickle
import numpy as np
import os
import glob
import gzip

# third party libraries
import PIL

# own libraries
from network import Network
from preprocessor import ImgPreprocessor

class BatchCleaner(object):

    def __init__(self, dirty_dir=None, model_path=None, limit=None):
        """Class for cleaning a whole directory with dirty images.

        dirty_dir   -- Path to the directory with the dirty images
        model_path  -- Path to the persistent model for cleaning
        limit       -- Optional limitation of images
        """

        self.cleaner = Cleaner(model_path=model_path)
        self.images = [img for img in glob.glob(dirty_dir + "*")[:limit]]

    def clean_and_save(self, output_dir=None):
        """Cleands all images and saves them to the `output_dir`."""

        for img in self.images:
            self.cleaner.clean_and_save(img, savepath=output_dir)

    def clean_for_submission(self, output_dir=None):
        """Cleands all images and saves them to a ziped file
        submission.txt.gz, into the output_dir."""

        filepath = os.path.join(output_dir, 'submission.txt.gz')
        with gzip.open(filepath, 'w') as f:
            f.write('id,value\n')
            for img in self.images:
                img, id = self.cleaner.clean(img)
                for line in self.cleaner.to_submission_format(img, id):
                    f.write(line)

class Cleaner(object):
    """Class to load a Network instance for cleaning images"""

    def __init__(self, model_path=None):
        """Loads a network given a `model_path` to the persistent instance"""

        f = open(model_path, 'rb')
        self.net = cPickle.load(f)
        self.border = self.__calc_border(self.net.layers[0])

    def clean(self, img_path=None):
        """Retruns a clean image as PIL.IMAGE given the `img_path` to
        the dirty image"""

        id = os.path.basename(img_path)[:-len('.png')]
        p = ImgPreprocessor(X_imgpath=img_path, border=self.border)
        patches = p.get_X_fast()
        y = self.net.predict(patches)
        y2 = np.vstack(np.array(y).flatten())
        w, h = p.X_img.shape
        orig = np.resize(y2, (w-2*self.border,h-2*self.border)) * 255
        del p, patches
        return PIL.Image.fromarray(orig), id

    def clean_and_show(self, img_path=None):
        """Cleans a dirty image given by the `img_path` and directly
        opens it."""

        img, _ = self.clean(img_path)
        img.show()

    def clean_and_save(self, img_path=None, savepath=None):
        """Cleans a dirty image given by the `img_path` and directly
        saves it to the `savepath`."""

        img, id = self.clean(img_path)
        img.convert('L').save(os.path.join(savepath, id + '.png'))
        img.close()

    def to_submission_format(self, cleaned_img=None, id=None):
        """Converts and returns a `cleaned_img` PIL.Image, to the submission
        form from Kaggle"""

        pixels = np.array(cleaned_img) / 255.0
        it = np.nditer(pixels, flags=['multi_index'])
        out = []
        while not it.finished:
            pixel = it[0]
            i, j = it.multi_index
            out.append('{}_{}_{},{}\n'.format(id, i+1, j+1, pixel))
            it.iternext()
        return out

    def metadata(self):
        """Returns the metadata from the model"""

        try: return self.net.meta
        except: return None

    def __calc_border(self, layer):
        """Calculate and returns the used border by the model"""

        return int((np.sqrt(layer.n_in) - 1) * 0.5)

