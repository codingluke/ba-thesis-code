import cPickle
import PIL
import numpy as np
import os
import glob
import gzip

from network import Network
from preprocessor import ImgPreprocessor

class BatchCleaner(object):

    def __init__(self, dirty_dir=None, limit=None, model_path=None):
        self.cleaner = Cleaner(model_path=model_path)
        self.images = [img for img in glob.glob(dirty_dir + "*")[:limit]]

    def clean_and_save(self, output_dir=None):
        for img in self.images:
            self.cleaner.clean_and_save(img, savepath=output_dir)

    def clean_for_submission(self, output_dir=None):
        filepath = os.path.join(output_dir, 'submission.txt.gz')
        with gzip.open(filepath, 'w') as f:
            f.write('id,value\n')
            for img in self.images:
                img, id = self.cleaner.clean(img)
                for line in self.cleaner.to_submission_format(img, id):
                    f.write(line)

class Cleaner(object):

    def __init__(self, model_path=None):
        f = open(model_path, 'rb')
        self.net = cPickle.load(f)
        self.border = self.__calc_border(self.net.layers[0])

    def clean(self, img_path=None):
        id = os.path.basename(img_path)[:-len('.png')]
        p = ImgPreprocessor(X_imgpath=img_path, border=self.border)
        patches = p._get_X_fast(modus='full')
        y = self.net.predict(patches)
        y2 = np.vstack(np.array(y).flatten())
        w, h = p.X_img.shape
        orig = np.resize(y2, (w,h)) * 255
        del p, patches
        return PIL.Image.fromarray(orig), id

    def clean_and_show(self, img_path=None):
        img, _ = self.clean(img_path)
        img.show()

    def clean_and_save(self, img_path=None, savepath=None):
        img, id = self.clean(img_path)
        img.convert('L').save(os.path.join(savepath, id + '.png'))
        img.close()

    def to_submission_format(self, cleaned_img=None, id=None):
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
      try:
        return self.net.meta
      except:
        return None

    def __calc_border(self, layer):
      return int((np.sqrt(layer.n_in) - 1) * 0.5)

