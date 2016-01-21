import cPickle
import PIL
import numpy as np

from network import Network
from preprocessor import ImgPreprocessor

class CleaningEngine(object):

    def __init__(self, model_path=None, border=None):
        f = open(model_path, 'rb')
        self.net = cPickle.load(f)
        self.border = border

    def clean(self, img_path=None):
        p = ImgPreprocessor(X_imgpath=img_path, border=self.border)
        patches = p._get_X_fast(modus='full')
        y = self.net.predict(patches)
        y2 = np.vstack(np.array(y).flatten())
        h, w = p.X_img.size
        orig = np.resize(y2, (w,h)) * 255
        return PIL.Image.fromarray(orig)

    def clean_and_show(self, img_path=None):
        img = self.clean(img_path)
        img.show()

    def clean_and_save(self, img_path=None, savepath=None):
        img = self.clean(img_path)
        img.convert('L').save(savepath, 'PNG')

