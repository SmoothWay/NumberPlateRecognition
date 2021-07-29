import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model
#from .TextImageGenerator import TextImageGenerator
import os, pathlib, sys
import itertools

NUMBER_PLATE_DIR = os.path.join(pathlib.Path(__file__).parent.absolute(), "./")
sys.path.append(NUMBER_PLATE_DIR)

class OCR:
    @classmethod
    def get_classname(cls):
        return cls.__name__

    def __init__(self):
        # Input parameters
        self.IMG_H = 64
        self.IMG_W = 128
        self.IMG_C = 1

        # Train parameters
        self.BATCH_SIZE = 32
        self.EPOCHS = 1

        # Network parameters
        self.CONV_FILTERS = 16
        self.KERNEL_SIZE = (3, 3)
        self.POOL_SIZE = 2
        self.TIME_DENSE_SIZE = 32
        self.RNN_SIZE = 512
        self.ACTIVATION = 'relu'
        self.DOWNSAMPLE_FACROT = self.POOL_SIZE * self.POOL_SIZE

        self.INPUT_NODE = "the_input_{}:0".format(type(self).__name__)
        self.OUTPUT_NODE = "softmax_{}".format(type(self).__name__)
        
        # callbacks hyperparameters
        self.REDUCE_LRO_N_PLATEAU_PATIENCE = 3
        self.REDUCE_LRO_N_PLATEAU_FACTOR   = 0.1

    def decode_batch(self, out):
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = ''
            for c in out_best:
                if c < len(self.letters):
                    outstr += self.letters[c]
            ret.append(outstr)
        return ret

    def normalize(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (self.IMG_W, self.IMG_H))

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        img = img.astype(np.float32)
        img -= np.amin(img)
        img /= np.amax(img)
        img = [[[h] for h in w] for w in img.T]

        x = np.zeros((self.IMG_W, self.IMG_H, 1))
        x[:, :, :] = img
        return x

    def predict(self, imgs, return_acc=False):
        Xs = []
        for img in imgs:
            x = self.normalize(img)
            Xs.append(x)
        pred_texts = []
        if bool(Xs):
            if len(Xs) == 1:
                net_out_value = self.MODEL.predict_on_batch(np.array(Xs))
            else:
                net_out_value = self.MODEL(np.array(Xs), training=False)
            pred_texts = self.decode_batch(net_out_value)
        if return_acc:
            return pred_texts, net_out_value
        return pred_texts

    def load(self, verbose = 0):
        path_to_model = NUMBER_PLATE_DIR + 'models\TextDetector\kz\Anpr_ocr_kz_2020_08_26_tensorflow_v2.h5'
        self.MODEL = load_model(path_to_model, compile=False)

        net_inp = self.MODEL.get_layer(name='{}'.format(self.MODEL.layers[0].name)).input
        net_out = self.MODEL.get_layer(name='{}'.format(self.MODEL.layers[-1].name)).output
        self.MODEL = Model(inputs=net_inp, outputs=net_out)
        if verbose:
            self.MODEL.summary(        )

        return self.MODEL