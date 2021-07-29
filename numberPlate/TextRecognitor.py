import sys, os
import numpy
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import TextRecognitors


def nplate_split(imgs, lines, coef=0.1):
    result_imgs = []
    for image, line in zip(imgs, lines):
        if line < 2:
            result_imgs.append(image)
        else:
            n = int(image.shape[0]/line + (image.shape[0] * coef))
            fPart = image[:n]
            for l in range(1, line - 1):
                mPart = image[:l*n]
                fPart  = numpy.concatenate((fPart, mPart), axis=1)
            lPart  = image[-n:]
            result_imgs.append(numpy.concatenate((fPart, lPart), axis=1))
    return result_imgs

class TextRecognitor:
    @classmethod
    def get_classname(cls):
        return cls.__name__

    def __init__(self, mode="auto"):
        self.detect_map = {'kz': 0}
        self.detectors = []
        self.detect_names = []

        self.LABEL_DEFAULT = "kz"
        self.LINES_COUNT_DEFAULT = 1

        _label = 'kz'
        TextPostprocessing = getattr(getattr(TextRecognitors, _label), _label)
        detector = TextPostprocessing()

        detector.load()
        self.detectors.append(detector)
        self.detect_names.append(_label)

    def predict(self, zones, return_acc=False):
        labels = []
        lines = []

        while len(labels) < len(zones):
            labels.append(self.LABEL_DEFAULT)
        while len(lines) < len(zones):
            lines.append(self.LINES_COUNT_DEFAULT)

        zones = nplate_split(zones, lines)
        predicted = {}
        orderAll = []
        resAll = []
        i = 0
        scores = []
        for zone, label in zip(zones, labels):
            detector = self.detect_map[label]
            if detector not in predicted.keys():
                predicted[detector] = {"zones": [], "order": []}
            predicted[detector]["zones"].append(zone)
            predicted[detector]["order"].append(i)
            i += 1

        for key in predicted.keys():
            resAll = resAll + self.detectors[int(key)].predict(predicted[key]["zones"], return_acc=return_acc)
            orderAll = orderAll + predicted[key]["order"]

        return [x for _, x in sorted(zip(orderAll, resAll), key=lambda pair: pair[0])]

    @staticmethod
    def get_static_module(name):
        return getattr(getattr(TextRecognitors, name), name)

    def get_module(self, mname):
        m_index = self.detect_names.index(mname)
        return self.detectors[m_index]