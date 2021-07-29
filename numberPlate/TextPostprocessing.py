import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import TextPostprocessings

def textPostprocessing(texts, textPostprocessNames):
    resTexts = []
    for text, textPostprocessName in zip(texts, textPostprocessNames):
        _textPostprocessName = textPostprocessName.replace("-", "_")
        if _textPostprocessName in dir(TextPostprocessings):
            TextPostprocessing = getattr(getattr(TextPostprocessings, _textPostprocessName), _textPostprocessName)
        else:
            TextPostprocessing = getattr(getattr(TextPostprocessings, "kz"), "kz")
        postprocessManager = TextPostprocessing()
        resTexts.append(postprocessManager.find(text))
    return resTexts
