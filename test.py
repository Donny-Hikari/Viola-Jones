
import numpy as np
from utils import loadImages
from boostedcascade import BoostedCascade, HaarlikeFeature, HaarlikeType

RawPredict = False

if __name__ == '__main__':
    boostedCascade = BoostedCascade.loadModel('data/micro/model/micro')

    faceImages = loadImages('data/micro/test/faces')
    nonfaceImages = loadImages('data/micro/test/non-faces')

    if RawPredict:
        # boostedCascade.preparePredictRaw(faceImages, nonfaceImages, True)
        # boostedCascade.savefeaturesdata('data/micro/test/features/micro')
        boostedCascade.loadfeaturesdata('data/micro/test/features/micro')
        yPredP, yPredN = boostedCascade.predictRaw()

        d = np.sum(yPredP == 1) / len(boostedCascade.P)
        f = np.sum(yPredN == 1) / len(boostedCascade.N)
    else:
        yPred = boostedCascade.predict(faceImages)
        d = np.sum(yPred == 1) / len(faceImages)
        yPred = boostedCascade.predict(nonfaceImages)
        f = np.sum(yPred == 1) / len(nonfaceImages)
        
    print('Detection rate: %f; False positive rate: %f' % (d, f))
