
import numpy as np
from utils import loadImages
from boostedcascade import BoostedCascade, HaarlikeFeature, HaarlikeType

RawPredict = False
GenerateFeatures = False
Database = 'middle'
ModelFile ='data/' + Database + '/model-60/' + Database

if __name__ == '__main__':
    boostedCascade = BoostedCascade.loadModel(ModelFile)

    # faceImages = loadImages('data/' + Database + '/test/faces')
    # nonfaceImages = loadImages('data/' + Database + '/test/non-faces')
    faceImages = loadImages('data/' + 'large' + '/train/faces')
    nonfaceImages = loadImages('data/' + 'large' + '/train/non-faces')

    if RawPredict:
        if GenerateFeatures:
            boostedCascade.preparePredictRaw(faceImages, nonfaceImages, verbose=True)
            boostedCascade.savefeaturesdata('data/' + Database + '/test/features/' + Database)
        
        boostedCascade.loadfeaturesdata('data/' + Database + '/test/features/' + Database)
        yPredP, yPredN = boostedCascade.predictRaw()

        d = np.sum(yPredP == 1) / len(boostedCascade.P)
        f = np.sum(yPredN == 1) / len(boostedCascade.N)
    else:
        yPredP = boostedCascade.predict(faceImages)
        d = np.sum(yPredP == 1) / len(faceImages)
        yPredN = boostedCascade.predict(nonfaceImages)
        f = np.sum(yPredN == 1) / len(nonfaceImages)
        
    print('Detection rate: %f; False positive rate: %f' % (d, f))
