
from utils import loadImages
from boostedcascade import BoostedCascade, HaarlikeFeature

GenerateFeatures = False

if __name__ == '__main__':
    boostedCascade = BoostedCascade(0.1, 0.3, 0.6)

    if GenerateFeatures:
        faceImages = loadImages('data/micro/train/faces')
        nonfaceImages = loadImages('data/micro/train/non-faces')

        boostedCascade.prepare(faceImages, nonfaceImages, verbose=True)
        boostedCascade.savefeaturesdata('data/micro/train/features/micro')
    else:
        boostedCascade.loadfeaturesdata('data/micro/train/features/micro')
        boostedCascade.train(verbose=True)
        boostedCascade.saveModel('data/micro/model/micro')
