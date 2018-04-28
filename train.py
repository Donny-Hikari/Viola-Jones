
from utils import loadImages
from boostedcascade import BoostedCascade, HaarlikeFeature

GenerateFeatures = True
Database = 'xxxlarge'
ModelFile = 'data/' + Database + '/model-100/' + Database

if __name__ == '__main__':
    # boostedCascade = BoostedCascade(0.03, 0.40, 0.99)
    # boostedCascade = BoostedCascade(0.04, 0.20, 0.985)
    boostedCascade = BoostedCascade(0.07, 0.60, 0.97)
    boostedCascade = BoostedCascade.loadModel(ModelFile)

    if GenerateFeatures:
        # faceImages = loadImages('data/' + Database + '/train/faces')
        # nonfaceImages = loadImages('data/' + Database + '/train/non-faces')
        nonfaceImages = loadImages('data/' + Database + '/train/tn-nofaces')

        boostedCascade.prepareex(nonfaceImages, False, is_append=True, verbose=True)
        # boostedCascade.prepare(faceImages, nonfaceImages, shuffle=True, verbose=True)
        boostedCascade.savefeaturesdata('data/' + Database + '/train/features/' + Database)
    else:
        print('Loading model...')
        boostedCascade = BoostedCascade.loadModel(ModelFile)
        print('Loading data...')
        boostedCascade.loadfeaturesdata('data/' + Database + '/train/features/' + Database)
        print('Trainning...')
        boostedCascade.train(is_continue=True, autosnap_filename=ModelFile, verbose=True)
        boostedCascade.saveModel(ModelFile)
