
import scipy.misc
from HaarCascade import HaarlikeFeature
from HaarCascade.adaboost import AdaBoostClassifier

im = scipy.misc.imread('data/1avril-lavigne.jpg', flatten=False, mode='F')

print(im.shape)
haarfeature = HaarlikeFeature()
features = haarfeature.extractFeatures(im)
