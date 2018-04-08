
import scipy.misc
from boostedcascade import BoostedCascade, HaarlikeFeature

im = scipy.misc.imread('data/1avril-lavigne.jpg', flatten=False, mode='F')

print(im.shape)
haarfeature = HaarlikeFeature()
features = haarfeature.extractFeatures(im)
