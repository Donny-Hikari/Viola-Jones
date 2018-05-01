
import os
import math
import sys

import numpy as np
import scipy.misc
from sklearn.utils import shuffle as skshuffle

def loadImages(path, verbose=False):
    images = []
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        if os.path.isdir(abs_path):
            images = images + loadImages(abs_path, verbose)
        else:
            img = scipy.misc.imread(abs_path, flatten=False, mode='F')
            images.append(img)
            if verbose:
                sys.stdout.write("\r\033[K") # Clear line
                print('%d images loaded.' % len(images), end='', flush=True)
    if verbose: sys.stdout.write("\r\033[K") # Clear line
    return images

def saveNegativeResult(nofaceimgs, pred_n, fp_outdir, tn_outdir, tn_fp_rate):
    os.makedirs(fp_outdir, exist_ok=True)
    os.makedirs(tn_outdir, exist_ok=True)

    fp_imgs = []
    tn_imgs = []
    for ind, r in enumerate(pred_n.astype(int)):
        if r == 1: fp_imgs.append(nofaceimgs[ind])
        else: tn_imgs.append(nofaceimgs[ind])

    for ind, fp_img in enumerate(fp_imgs):
        scipy.misc.imsave(os.path.join(fp_outdir, 'fp-noface-' + str(ind) + '.jpg'), fp_img)

    n_tn = min(len(tn_imgs), int(tn_fp_rate*len(fp_imgs)))
    tn_imgs = skshuffle(tn_imgs)[:n_tn]
    for ind, tn_img in enumerate(tn_imgs):
        scipy.misc.imsave(os.path.join(tn_outdir, 'tn-noface-' + str(ind) + '.jpg'), tn_img)
