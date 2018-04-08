
import numpy as np
import scipy.misc

import os
import math

def loadImages(path):
    images = []
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        if os.path.isdir(abs_path):
            images = images + loadImages(path)
        else:
            img = scipy.misc.imread(abs_path, flatten=False, mode='F')
            images.append(img)
    return images
