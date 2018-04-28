# 
# haarlikefeature.py
#   Extract haar-like features from images.
# 
# Author : Donny
# 

import numpy as np
from enum import Enum

class HaarlikeType(Enum):
    TWO_HORIZONTAL = 0
    TWO_VERTICAL = 1
    THREE_HORIZONTAL = 2
    THREE_VERTICAL = 3
    FOUR_DIAGONAL = 4
    TYPES_COUNT = 5

class HaarlikeFeature:
    """Extract haar-like features from images.
    """

    HaarWindow = [
        (2, 1),
        (1, 2),
        (3, 1),
        (1, 3),
        (2, 2)
    ]

    def __init__(self):
        # self.wnd_size = (0, 0)
        pass

    def determineFeatures(self, width, height):
        """Determine the features count while the window is (width, height),
           as well as giving the descriptions of each feature.

        Call this function before calling extractFeatures.

        Parameters
        ----------
        width : int
            The width of the window.
        height : int
            The height of the window.

        Returns
        -------
        features_cnt : int
            The features count while the window is (width, height).
        descriptions : list of shape = [features_cnt, [haartype, x, y, w, h]]
            The descriptions of each feature.
        """
        features_cnt = 0
        for haartype in range(HaarlikeType.TYPES_COUNT.value):
            wndx, wndy = __class__.HaarWindow[haartype]
            for x in range(0, width-wndx+1):
                for y in range(0, height-wndy+1):
                    features_cnt += int((width-x)/wndx)*int((height-y)/wndy)
        
        descriptions = np.zeros((features_cnt, 5))
        ind = 0
        for haartype in range(HaarlikeType.TYPES_COUNT.value):
            wndx, wndy = __class__.HaarWindow[haartype]
            for w in range(wndx, width+1, wndx):
                for h in range(wndy, height+1, wndy):
                    for x in range(0, width-w+1):
                        for y in range(0, height-h+1):
                            descriptions[ind] = [haartype, x, y, w, h]
                            ind += 1
        
        # print(features_cnt, descriptions.shape)
        # self.wnd_size = (height, width)
        return features_cnt, descriptions

    def extractFeatures(self, ognImage_, features_descriptions):
        """Extract features from an image.

        Please call determineFeatures first.

        Parameters
        ----------
        ognImage_ : array-like of shape = [height, width]
            The original image.

        Returns
        -------
        features : np.array of shape = [features_cnt, val]
        """
        ognImage = np.array(ognImage_)
        height, width = ognImage.shape

        # Call determineFeatures first.
        features_cnt = len(features_descriptions)
        descriptions = features_descriptions

        # assert (height, width) == self.wnd_size
        
        features = np.zeros((int(features_cnt)))

        itgImage = self._getIntegralImage(ognImage)

        cnt = 0
        for description in descriptions:
            features[cnt] = self._getFeatureIn(
                itgImage,
                HaarlikeType(description[0]), 
                description[1],
                description[2],
                description[3],
                description[4]
            )
            cnt += 1
        
        return features
        

    def _getIntegralImage(self, ognImage):
        """Get the integral image.

        Integral Image:
        + - - - - -        + -  -  -  -  -  -
        | 1 2 3 4 .        | 0  0  0  0  0  .
        | 5 6 7 8 .   =>   | 0  1  3  6 10  .
        | . . . . .        | 0  6 14 24 36  .
                           | .  .  .  .  .  .

        Parameters
        ----------
        _ognImage : np.array of shape = [height, width]
            The original image

        Returns
        -------
        itgImage : np.array of shape = [height+1, width+1]
            The integral image
        """
        h, w = ognImage.shape
        # print(w,h)
        itgImage = np.zeros((h+1, w+1))

        for y in range(1, h+1):
            for x in range(1, w+1):
                itgImage[y, x] = itgImage[y, x-1] + itgImage[y-1, x] - itgImage[y-1, x-1] + ognImage[y-1, x-1]

        return itgImage

    def _getSumIn(self, itgImage, x, y, w, h):
        """Get sum of image in rectangle [x, y, w, h]

        Parameters
        ----------
        itgImage : np.array of shape = [height+1, width+1]
            The integral image.
        x : int
            The starting column.
        y : int
            The starting row.
        w : int
            The width of the rectangle.
        h : int
            The height of the rectangle.

        Returns
        -------
        sum : int
            The sum of the pixels in the rectangle, excluding column w and row h.
        """
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        return itgImage[h, w] + itgImage[y, x] - (itgImage[y, w] + itgImage[h, x])

    def _getFeatureIn(self, itgImage, feature_type, x, y, w, h):
        """Get haar feature in rectangle [x, y, w, h]

        Parameters
        ----------
        itgImage : np.array
            The integral image.
        feature_type : {HaarlikeType, number of HaarlikeType id}
            Tpye of the haar-like feature to extract.
        x : int
            The starting column.
        y : int
            The starting row.
        w : int
            The width of the rectangle.
        h : int
            The height of the rectangle.

        Returns
        -------
        diff : int
            The difference of white and black area, which represent the feature of the rectangle.
        """
        if not isinstance(feature_type, HaarlikeType):
            feature_type = HaarlikeType(feature_type)
            
        white = 0
        black = 0
        if feature_type == HaarlikeType.TWO_HORIZONTAL:
            white = self._getSumIn(itgImage, x, y, w/2, h)
            black = self._getSumIn(itgImage, x + w/2, y, w/2, h)
        elif feature_type == HaarlikeType.TWO_VERTICAL:
            white = self._getSumIn(itgImage, x, y, w, h/2)
            black = self._getSumIn(itgImage, x, y + h/2, w, h/2)
        elif feature_type == HaarlikeType.THREE_HORIZONTAL:
            white = self._getSumIn(itgImage, x, y, w/3, h) + self._getSumIn(itgImage, x + w*2/3, y, w/3, h)
            black = self._getSumIn(itgImage, x + w/3, y, w/3, h)
        elif feature_type == HaarlikeType.THREE_VERTICAL:
            white = self._getSumIn(itgImage, x, y, w, h/3) + self._getSumIn(itgImage, x, y + h*2/3, w, h/3)
            black = self._getSumIn(itgImage, x, y + h/3, w, h/3)
        elif feature_type == HaarlikeType.FOUR_DIAGONAL:
            white = self._getSumIn(itgImage, x, y, w/2, h/2) + self._getSumIn(itgImage, x + w/2, y + h/2, w/2, h/2)
            black = self._getSumIn(itgImage, x + w/2, y, w/2, h/2) + self._getSumIn(itgImage, x, y + h/2, w/2, h/2)
        
        return white - black
