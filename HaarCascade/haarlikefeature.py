
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

    HaarWindow = [
        (2, 1),
        (1, 2),
        (3, 1),
        (1, 3),
        (2, 2)
    ]

    def __init__(self):
        pass

    def extractFeatures(self, _ognImage):
        """Extract features from an image.

        Parameters
        ----------
        _ognImage : array-like of shape = [height, width]
            The original image.

        Returns
        -------
        features : np.array of shape = [features_cnt, (x, y, w, h)]
        """
        
        ognImage = np.array(_ognImage)
        height, width = ognImage.shape
        
        features_cnt = 0
        for haartype in range(HaarlikeType.TYPES_COUNT.value):
            wndx, wndy = __class__.HaarWindow[haartype]
            for x in range(0, width-wndx+1):
                for y in range(0, height-wndy+1):
                    features_cnt += int((width-x)/wndx)*int((height-y)/wndy)
        print(features_cnt)
        features = np.zeros((int(features_cnt), 5))

        itgImage = self._getIntegralImage(ognImage)

        cnt = 0
        for haartype in range(HaarlikeType.TYPES_COUNT.value):
            wndx, wndy = __class__.HaarWindow[haartype]
            print('Iterating on haar-like type',haartype,':',HaarlikeType(haartype))
            for x in range(0, width-wndx+1):
                for y in range(0, height-wndy+1):
                    for w in range(wndx, width-x+1, wndx):
                        for h in range(wndy, height-y+1, wndy):
                            val = self._getFeatureIn(HaarlikeType(haartype), itgImage, x, y, w, h)
                            features[cnt] = (val, x, y, w, h)
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
        print(w,h)
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

    def _getFeatureIn(self, feature_type, itgImage, x, y, w, h):
        """Get haar feature in rectangle [x, y, w, h]

        Parameters
        ----------
        feature_type : HaarlikeType
            Tpye of the haar-like feature to extract.
        itgImage : np.array
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
        diff : int
            The difference of white and black area, which represent the feature of the rectangle.
        """
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
