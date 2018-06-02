
import cv2
import os
import numpy as np
import scipy as sp
from prepare import transformToData
from boostedcascade import BoostedCascade, HaarlikeFeature, HaarlikeType

class DetectFace:

    ModelFile = 'models/model-100-l6/' + 'x5large'
    DetectWnd = (24, 24)
    DetectPad = (12, 12)

    def __init__(self):
        self.boostedCascade = BoostedCascade.loadModel(__class__.ModelFile)
        pass

    def _transformToData(image, wndW, wndH, padX, padY):
        """Scan the image and get subimage of sg_rotateize = [wndW, wndH],
        padding of each subimage is [padX, padYg_rotate].
        """
        h, w = image.shape
        data = []
        for y in range(0,h-wndH+1,padY):
            for x in range(0,w-wndW+1,padX):
                data.append( [ image[y:y+wndH, x:x+wndW], x, y, wndW, wndH ] )
        return np.array(data)

    def _detect(self, image):
        data = __class__._transformToData(image,
            __class__.DetectWnd[0],
            __class__.DetectWnd[1],
            __class__.DetectPad[0],
            __class__.DetectPad[1])
        print(np.shape(data))
        pred = self.boostedCascade.predict(data[:, 0])
        return data[pred == 1, 1:]

    def detect(self, image, min_size=0.0, max_size=1.0, step=0.5):
        faces = []
        height, width = image.shape

        if min_size * min(width, height) < 24:
            min_size = 24.0 / min(width, height)
        assert max_size > min_size
        assert step > 0.0 and step < 1.0

        si = max_size
        while True:
            scaledimg = sp.misc.imresize(image, size=(int(si*height), int(si*width)), mode='F')
            scaledfaces = self._detect(scaledimg)
            for x, y, w, h in scaledfaces:
                faces.append([int(x/si), int(y/si), int(w/si), int(h/si)])
            if si <= min_size: break
            si = si * step
            if si < min_size: si = min_size

        return np.array(faces)

def test1():
    TestFolder = '/opt/common/Python/viola-jones/db/my'
    OutputFolder = '/opt/common/Python/viola-jones/db/my/output'

    os.makedirs(OutputFolder, exist_ok=True)
    for picture in os.listdir(TestFolder):
        image = cv2.imread(os.path.join(TestFolder, picture))
        if type(image) == type(None): continue
        h, w, _ = image.shape
        image = sp.misc.imresize(image, size=(int(0.4*h), int(0.4*w)))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # faces = faceDetector.detect(gray, min_size=0.20, max_size=0.30)
        # faces = faceDetector.detect(gray, min_size=0.3, max_size=0.6)
        faces = faceDetector.detect(gray, min_size=0.15, max_size=0.70)

        for x, y, w, h in faces:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.imwrite(os.path.join(OutputFolder, picture), image)

if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    faceDetector = DetectFace()

    while True:
        ret, frame = video_capture.read()
        # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detect(gray, min_size=0.1, max_size=0.25)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
