# 
# prepare.py
#   Prepare data for boostedcascade.
# 
# Author : Donny
# 

import numpy as np
import scipy.misc

import os
import math

def transformToData(image, wndW, wndH, padX, padY):
    """Scan the image and get subimage of size = [wndW, wndH],
       padding of each subimage is [padX, padY].
    """
    h, w = image.shape
    data = []
    for y in range(0,h-wndH,padY):
        for x in range(0,w-wndW,padX):
            data.append(image[y:y+wndH, x:x+wndW])
    return data

def transformToDataWithScale(image, wndW, wndH, padX, padY, min_size=0.0, max_size=1.0, step=0.5):
    height, width = image.shape
    data = []

    if min_size * min(width, height) < 24:
        min_size = 24.0 / min(width, height)
    assert max_size > min_size
    assert step > 0.0 and step < 1.0

    si = max_size
    while True:
        scaledimg = scipy.misc.imresize(image, size=(int(si*height), int(si*width)), mode='F')
        scaleddata = transformToData(scaledimg, wndW, wndH, padX, padY)
        data.extend(scaleddata)
        if si <= min_size: break
        si = si * step
        if si < min_size: si = min_size

    return data

def generateNonface(srcpath, destpath, imgsize, scanpad=(48,48)):
    """Generate non-face images in srcpath, and save to destpath.
    """
    if not isinstance(imgsize, tuple):
        raise ValueError("imgsize must be tuple")
    if not isinstance(scanpad, tuple):
        raise ValueError("scanpad must be tuple")

    for file_or_dir in os.listdir(srcpath):
        abs_srcpath = os.path.abspath(os.path.join(srcpath, file_or_dir))
        abs_destpath = os.path.abspath(os.path.join(destpath, file_or_dir))

        if os.path.isdir(abs_srcpath):
            os.makedirs(abs_destpath, exist_ok=True)
            generateNonface(abs_srcpath, abs_destpath, imgsize, scanpad)
        else:
            if file_or_dir.endswith('.jpg'):
                print('Processing non-face image %s' % file_or_dir)
                image = scipy.misc.imread(abs_srcpath, flatten=False, mode='F')
                outname, ext = os.path.splitext(abs_destpath)
                data = transformToDataWithScale(image, imgsize[0], imgsize[1], scanpad[0], scanpad[1])
                for ind in range(len(data)):
                    scipy.misc.imsave(outname + '-' + str(ind) + ext, data[ind])

def generateFace(srcpath, destpath, listpath, verbose=False):
    """Generate face images in srcpath, described by listfiles in listpath,
       and save to destpath.
    """
    cnt_miss = 0
    faceind = 0
    for facelist in os.listdir(listpath):
        if facelist.endswith('-ellipseList.txt'):
            print('Processing facelist %s' % facelist)
            abs_facelist = os.path.abspath(os.path.join(listpath,facelist))
            with open(abs_facelist) as f:
                allline = f.readlines()
                allline = [l.rstrip('\n') for l in allline]
                il = 0
                while il < len(allline):
                    imgfilename = allline[il]; il+=1
                    if not imgfilename:
                        break

                    if verbose: print('Processing face image %s.jpg' % imgfilename)
                    try:
                        image = scipy.misc.imread(os.path.join(srcpath,imgfilename + '.jpg'), mode='F')
                    except FileNotFoundError:
                        if verbose: print('Face image %s not found.' % imgfilename)
                        cnt_miss += 1
                        facecnt = int(allline[il]); il+=1+facecnt
                        continue
                        
                    height, width = image.shape
                    imgpad = int(max(height, width)/2)
                    image = np.pad(image, [(imgpad,imgpad), (imgpad,imgpad)],
                        mode='constant', constant_values=0) # Pad image

                    facecnt = int(allline[il]); il+=1
                    for i in range(facecnt):
                        ellipse = allline[il]; il+=1
                        major_radius, minor_radius, angle, ctx, cty, acc = \
                            map(float, ellipse.split())
                        radian = angle*math.pi/180
                        
                        # May get some noise around, but it's fine.
                        # There is noise when detecting.
                        h = int(math.cos(radian)*major_radius*2)
                        w = int(math.cos(radian)*minor_radius*2)
                        if h < w: h = w
                        else: w = h
                        y = int(cty - h/2 + imgpad)
                        x = int(ctx - w/2 + imgpad)

                        try:
                            outimgname = os.path.basename(imgfilename) + '-' + str(faceind) + '.jpg'
                            scipy.misc.imsave(os.path.join(destpath, outimgname),
                                image[y:y+h, x:x+w])
                            faceind += 1
                            if verbose: print('Face image %s generated.' % outimgname)
                        except Exception as expt:
                            print(expt)
                            print(x,y,w,h, ' ',width, height)
    
    print('Faces generation done with %d faces generated and %d faces lost.' % (faceind, cnt_miss))
    return faceind, cnt_miss

def stretchFace(srcpath, destpath, imgsize, verbose=False):
    """Stretch faces to 
    """
    print('Stretching faces...')
    for ognface in os.listdir(srcpath):
        if ognface.endswith('.jpg'):
            if verbose: print('Stretching face image %s' % ognface)
            image = scipy.misc.imread(os.path.join(srcpath, ognface), mode='F')
            image = scipy.misc.imresize(image, size=imgsize, mode='F') # TODO size=(height, width)
            scipy.misc.imsave(os.path.join(destpath, ognface), image)
    print('Face stretching done.')

def generateNoFaceFromFaceBk(srcpath, destpath, listpath, imgsize, scanpad, verbose=False):
    """Generate no-face images from the background of images contain faces.
    """
    os.makedirs(destpath, exist_ok=True)
    if not isinstance(imgsize, tuple):
        raise ValueError("imgsize must be tuple")
    if not isinstance(scanpad, tuple):
        raise ValueError("scanpad must be tuple")

    cnt_miss = 0
    nofaceind = 0
    for facelist in os.listdir(listpath):
        if facelist.endswith('-ellipseList.txt'):
            print('Processing facelist %s' % facelist)
            abs_facelist = os.path.abspath(os.path.join(listpath,facelist))
            with open(abs_facelist) as f:
                allline = f.readlines()
                allline = [l.rstrip('\n') for l in allline]
                il = 0
                while il < len(allline):
                    imgfilename = allline[il]; il+=1
                    if not imgfilename:
                        break

                    if verbose: print('Processing face image %s.jpg' % imgfilename)
                    try:
                        image = scipy.misc.imread(os.path.join(srcpath,imgfilename + '.jpg'), mode='F')
                    except FileNotFoundError:
                        if verbose: print('Face image %s not found.' % imgfilename)
                        cnt_miss += 1
                        facecnt = int(allline[il]); il+=1+facecnt
                        continue
                        
                    height, width = image.shape

                    facecnt = int(allline[il]); il+=1
                    for i in range(facecnt):
                        ellipse = allline[il]; il+=1
                        major_radius, minor_radius, angle, ctx, cty, acc = \
                            map(float, ellipse.split())
                        radian = angle*math.pi/180
                        
                        # May get some noise around, but it's fine.
                        # There is noise when detecting.
                        h = int(math.cos(radian)*major_radius*2)
                        w = int(math.cos(radian)*minor_radius*2)
                        if h < w: h = w
                        else: w = h
                        y = int(cty - h/2)
                        if y < 0: y = 0
                        elif y >= height: y = height - 1
                        x = int(ctx - w/2)
                        if x < 0: x = 0
                        elif x >= width: x = width - 1

                        image[y:y+h, x:x+w] = 0 # crop out face

                    outimgname = os.path.join(destpath, os.path.basename(imgfilename))
                    # data = transformToData(image, imgsize[0], imgsize[1], scanpad[0], scanpad[1])
                    data = transformToDataWithScale(image, imgsize[0], imgsize[1], scanpad[0], scanpad[1])
                    for ind in range(len(data)):
                        scipy.misc.imsave(outimgname + '-' + str(ind) + '.jpg', data[ind])
                        nofaceind += 1
    
    print('Faces generation done with %d no-faces generated and %d face images lost.' % (nofaceind, cnt_miss))
    return nofaceind, cnt_miss

if __name__ == '__main__':
    DetectWndW = 24
    DetectWndH = 24
    DetectPadX = 12
    DetectPadY = 12
    ScanPadX = 48
    ScanPadY = 48

    generateNonface('db/non-faces', 'data/non-faces-ex',
                    imgsize=(DetectWndW, DetectWndH),
                    scanpad=(ScanPadX, ScanPadY))
    # generateFace('db/faces', 'db/pure-faces', 'db/FDDB-folds')
    # stretchFace('db/pure-faces', 'data/faces',
    #             imgsize=(DetectWndW, DetectWndH))
    # generateNoFaceFromFaceBk('db/faces',
    #                          'data/non-faces/facebk-ex',
    #                          'db/FDDB-folds',
    #                          imgsize=(DetectWndW, DetectWndH),
    #                          scanpad=(ScanPadX, ScanPadY))
