#!/usr/bin/python
 
__author__ = ('David Dunn')
__version__ = '0.1'

import cv2, math
import numpy as np
import dCamera as dc
from scipy.interpolate import griddata
from scipy.signal import medfilt
#from scipy.misc import imsave

def calcWhite(dtype):
    ''' Get the value of white for the given type '''
    if issubclass(dtype, np.integer):
        return np.iinfo(dtype).max
    return 1

class Display(object):
    ''' A class that defines the physical properties of a display '''
    def __init__(self, resolution=(1080,1920), size=(.071,.126), bezel=(.005245,.01)):  # default to Samsung Note 3
        self.resolution = resolution
        self.size = size
        self.bezel = bezel
        self.depth = None
    @property
    def width(self):
        return self.resolution[1]
    @property
    def height(self):
        return self.resolution[0]
    def pixelSize(self):
        return (self.size[0]/self.resolution[0], self.size[1]/self.resolution[1])

def calibrateDisplay(res, window, camera=None):
    '''Use gray codes and camera to calibrate a non-planar display'''
    images = genGreyCodes(res)
    #cameraImages = captureGreyCodes(images, window, camera)
    LUTs, mask = evalGreyCodes(images, cameraImages)
    LUTs = reverseMaps(LUTs, mask, res)
    #cv2.remap()

def genGreyCodes(resolution,dtype=np.uint8):
    ''' Generate Gray codes for vertical and horizontal stripe patterns.
    See: http://en.wikipedia.org/wiki/Gray_code 
    Will return 3 lists of images: black & white, horizontal grey codes, vertical grey codes'''
    res = np.copy(resolution)
    #res = np.flipud(res)
    Nary = np.ceil(np.log2(res)).astype(np.int32)
    offset = np.floor((np.exp2(Nary)-res)/2).astype(np.int32)
    flipFunc = [np.flipud,np.fliplr]
    images = []
    # Get zero and one frames for levels
    black = np.zeros(res,dtype)
    white = np.ones(res,dtype) * calcWhite(dtype)
    images.append([black, white])
    for idx, N in enumerate(Nary):          # vertical and horizontal
        images.append([])
        for K in range(N-1,0,-1):           # iterate over each image
            size = np.copy(res)
            size[idx] = 2**K
            black = np.zeros(size,dtype)
            white = np.ones(size,dtype) * calcWhite(dtype)
            img = np.concatenate((black,white),axis=idx)    # concat black and white
            for i in range(N-K-1):
                img = np.concatenate((img, flipFunc[idx](img)),axis=idx)        # concat previous and flip of previous
            if idx == 0:                                                        # crop to proper size for output
                images[idx+1].append(img[offset[idx]:res[idx]+offset[idx],:])
            else:
                images[idx+1].append(img[:,offset[idx]:res[idx]+offset[idx]])
    #images.reverse()
    return images

def captureGreyCodes(displayImages, window, cam=None):
    ''' Will synchronously display and capture the images given 
    Will return lists of images as passed in '''
    cam = dc.Camera(0) if cam is None else cam
    #TODO: write this, but it really is dependant on your setup, so...

def evalGreyCodeCameraSpace(images, resolution, maskFilter=7, maskThresh=65, imageFilter=3, imageThresh=100):
    ''' From the set of images captured from a camera, create the lookup table for warping
    Will return a list of maps and a mask for the display'''
    res = np.copy(resolution)
    res = np.flipud(res)
    Nary = np.ceil(np.log2(res)).astype(np.int32)
    offset = np.floor((np.exp2(Nary)-res)/2).astype(np.int32)
    # do some unpacking setup
    levels = images[0]
    black = dc.toGray(levels[0])
    white = dc.toGray(levels[1])
    greyImages = images[1:]
    maps = []
    # We need to determine the region of the display in the camera
    mask = cv2.subtract(white,black) > maskThresh
    for idx, N in enumerate(Nary):                  # vertical and horizontal
        maps.append(np.zeros_like(black,dtype=np.uint16))
        previous = np.zeros_like(mask)
        for img, K in enumerate(range(N-1,0,-1)):   # loop through pics, convert grey code to binary, and add the result
            grey = np.bitwise_and(white,dc.toGray(greyImages[idx][img]))[:,:] > imageThresh
            if 2**K > imageFilter:
                grey = medfilt(grey,imageFilter).astype(np.bool)
            grey = np.logical_and(grey, mask)               # reduce to screen area
            previous = np.logical_xor(grey, previous)       # convert to binary
            maps[idx] += previous.astype(np.uint16) * 2**K  # convert to decimal
    # subtract offest from map
    for idx, off in enumerate(offset):
        maps[idx] -= off
        maps[idx] = np.clip(maps[idx], 0, res[idx]-1)
    maps = np.asarray(maps)
    mask = medfilt(mask,maskFilter).astype(np.bool)
    return maps, mask

def toDisplaySpace(maps, mask, resolution, angleMap=None, filter=5):
    ''' given the maps and mask from the camera of the display this will create a lookup table to reverse the function '''
    # transpose the directionality of the maps given
    res = np.copy(resolution)
    res = np.flipud(res)
    displayMaps = []
    if angleMap is None:
        indices = np.mgrid[0:maps[0].shape[0],0:maps[0].shape[1]].astype(np.uint16)
        indices = np.dstack((indices[0],indices[1]))
    else:
        indices = np.dstack((angleMap[:,:,1],angleMap[:,:,0])) 
    # unroll two maps and one meshgrid using mask to cull
    flatmask = mask.flatten()
    grid_x, grid_y = np.mgrid[0:resolution[0],0:resolution[1]]
    mapx = maps[1].flatten()[flatmask]
    mapy = maps[0].flatten()[flatmask]
    for i in range(2):
        idx = indices[:,:,i].flatten()[flatmask]
        grid_z = griddata((mapx,mapy), idx, (grid_x, grid_y), method='linear').astype(np.float32)
        out = np.transpose(grid_z)
        out = np.fliplr(out)
        # filter the result using medianBlur
        if filter > 2:
            out = medfilt(out, filter)
        displayMaps.append(out)
    displayMaps = np.asarray(displayMaps)
    return displayMaps

def generateUVluts(displayMaps):
    ''' given a set of display maps (as angle values) find the largest and smallest angles and '''
    gXmin = float('inf')
    gXmax = float('-inf')
    gYmin = float('inf')
    gYmax = float('-inf')
    images = []
    for displayMap in displayMaps:
            xmin = displayMap[0][~np.isnan(displayMap[0])].min()
            xmax = displayMap[0][~np.isnan(displayMap[0])].max()
            ymin = displayMap[1][~np.isnan(displayMap[1])].min()
            ymax = displayMap[1][~np.isnan(displayMap[1])].max()
            gXmin = min(gXmin,xmin)
            gYmin = min(gYmin,ymin)
            gXmax = max(gXmax,xmax)
            gYmax = max(gYmax,ymax)
    xsize = gXmax-gXmin
    ysize = gYmax-gYmin
    for displayMap in displayMaps:
        displayMap[0] -= gXmin
        displayMap[0] *= (1/xsize)
        displayMap[0] = 1. - displayMap[0]
        displayMap[1] -= gYmin
        displayMap[1] *= (1/ysize)
        displayMap[1] = 1. - displayMap[1]
        image = np.dstack((displayMap[1],displayMap[0],np.zeros_like(displayMap[0])))
        image[np.isnan(image)] = 0.
        images.append(image)
    print 'X range is: %s through %s'%(math.degrees(gXmin),math.degrees(gXmax))
    print 'Y range is: %s through %s'%(math.degrees(gYmin),math.degrees(gYmax))
    size = displayMaps[0].shape
    print size
    angleMap = np.mgrid[0:size[1],0:size[2]].astype(np.float32)
    print angleMap.shape
    angleMap[0] /= size[1]-1.0
    angleMap[0] *= xsize
    angleMap[0] += gXmin
    angleMap[1] /= size[2]-1.0
    angleMap[1] *= ysize
    angleMap[1] += gYmin
    angleMap = np.dstack((angleMap[1],angleMap[0],np.zeros_like(angleMap[0])))
    return images, angleMap
    #imsave('map.png',image)


def centralMonotonic(array,mask,axis=0):
    #this is stupid - might as well use a medfilt
    new = np.copy(array)
    if axis==0:
        top = np.copy(new[:new.shape[0]/2,:])
        topDiff = abs(np.diff(top.astype(np.int32),axis=axis)) > 10
        top[topDiff] = top.max()
        topMin = np.flipud(np.minimum.accumulate(np.flipud(top)))
        #top = np.maximum(new[:new.shape[0]/2,:],topMin)
        bot = np.copy(new[new.shape[0]/2:,:])
        botDiff = abs(np.diff(bot.astype(np.int32),axis=axis)) > 10
        bot[botDiff] = bot.min()
        botMax = np.maximum.accumulate(bot)
        #bot = np.minimum(new[new.shape[0]/2:,:],botMax)
        clean = np.vstack((topMin,botMax))
    elif axis==1:
        left = np.copy(new[:,0:new.shape[1]/2])
        leftDiff = abs(np.diff(left.astype(np.int32),axis=axis)) > 10
        left[leftDiff] = left.min()
        leftMax = np.fliplr(np.maximum.accumulate(np.fliplr(left),axis=1))
        right = np.copy(new[:,new.shape[1]/2:])
        rightDiff = abs(np.diff(right.astype(np.int32),axis=axis)) > 10
        right[rightDiff] = right.max()
        rightMin = np.minimum.accumulate(right,axis=1)
        clean = np.hstack((leftMax,rightMin))
    clean[~mask] = np.copy(array)[~mask]
    return clean

def checkUp():
    clean = cv2.imread('cleaned_0020.png',-1)
    clean = cv2.cvtColor(clean,cv2.COLOR_BGR2RGB)
    img = np.load('0020.npy')
    m = clean != 0
    new = np.zeros_like(img)
    new[m] = img[m]
    np.save(new, 'new_0000.npy')
    img[:,:,:] = new[:,:,:]

    test = img[:-1,:] - img[1:,:]
    n = test != 0
    test = img[:,:-1] - img[:,1:]
    m = test != 0
    p = np.zeros_like(img)
    p[1:,:,0] = n[:,:,0]
    p[:,1:,1] = m[:,:,1]

    new = medfilt(img[:,:,0], [5,1])
