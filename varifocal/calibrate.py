#!/usr/bin/python
 
__author__ = ('David Dunn')
__date__ = '2016-09-05'
__version__ = '0.1'

import cv2, sys, time, os, fnmatch, ctypes
import numpy as np
from scipy.misc import imsave
from scipy.interpolate import interp1d
from scipy.signal import medfilt
#from main import runDisplay
#import expglobals, expgui, explogic
#from exptools import ImageTools
from multiprocessing import Process, Value, Array
import dDisplay as dd
import dCamera as dc
import dGraph as dg
import dGraph.ui as ui
import dGraph.textures as dgt
import dGraph.util as dgu
from optics_correction import mapping

WINDOWS = [
    {
    "name": 'HMD Left',
    #"location": (0, 0),
    "location": (3266, 1936), # px coordinates of the startup screen for window location
    "size": (830, 800), # px size of the startup screen for centering
    "center": (290,216), # center of the display
    "refresh_rate": 60, # refreshrate of the display for precise time measuring
    "px_size_mm": 0.09766, # px size of the display in mm
    "distance_cm": 20, # distance from the viewer in cm,
    #"is_hmd": False,
    #"warp_path": 'data/calibration/newRight/',
    },
]
DISPLAY = 0
CAMERA = (None)
RESOLUTION = (768,1024)
POINT = (384,512)
DEPTH = 20
#DEPTH = 100
SIDES = {0:'left_masked'}#,1:'newLeft'}
SIDE = SIDES[DISPLAY]
CORNERS = (10,7)
DIRECTORY = './dDisplay/depth/data'
CAMCALIB = os.path.join(DIRECTORY,'cameraCalibration')
GLWARP = False
#IMAGE = 'data/images/circleCheck.png'
#IMAGE = 'data/images/airplane.png'
#IMAGE = 'data/images/smallC.png'
#IMAGE = 'data/images/smallBox.png'
IMAGE = 'data/images/pattern1.png'
#IMAGE = 'data/images/rightRenders/20cm.png'
#IMAGE = 'data/images/rightRenders/201cm.png'
#IMAGE = 'data/images/leftRenders/20cm.png'
#IMAGE = 'data/images/leftRenders/201cm.png'
TRIGGER = False

def wait(window):
    global TRIGGER
    while not (TRIGGER and not ui.window_should_close(window)):
        ui.swap_buffers(window)
        ui.poll_events()
    TRIGGER = False

def keyCallback(window, key, scancode, action,  mods):
    if action == ui.PRESS and key == ui.KEY_ESCAPE:
        ui.set_window_should_close(window, True)
    if action == ui.PRESS and key == ui.KEY_ENTER:
        global TRIGGER
        TRIGGER = True
    print key

def initDisplay():
    global DISPLAY
    ui.init()
    #open windows
    windows = []
    # Control window with informational display
    windows.append(ui.open_window('Calibration',1500,50,640,480,None,keyCallback))
    # HMD Window
    window = WINDOWS[DISPLAY]
    windowName = window["name"]
    winLocation = window["location"]
    winSize = window["size"]
    windows.append(ui.open_window(windowName,winLocation[0],winLocation[1],winSize[0],winSize[1],windows[0],keyCallback))
    ui.make_context_current(windows[1])
    return windows

def initCamera():
    global CAMERA, CAMCALIB
    #init camera
    ptg = dc.Camera(*CAMERA)
    #ptg.claibrate(CORNERS)
    #files = [f for f in os.listdir(CAMCALIB) if os.path.isfile(os.path.join(CAMCALIB, f))]
    #images = [cv2.imread(os.path.join(CAMCALIB, f)) for f in files]
    #ret, matrix, dist = dc.calibrate(images, CORNERS, **kwargs)
    #print ret
    #ptg.matrix = matrix
    #ptg.dist = dist
    ptg.matrix = np.array( [[  2.88992746e+03,   0.00000000e+00,   1.98587472e+03],
                            [  0.00000000e+00,   2.88814288e+03,   1.46326356e+03],
                            [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
    ptg.distortion = np.array([[-0.04156773, -0.02731505, -0.00368037, -0.00125477,  0.02314031]])
    return ptg

def captureWarpInteractive(windows):
    #generate greycodes
    winSize = ui.get_window_size(windows[1])
    codes = dd.genGreyCodes(winSize)
    texture = dgt.createTexture(codes[0][0])
    texture2 = dgt.createTexture(np.zeros((480,640,3),dtype=np.float32))
    # set the focus
    dgt.updateTexture(texture,codes[1][5])
    ui.make_context_current(windows[1])
    dgu.drawQuad(texture)
    ui.swap_buffers(windows[1])
    dgu.drawQuad(texture)
    ui.swap_buffers(windows[1])
    wait(windows[1])
    #display grey codes and allow for capture
    for idx, sequence in enumerate(codes):
        seq = []
        for idx2, image in enumerate(sequence):
            dgt.updateTexture(texture,image)
            ui.make_context_current(windows[1])
            dgu.drawQuad(texture)
            ui.swap_buffers(windows[1])
            wait(windows[1])
            #capture image


def captureWarp(id,currentDepth,requestedDepth,stillRunning,*args):
    global DEPTH, DISPLAY, SIDE, DIRECTORY
    WINDOWS = initDisplay()
    #generate greycodes
    winSize = ui.get_window_size(WINDOWS[1])
    codes = dd.genGreyCodes(winSize)
    texture = dgt.createTexture(codes[0][0])
    texture2 = dgt.createTexture(np.zeros((480,640,3),dtype=np.float32))
    captures = []
    #set display depth
    requestedDepth[DISPLAY].value = DEPTH
    ptg = initCamera()
    expgui.updateTexture(texture,codes[1][5])
    glfw.make_context_current(WINDOWS[1])
    expgui.drawQuad(texture)
    glfw.swap_buffers(WINDOWS[1])
    expgui.drawQuad(texture)
    glfw.swap_buffers(WINDOWS[1])
    check = ptg.captureFrames()
    check = ptg.undistort(check)
    directory = '%s/%s/%04d'%(DIRECTORY,SIDE,DEPTH)
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite('%s/check.png'%directory, check[len(check)-1])
    #display grey codes and allow for capture
    for idx, sequence in enumerate(codes):
        seq = []
        for idx2, image in enumerate(sequence):
            expgui.updateTexture(texture,image)
            glfw.make_context_current(WINDOWS[1])
            expgui.drawQuad(texture)
            glfw.swap_buffers(WINDOWS[1])
            #time.sleep(1)
            #capture image
            for i in range(10):
                ret, frame = ptg.read()
            expgui.updateTexture(texture2,frame)
            glfw.make_context_current(WINDOWS[0])
            expgui.drawQuad(texture2)
            glfw.swap_buffers(WINDOWS[0])
            frame = ptg.undistort([frame,])[0]
            seq.append(frame)
            cv2.imwrite('%s/capture_%02d_%02d.png'%(directory,idx,idx2),frame)
        captures.append(ptg.undistort(seq))
    #process captures
    #cameraMas, mask = dd.evalGreyCodeCameraSpace(captures, winSize)
    #displayMaps = dd.toDisplaySpace(cameraMaps, mask, winSize,angleMap=)
    
    #cv2.imwrite('warp.png',map)
            
    stillRunning.value = False
    glfw.terminate()

def genAngleMap(depth, side,checkSize,checkPixels=[20,20],width=10,roi=[0,0,768,1024],write=True,k_harris=[2,3,0.04]):
    global DIRECTORY
    # Input and output locations of the calibration image.
    fn_cal                     = '%s/%s/%04d/check.JPG'%(DIRECTORY,side,depth)
    img                        = cv2.imread(fn_cal, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    fnmapimg                   = '%s/%s/%04d/angles.png'%(DIRECTORY,side,depth)
    # Step sizes of the checkerboard patterns at both axes in mm.
    step                       = [checkSize,checkSize]
    # Distance threshold in pixels to detect the interesting points wrt to the center of the image.
    target_thr                 = checkPixels
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Filter by Area.
    params.filterByArea        = True
    params.minArea             = 1
    params.maxArea             = 100
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity      = 0.1
    # Filter by Convexity
    params.filterByConvexity   = False
    # Filter by Inertia
    params.filterByInertia     = False
    # Degree of the polynomial that you want to generate.
    degree                     = [1,1]
    # Corner Harris parameters.
    # Call the calibration
    map                        = mapping.FindMap(degree,img,roi,depth,step,params,k_harris,target_thr,width,True)
    if write:
        # Save the output as image and a numpy array.
        fnmaparr = fnmapimg.replace('.png','.npy')
        np.save(fnmaparr,map)
        imsave(fnmapimg,map)
        # Prompt save files.
        print('written: %s' % fnmapimg)
        print('written: %s' % fnmaparr)
    return map

'''
import dDisplay.depth.calibrate as cal
map = cal.genAngleMap(15,'right',10.6,[20,45],width=15,roi=[321,266,841,689])
map = cal.genAngleMap(20,'right',23.,[20,10])
map = cal.genAngleMap(25,'right',23.,[15,10])
map = cal.genAngleMap(33,'right',23.,[30,40])
map = cal.genAngleMap(50,'right',23.,[30,20],roi=[285,210,723,555])
map = cal.genAngleMap(100,'right',30.18,[20,20],roi=[382,209,703,456])
map = cal.genAngleMap(500,'right',120.4,[20,60],width=5,roi=[456,324,618,434])
map = cal.genAngleMap(1000,'right',120.4,[20,60],width=5,roi=[492,285,582,329])

cal.genDisplayMaps([15,20,25,33,50,100,1000],0)

map = cal.genAngleMap(20,'left_clean',23.,[200,200],width=20,roi=[850,630,3100,2260],k_harris=[3,11,0.04])
cal.genDisplayMaps([20],0,cameraMaps=False,roi=[1400,600,2900,2100])

map = cal.genAngleMap(20,'stereoLeftClean',10.6,[10,10],width=15)
map = cal.genAngleMap(25,'stereoLeftClean',10.6,[20,20],width=15)
map = cal.genAngleMap(33,'stereoLeftClean',23.,[20,15])
map = cal.genAngleMap(25,'stereoLeft',23.,[30,60])
map = cal.genAngleMap(50,'stereoLeft',23.,[30,30],roi=[512,240,750,524])
map = cal.genAngleMap(500,'stereoLeft',120.4,[10,10],roi=[450,324,620,460])

map = cal.genAngleMap(20,'stereoLeftClean',10.6,[20,30],roi=[266,258,632,555])
map = cal.genAngleMap(25,'stereoLeftClean',10.6,[20,20],roi=[318,275,615,510])
map = cal.genAngleMap(33,'stereoLeftClean',23.,[20,15])
map = cal.genAngleMap(50,'stereoLeftClean',23.,[30,15])
map = cal.genAngleMap(100,'stereoLeftClean',30.18,[30,20])

cal.genDisplayMaps([25,50,500],1)
'''
def genDisplayMaps(depths,display,cameraMaps=True,roi=None):
    global DIRECTORY, SIDES, SIDE, WINDOWS
    DISPLAY = display
    SIDE = SIDES[DISPLAY]
    print "Working in %s/%s"%(DIRECTORY,SIDE)
    window = WINDOWS[DISPLAY]
    winSize = window["size"]
    displayMaps = []
    for depth in depths:
        directory = '%s/%s/%04d'%(DIRECTORY,SIDE,depth)
        if cameraMaps:
            captures = []
            for file in sorted(os.listdir(directory)):
                if fnmatch.fnmatch(file, 'capture*'):
                    seq = int(file.split('_')[1])
                    if not len(captures) > seq:
                        captures.append([])
                    captures[seq].append(cv2.imread('%s/%s'%(directory,file),cv2.CV_LOAD_IMAGE_GRAYSCALE))
            angleMap = np.load('%s/angles.npy'%directory)
            print "Generating camera maps for: %scm"%depth
            cameraMaps, mask = dd.evalGreyCodeCameraSpace(captures, winSize)
            np.save('%s/cameraMap.npy'%directory,cameraMaps)
            np.save('%s/mask.npy'%directory,mask)
            cameraMaps[0] = medfilt(cameraMaps[0],5)
            cameraMaps[1] = medfilt(cameraMaps[1],5)
            kernel = np.ones((5,5),np.float64)/25
            #cameraMaps[0] = cv2.filter2D(cameraMaps[0],-1,kernel)
            #cameraMaps[1] = cv2.filter2D(cameraMaps[1],-1,kernel)
            imsave('%s/cameraMap.png'%directory,np.dstack((cameraMaps[0],cameraMaps[1],np.zeros_like(cameraMaps[0]))))
            imsave('%s/mask.png'%directory,mask)
        else:
            angleMap = np.load('%s/angles.npy'%directory)
            cameraMaps = np.load('%s/cameraMap.npy'%directory)
            mask = np.load('%s/mask.npy'%directory)
            kernel = np.ones((5,5),np.float64)/25
        print "Generating display maps for: %scm"%depth
        if roi is None:
            displayMap = dd.toDisplaySpace(cameraMaps,mask,winSize,angleMap=angleMap,filter=5)
        else:
            displayMap = dd.toDisplaySpace(cameraMaps[:,roi[0]:roi[2],roi[1]:roi[3]],mask[roi[0]:roi[2],roi[1]:roi[3]],winSize,angleMap=angleMap[roi[0]:roi[2],roi[1]:roi[3]],filter=5)
        np.save('%s/displayMap.npy'%directory,displayMap)
        displayMap[0] = medfilt(displayMap[0],5)
        displayMap[1] = medfilt(displayMap[1],5)
        #displayMap[0] = medfilt(displayMap[0],5)
        #displayMap[1] = medfilt(displayMap[1],5)
        displayMap[0] = cv2.filter2D(displayMap[0],-1,kernel)
        displayMap[1] = cv2.filter2D(displayMap[1],-1,kernel)
        #displayMap = cv2.GaussianBlur(displayMap,(5,5),0)
        displayMaps.append(displayMap)
        img = np.zeros_like(displayMap)
        img[~np.isnan(displayMap)] = displayMap[~np.isnan(displayMap)] 
        imsave('%s/displayMap.png'%directory,np.dstack((img[1],img[0],np.zeros_like(img[0]))))
    print "Generating UV look up tables."
    images, angleMap = dd.generateUVluts(displayMaps)
    for idx, image in enumerate(images):
        np.save('%s/%s/%04d.npy'%(DIRECTORY,SIDE,depths[idx]),image.astype(np.float64))
        imsave('%s/%s/%04d.png'%(DIRECTORY,SIDE,depths[idx]),image)
    np.save('%s/%s/angleMap.npy'%(DIRECTORY,SIDE),angleMap.astype(np.float64))
    imsave('%s/%s/angleMap.png'%(DIRECTORY,SIDE),angleMap)

def calibratePoint(id,currentDepth,requestedDepth,stillRunning,sharedImage):
    ''' NOTE: curent method should only work for the point straight down the axis - otherwise we need to map angles'''
    print 'Entering calibratePoint'
    global DEPTH, DISPLAY, SIDE, DIRECTORY, IMAGE, POINT
    requestedDepth[DISPLAY].value = DEPTH
    WINDOWS = initDisplay()
    winSize = glfw.get_window_size(WINDOWS[1])
    img = np.zeros(winSize,dtype=np.uint8)
    # Create a marker at POINT
    img[POINT[0]-5:POINT[0]+5,POINT[1]-5:POINT[1]+5] = 255
    img = np.fliplr(img)
    imgTex = expgui.createTexture(img)
    # load lut
    tables, depths = explogic.ExperimentScreen.loadLuts('%s/%s'%(DIRECTORY,SIDE))
    lutImg = tables[depths.index(DEPTH)]
    lutTex = expgui.createTexture(lutImg)
    glfw.make_context_current(WINDOWS[1])
    frameTex, frameBuf = expgui.createWarp(tables[0].shape[1],tables[0].shape[0])
    shader = expgui.lutShader()
    # get camera image
    image = toNumpyImage(sharedImage)
    # setup blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 50
    params.maxThreshold = 150
    params.minArea=30
    params.maxArea=100
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector(params) 
    count = 0
    while not glfw.window_should_close(WINDOWS[0]) and not glfw.window_should_close(WINDOWS[1]):
        #get the right luts
        glfw.make_context_current(WINDOWS[1])
        expgui.drawWarp(frameTex, frameBuf, lutTex, imgTex, shader)
        #expgui.drawQuad(imgTex)
        glfw.swap_buffers(WINDOWS[1])
        glfw.poll_events()
        count += 1
        if count % 30:
            with sharedImage.get_lock():
                frame = np.copy(image)
                print frame.shape
                print frame.dtype
                frame2 = 1.-frame
                imsave('blobShared.png',frame2)
                ret,thresh = cv2.threshold(frame2,params.maxThreshold,255,cv2.THRESH_TRUNC)
            #blob detect
            culled = []  
            coord = []
            keypoints = detector.detect(thresh)
            if len(keypoints):
                for key in keypoints:
                    if bounds[0] < key.pt[0] < bounds[1] and bounds[2] < key.pt[1] < bounds[3]:
                        coord.append(key.pt)
                        culled.append(key)
                #frame = cv2.drawKeypoints(frame, culled, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            print coord
            #
    stillRunning.value = False
    glfw.terminate()

def streamCamera(stillRunning,sharedImage,currentDepth):
    global DISPLAY
    print 'Entering streamCamera'
    cam = initCamera()
    alpha = 0.
    if not cam.open():
        return None
    image = toNumpyImage(sharedImage)
    print 'Camera entering while loop'
    while(True):
        ret, frame = cam.read()     # Capture the frame
        size = frame.shape
        newMtx,roi = cv2.getOptimalNewCameraMatrix(cam.matrix,cam.distortion,(size[1],size[0]),alpha)
        map1, map2 = cv2.initUndistortRectifyMap(cam.matrix,cam.distortion,np.eye(3),newMtx,(size[1],size[0]),cv2.CV_16SC2)
        frame = cv2.remap(frame,map1,map2,cv2.INTER_LINEAR)
        display = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        depth = currentDepth[0].value
        diopter = 100./max(depth,.1)
        height = int((.04727+diopter/7.333)*-display.shape[0])+display.shape[0]
        point = (5,height)
        height = int((.03+depth/1100.)*display.shape[0])
        point2 = (100,height)
        ImageTools.drawStr(display,point2,'-%.4gcm'%depth,scale=1.7,thick=2,color=(0,0,255),backCol=(255,255,255))
        ImageTools.drawStr(display,point,'%1.2fd-'%diopter,scale=1.7,thick=2,color=(0,255,0),backCol=(255,255,255))
        cv2.imshow('frame',display)   # Display the frame
        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:                # escape
            break
        elif ch == 32:              # space bar
            imsave('blobOrig.png',frame)
        with sharedImage.get_lock():
            image[:,:] = frame[:,:]
    cam.release()                   # release the capture
    cv2.destroyAllWindows()

def verifyCalibration(id,currentDepth,requestedDepth,stillRunning,*args):
    print 'Entering verifyCalibration'
    global DEPTH, DISPLAY, SIDE, DIRECTORY, IMAGE, GLWARP
    WINDOWS = initDisplay()
    speed = 20 #smaller is faster
    img = cv2.imread(IMAGE,-1)
    winSize = glfw.get_window_size(WINDOWS[1])
    img = img[0:winSize[1],0:winSize[0]]
    
    #pts1 = np.float32([[612,410],[792,366],[621,589],[808,543]])
    #pts2 = np.float32([[612,410],[792,410],[612,589],[792,589]])
    #M = cv2.getPerspectiveTransform(pts1,pts2)
    #img = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
    
    imgTex = expgui.createTexture(img)
    tables, depths = explogic.ExperimentScreen.loadLuts('%s/%s'%(DIRECTORY,SIDE))
    lutMap = interp1d(depths,range(len(tables)))
    # double up on the ends so we can see it
    depths.insert(0,depths[0])
    depths.append(depths[-1])
    depthMap = interp1d(np.linspace(0.,1.,len(depths)),depths)
    glfw.make_context_current(WINDOWS[1])
    frameTex, frameBuf = expgui.createWarp(tables[0].shape[1],tables[0].shape[0])
    shader = expgui.lutMixShader()
    lutTex = []
    for table in tables:
        #table = np.dstack((table[:,:,0],table[:,:,2],table[:,:,1]))
        #table = np.flipud(table)
        #table = np.fliplr(table)
        lutTex.append(expgui.createTexture(table))
    print 'Calibration entering while loop'
    while not glfw.window_should_close(WINDOWS[0]) and not glfw.window_should_close(WINDOWS[1]):
        #set depth
        #triangle = lambda x: abs(abs(((x+1)%4)-2)-1)
        #depth = depthMap(triangle(time.time()/speed))
        depth = DEPTH
        requestedDepth[DISPLAY].value = depth
        #get the right luts
        glfw.make_context_current(WINDOWS[1])
        low, high, factor = explogic.ExperimentScreen.chooseLut(lutMap,depth)
        if GLWARP:
            expgui.drawMixWarp(frameTex, frameBuf, lutTex[low], lutTex[high], factor, imgTex, shader)
        else:
            maps = cv2.addWeighted(tables[low],factor,tables[high],1-factor,0)
            result = cv2.remap(img,maps[:,:,0].astype(np.float32)*maps.shape[0],(1-maps[:,:,1].astype(np.float32))*maps.shape[1],cv2.INTER_LINEAR)
            expgui.updateTexture(imgTex, result)
            expgui.drawQuad(imgTex)
        glfw.swap_buffers(WINDOWS[1])
        glfw.poll_events()
    stillRunning.value = False
    glfw.terminate()

def verifyDepth(id,currentDepth,requestedDepth,stillRunning):
    global DEPTH, DISPLAY, SIDE, DIRECTORY, WINDOWS
    pass

def toNumpyImage(mp_arr):
    global RESOLUTION
    array = np.frombuffer(mp_arr.get_obj())
    array.shape = RESOLUTION
    return array

def main(function=captureWarp,display=None,camera=None,depth=None,directory=None,image=None,resolution=None,point=None):
    global DISPLAY,SIDES,SIDE,CAMERA,DEPTH,DIRECTORY,IMAGE,RESOLUTION,POINT
    DISPLAY = display if display is not None else DISPLAY
    SIDE = SIDES[DISPLAY]
    CAMERA = camera if camera is not None else CAMERA
    DEPTH = depth if depth is not None else DEPTH
    DIRECTORY = directory if directory is not None else DIRECTORY
    IMAGE = image if image is not None else IMAGE
    RESOLUTION = resolution if resolution is not None else RESOLUTION
    POINT = point if point is not None else POINT

    currentDepth = []
    requestedDepth = []
    for i in range(expglobals.NUM_DEPTH_DISPLAYS):
        currentDepth.append(Value('d', 0.))
        requestedDepth.append(Value('d',0.))
    stillRunning = Value('b', True)
    sharedImage = Array(ctypes.c_double, RESOLUTION[0]*RESOLUTION[1])

    processes = []
    processes.append(Process(target=function, args=(DISPLAY,currentDepth,requestedDepth,stillRunning,sharedImage)))
    #if function != captureWarp:
    #    processes.append(Process(target=streamCamera, args=(stillRunning,sharedImage,currentDepth)))
    processes.append(Process(target=runDisplay, args=(sys.argv,DISPLAY,currentDepth,requestedDepth,stillRunning)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()

def interactiveMain():
    windows = initDisplay()
    # capture the images by hand
    captureWarpInteractive(windows)
    # transfer the images to some directory
    wait(windows[1])
    
    ui.terminate

if __name__ == "__main__":
    #main(calibratePoint)
    #main(verifyCalibration)
    #main(captureWarp)
    interactiveMain()
'''
captures = []
for i in range(3):
    seq = []
    for j in range(9):
        seq.append(cv2.imread('../AccomodationExperiment/data/calibration/capture_%02d_%02d.png'%(i,j),-1))
    captures.append(seq)



import ctypes, cv2
from multiprocessing import Array
import numpy as np
a = Array(ctypes.c_double, 786432)


image = tonumpyarray(a)
image.shape = (768,1024)

cv2.imshow('temp',image)
cv2.waitKey()

'''