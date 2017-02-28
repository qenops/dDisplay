#!/usr/bin/python
''' Demo for depth display - for Siggraph E-Tech submission

David Dunn
Feb 2017 - created

www.qenops.com

'''
__author__ = ('David Dunn')
__version__ = '1.0'

import dDisplay as dd
import dDisplay.varifocal as vf
import dGraph as dg
import dGraph.test.test2 as test
import dGraph.ui as ui
import multiprocessing as mp
import numpy as np
import time, math

TIMING = False
#TIMING = True
#SIDE = 'Right'
SIDE = 'Left'
WINDOWS = [
    {
    #"name": 'HMD Left',
    "name": 'HMD Right',
    #"location": (0, 0),
    #"location": (3266, 1936), # px coordinates of the startup screen for window location
    "location": (2640, 1936), # px coordinates of the startup screen for window location
    "size": (830, 800), # px size of the startup screen for centering
    "center": (290,216), # center of the display
    "refresh_rate": 60, # refreshrate of the display for precise time measuring
    "px_size_mm": 0.09766, # px size of the display in mm
    "distance_cm": 20, # distance from the viewer in cm,
    #"is_hmd": False,
    #"warp_path": 'data/calibration/newRight/',
    },
]
if SIDE == 'Left':
    WINDOWS[0]['name'] = 'HMD Left'
    WINDOWS[0]['location'] = (3266, 1936)
START = 0.
DEPTHS = [20,50,700]

def setupDisplay():
    led = False
    camera = 0
    points = ((  0, 332.8, 229.4),
            ( 20, 314.2, 301.8),
            (100, 306.0, 323.8),
            (900, 290.1, 362.1),
            
            )
    #dist = [p[0] for p in points]
    #ledX = [p[1] for p in points]
    #ledY = [p[2] for p in points]
    #dist = (    0,   25,   33,   50,   66,  100,  150,  200,)# 500, 700)
    dist = None
    signal=None
    #ledX = (313.9,298.1,297.4,296.3,294.5,293.9,292.7,292.5,)# 292.7,292.7)
    #ledY = (227.9,279.7,281.2,283.2,286.9,287.9,290.0,290.5,)# 290.0,290.0)
    ledX = None
    ledY = None
    bounds = (0,1000,0,1000)
    minArea = 50
    maxArea = 390
    minThreshold=24
    maxThreshold=200
    thresholdStep=10
    minDistance = 10
    #minRepeatablilty=2
    #circularity=True
    #minCircularity=80
    port = '/dev/ttyACM0'
    display = vf.VarifocalDisplay(port=port,led=led,camera=camera,dist=dist,signal=signal,ledX=ledX,ledY=ledY,bounds=bounds,minArea=minArea,maxArea=maxArea,minThreshold=minThreshold,maxThreshold=maxThreshold, thresholdStep=thresholdStep)
    return display

def runDisplay(display, requestedDepth, stillRunning):
    currentDepth = 20
    signal = np.array([[   0, 18,19.5],
                       [-300,  0,10.5],
                       [-300,-11,   0]])
    while stillRunning.value:
        if requestedDepth.value != currentDepth:
            period = signal[DEPTHS.index(currentDepth),DEPTHS.index(requestedDepth.value)]
            currentDepth = requestedDepth.value
            display.sendSignal(period)
    display.close()

def animate(renderStack):
    now = time.time()-START
    y = math.sin(now*math.pi)
    x = math.cos(now*math.pi*2)/4
    #offset = np.array((x,y,0))*4
    #rotate = np.array((5.,-15.,0.)) + offset
    #renderStack.objects['teapot'].setRotate(*rotate) 
    renderStack.objects['teapot'].rotate += np.array((x,y,0.)) 

def addInput(requestedDepth,renderStack):
    ui.add_key_callback(switchDepths, ui.KEY_1, value=20, requestedDepth=requestedDepth,renderStack=renderStack)
    ui.add_key_callback(switchDepths, ui.KEY_2, value=50, requestedDepth=requestedDepth,renderStack=renderStack)
    ui.add_key_callback(switchDepths, ui.KEY_3, value=700, requestedDepth=requestedDepth,renderStack=renderStack)

def switchDepths(window,value,requestedDepth,renderStack):
    print "Requesting Depth: %s"%value
    requestedDepth.value = value
    signal = np.array([[-.02,.02  , -2.],
                       [-.05,.1052, -5.],
                       [-.205,.53   ,-20.],
                       #[-.70,1.945,-70.],
                      ])
    '''signal = np.array([[-.02,.02  , -2.],
                       [-.02,.0295, -2.],
                       [-.02,.031 , -2.],
                      ])'''
    scaler = np.array([[.035,.035,.035],
                       [.029,.0255,.029],
                       [.0278,.0238,.0278],
                      ])
    trans = signal[DEPTHS.index(value)]
    scale = scaler[DEPTHS.index(value)]
    renderStack.objects['teapot'].setTranslate(*trans)
    #renderStack.objects['teapot'].setScale(*scale)
    
def setup():
    ui.init()
    windows = []
    idx = 0
    winData = WINDOWS[idx]
    renderStack = ui.RenderStack()
    renderStack.display = dd.Display(resolution=winData['size'])
    share = None if idx == 0 else windows[0]
    window = renderStack.addWindow(ui.open_window(winData['name'], winData['location'][0], winData['location'][1], renderStack.display.width, renderStack.display.height, share=share))
    if not window:
        ui.terminate()
        exit(1)
    ui.make_context_current(window)
    dg.initGL()
    windows.append(window)
    ui.add_key_callback(ui.close_window, ui.KEY_ESCAPE)
    scene = test.loadScene(renderStack)
    renderStack.graphicsCardInit()
    return renderStack, scene, windows

def runLoop(renderStack, windows, requestedDepth, stillRunning):
    # Print message to console, and kick off the loop to get it rolling.
    print("Hit ESC key to quit.")
    index = 0
    while not ui.window_should_close(windows[0]):
        animate(renderStack)
        ui.make_context_current(windows[0])
        test.drawScene(renderStack)
        ui.swap_buffers(windows[0])
        if TIMING:
            now = time.time()
            if (now % 5) < .01:
                index = int(((now % 15)/5))%len(DEPTHS)
                switchDepths(windows[0],DEPTHS[index],requestedDepth,renderStack)
        ui.poll_events()
        #ui.wait_events()
    stillRunning.value=False
    ui.terminate()
    exit(0)

def runDemo():
    global START
    now = time.time()
    while now % 4 > .0001:
        now = time.time()
    START = time.time()
    renderStack, scene, windows = setup()
    if SIDE == 'Left':
        display = setupDisplay()
    requestedDepth = mp.Value('d',20)
    addInput(requestedDepth,renderStack)
    stillRunning = mp.Value('b', True)

    processes = []
    if SIDE == 'Left':
        processes.append(mp.Process(target=runDisplay, args=(display, requestedDepth,stillRunning)))
    for p in processes:
        p.start()
    runLoop(renderStack,windows,requestedDepth, stillRunning)
    stillRunning.value=False
    for p in processes:
        p.join()

if __name__ == '__main__':
    runDemo()
