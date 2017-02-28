import sys, cv2, time, numpy as np
sys.path.append(r'../python')
from multiprocessing import Process, Value
import depthDisplay as dd
import sys

def runCamera(display,currentDepth,signalTime,stillRunning):
    previousDepth = 0
    collecting = 0
    data = []
    count = 0
    while stillRunning.value:
        #print "Server is alive";
        #currentDepth[id].value = display.updateState(requestedDepth[id].value)
        if display.readLED():
            previousDepth = currentDepth.value
            currentDepth.value = display.currentDepth
        '''
        if collecting == 0 and currentDepth.value > previousDepth + previousDepth/13.:                
            collecting = 10
            data = []
            data.append(previousDepth)
            now = time.time()
            latency = now - signalTime.value
            print latency
        if collecting > 0:
            collecting -= 1
            data.append(currentDepth.value)
            if collecting == 0:
                e = np.argmax(data)
                #print data
                end = data[e]
                delta = end - data[0]
                print data[0], delta
                #display.ledCam.close()
        #print data
        #print currentDepth.value, previousDepth
        count += 1
        '''
    print "Done camera"

def update(dummy=None):
    pass

def runControls(display,currentDepth,signalTime,stillRunning):
    cv2.namedWindow('Depth Display Controls')
    #cv2.createTrackbar('Miliseconds', 'Depth Display Controls', 20, 100, update)
    #cv2.createTrackbar('Multiplier', 'Depth Display Controls', 1, 10, update)
    #cv2.createTrackbar('Dummy', 'Depth Display Controls', 1, 1, update)
    cv2.createTrackbar('Goal', 'Depth Display Controls', 25, 1000, update)
    cv2.createTrackbar('Increase A', 'Depth Display Controls', 736, 1000, update)
    cv2.createTrackbar('Increase A Exponent (Neg)', 'Depth Display Controls', 7, 10, update)
    cv2.createTrackbar('Increase B', 'Depth Display Controls', 0, 1000, update)
    cv2.createTrackbar('Increase C', 'Depth Display Controls', 0, 1000, update)
    cv2.createTrackbar('Decrease D', 'Depth Display Controls', 0, 1000, update)
    cv2.createTrackbar('Decrease A', 'Depth Display Controls', 736, 1000, update)
    cv2.createTrackbar('Decrease A Exponent (Neg)', 'Depth Display Controls', 7, 10, update)
    cv2.createTrackbar('Decrease B', 'Depth Display Controls', 0, 1000, update)
    cv2.createTrackbar('Decrease C', 'Depth Display Controls', 0, 1000, update)
    cv2.createTrackbar('Decrease D', 'Depth Display Controls', 0, 1000, update)
    while(True):
        goal = cv2.getTrackbarPos('Goal', 'Depth Display Controls')
        epsilon = goal/13.
        depth = currentDepth.value
        if depth < goal - epsilon:
            #display.increaseDepth(goal-depth)
            a = cv2.getTrackbarPos('Increase A', 'Depth Display Controls')
            a = a * 10**-cv2.getTrackbarPos('Increase A Exponent (Neg)', 'Depth Display Controls')
            b = cv2.getTrackbarPos('Increase B', 'Depth Display Controls')
            c = cv2.getTrackbarPos('Increase C', 'Depth Display Controls')
            d = cv2.getTrackbarPos('Increase D', 'Depth Display Controls')/1000.
            seconds = a*(b+goal-depth)/max(c+depth,1.) + d
            seconds = max(seconds, 0.)
            #print seconds
            display.sendSignal(display.openSignal)
            time.sleep(seconds)
            display.sendSignal(display.closeSignal)
        elif depth > goal + epsilon:
            #display.decreaseDepth(depth-goal)
            a = cv2.getTrackbarPos('Decrease A', 'Depth Display Controls')
            a = a * 10**-cv2.getTrackbarPos('Decrease A Exponent (Neg)', 'Depth Display Controls')
            b = cv2.getTrackbarPos('Decrease B', 'Depth Display Controls')
            c = cv2.getTrackbarPos('Decrease C', 'Depth Display Controls')
            d = cv2.getTrackbarPos('Decrease D', 'Depth Display Controls')/1000.
            seconds = a*(b+depth-goal)/max(c+depth,1.) + d
            seconds = max(seconds, 0.)
            display.sendSignal(display.ventSignal)
            time.sleep(seconds)
            display.sendSignal(display.ventCloseSignal)
        print depth
        ch = cv2.waitKey(1000)
        if ch & 0xFF == 27:         # escape
            break
        '''
        elif ch == 1113938:         # Upkey
            seconds = cv2.getTrackbarPos('Miliseconds', 'Depth Display Controls')/1000.
            seconds = seconds * cv2.getTrackbarPos('Multiplier', 'Depth Display Controls')
            signalTime.value = time.time()
            display.increaseDepth(seconds,True)
        elif ch == 1113940:         # DownKey
            seconds = cv2.getTrackbarPos('Miliseconds', 'Depth Display Controls')/1000.
            seconds = seconds * cv2.getTrackbarPos('Multiplier', 'Depth Display Controls')
            signalTime.value = time.time()
            display.decreaseDepth(seconds,True)
        '''
    stillRunning.value= False
    a = cv2.getTrackbarPos('Increase A', 'Depth Display Controls')
    a = a * 10**-cv2.getTrackbarPos('Increase A Exponent (Neg)', 'Depth Display Controls')
    b = cv2.getTrackbarPos('Increase B', 'Depth Display Controls')
    c = cv2.getTrackbarPos('Increase C', 'Depth Display Controls')
    d = cv2.getTrackbarPos('Increase D', 'Depth Display Controls')/1000.
    print "Increase: s = %s * (%s+delta)/(%s+depth) + %s"%(a,b,c,d)
    a = cv2.getTrackbarPos('Decrease A', 'Depth Display Controls')
    a = a * 10**-cv2.getTrackbarPos('Decrease A Exponent (Neg)', 'Depth Display Controls')
    b = cv2.getTrackbarPos('Decrease B', 'Depth Display Controls')
    c = cv2.getTrackbarPos('Decrease C', 'Depth Display Controls')
    d = cv2.getTrackbarPos('Decrease D', 'Depth Display Controls')/1000.
    print "Decrease: s = %s * (%s+delta)/(%s+depth) + %s"%(a,b,c,d)

if __name__ == "__main__":
    currentDepth = Value('d', 0.)
    signalTime = Value('d',0.)
    stillRunning = Value('b', True)
    print "Starting display"
    if sys.argv[1] == 0:
        camera = 0
        dist = ( -1,  0, 25, 35, 50, 100, 250,1000,1500)
        ledX = (  0,223,205,190,170, 150, 140, 132, 100)
        ledY = (  0,349,310,280,234, 193, 170, 150, 100)
        bounds = (0,1000,0,1000)
        minArea = 50
        maxArea = 390
        minThreshold=24
        maxThreshold=55
        thresholdStep=10
        minDistance = 74
        port = '/dev/ttyACM0'
    else:
        camera = 3
        dist = ( -1,  0, 25, 35, 50,100,250,1000,1500)
        ledX = (  0,185,173,158,134,120,102,86.6, 70)
        ledY = (  0, 87,152,198,286,335,396, 449, 500)
        bounds = (70,200,80,1000)
        minArea = 50
        maxArea = 200
        minThreshold=60
        maxThreshold=200
        thresholdStep=4
        minDistance = 55
        port = '/dev/ttyACM1'
    display = dd.DepthDisplay(port=port,camera=camera,dist=dist,ledX=ledX,ledY=ledY,bounds=bounds,minArea=minArea,maxArea=maxArea,minThreshold=minThreshold,maxThreshold=maxThreshold, thresholdStep = thresholdStep)
    processes = []
    processes.append(Process(target=runCamera, args=(display,currentDepth,signalTime,stillRunning,)))
    processes.append(Process(target=runControls, args=(display,currentDepth,signalTime,stillRunning,)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    display.close()
    print "Closing display"
