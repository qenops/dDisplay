
import sys
sys.path.append(r'../python')
import numpy as np
import serial
from scipy.interpolate import interp1d, interp2d
import time, random
#import dDisplay.blobDetectorParameters as blob

class VarifocalDisplay(object):
    ''' An object to control a varifocal display '''
    def __init__(self,port='/dev/ttyACM0',baud=115200,dist=None,signals=None,**kwargs):
        try:
            print('trying port %s and baud %s'%(port, baud))
            self.arduino = serial.Serial(port, baud)
            print('Connected port %s'%port)
        except:
            print('FATAL ERROR: Cannot connect to Adruino => will simulate.')
            self.arduino = None
        self.currentDepths = [0,0]
        self.goalDepth = 0
        self.lastSignals = [0,0]
        self.signs = (-1,1)
        dist = (-1, 0, 50,250,1000,1500) if dist is None else dist
        signals = ((0,0,2400,2600,2700,2800),(0,0,2400,2600,2700,2800)) if signals is None else signals
        self.signalMaps = (interp1d(dist,signals[0]),interp1d(dist,signals[1]))
        self.depthMaps = (interp1d(signals[0],dist),interp1d(signals[1],dist))
    def getFocus(self):
        if self.arduino:
            bytesToRead = self.arduino.inWaiting()
            while (bytesToRead > 15):
                self.arduino.read(bytesToRead)           # Throw away everything in buffer - is this good? will we miss a signal?
                bytesToRead = self.arduino.inWaiting()
            data = [int(self.arduino.readline())]
            data.append(int(self.arduino.readline()))
            for i in data:
                if i < -9000: # left eye op code
                    pass
                elif i > 9000: # right eye op code
                    pass
                elif i < 0: # left eye depth signal
                    self.currentDepths[0] = self.depthMaps[0](i*-1)
                elif i > 0: # right eye depth signal
                    self.currentDepths[1] = self.depthMaps[1](i)
        return self.currentDepths
    # @param focus - focus depth in cm.
    # @param speed - speed of refocus in diopters/second. [speed <= 0] => maximum speed.
    def setFocus(self,focus):
        print("Setting display depth to %s"%focus)
        self.goalDepth = focus
        self.lastSignals = [self.signalMaps[i](focus) * self.signs[i] for i in range(2)]
        for signal in self.lastSignals:
            self.sendSignal(signal)
    def sendSignal(self, signal):
        self.lastSignal = signal
        if self.arduino:
            self.arduino.write(('%s\n'%self.lastSignal).encode('ascii'))
        #print "Sending %s"%self.lastSignal
    def close(self):
        self.sendSignal(0)
        if self.arduino:
            self.arduino.close()
    def measureLatency(self,loops):
        func = [self.increaseDepth,self.decreaseDepth]
        frameCount = 0
        frameTime = 0
        data = [[],[],[]]
        for i in range(loops*2):
            self.readLED()
            data[2].append((self.coord,time.time()))
            func[i%2](100)
            aboveNoise = False
            startTime = time.time()
            for j in range(30):
                prev = self.coord
                self.readLED()
                data[2].append((self.coord,time.time()))
                if np.any(abs(self.coord - prev) > self.ledNoise):
                    aboveNoise = True
                if aboveNoise and np.all(abs(self.coord - prev) < self.ledNoise):
                    endTime = time.time()
                    break
            data[i%2].append(endTime-startTime)
            '''
            startTime = time.time()
            for j in range(6):
                self.readLED()
                data[2].append((self.coord,time.time()))
                endTime = time.time()
                frameTime += endTime-startTime
                frameCount += 1
                startTime = endTime
        data.append(frameTime/frameCount)
        print 'Average frame time = %s'%data[-1]
        '''
        return data


def trackLED(ledCam,params=None,minArea=16,maxArea=75,minThreshold=0,maxThreshold=150,bounds=(0,9000,0,9000),thresholdStep=7,minDistance=10):
    # get an image
    ledCam.open()
    ret, frame = ledCam.read()
    # convert to greyscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # inverse the image
    frame2 = np.invert(frame)
    ret,thresh = cv2.threshold(frame2,maxThreshold,255,cv2.THRESH_TRUNC)
    # blob the image
    if params is None:
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = minThreshold
        params.maxThreshold = maxThreshold
        params.thresholdStep = thresholdStep
        params.minDistBetweenBlobs = minDistance
        params.minArea=minArea
        params.maxArea=maxArea
        params.filterByConvexity = False
        params.minConvexity = .93
        params.maxConvexity = 1.
        params.filterByInertia = False
        params.minInertiaRatio = .7
        params.maxInertiaRatio = 1.
    detector = cv2.SimpleBlobDetector(params) 
    # Detect blobs.
    keypoints = detector.detect(thresh)
    culled = []  
    coord = []
    if len(keypoints):
        for key in keypoints:
            if bounds[0] < key.pt[0] < bounds[1] and bounds[2] < key.pt[1] < bounds[3]:
                coord.append(key.pt)
                culled.append(key)
        frame = cv2.drawKeypoints(frame, culled, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        coord.append((0.,0.))
    return coord, frame

def arduino():
    # use pyserial to control the pressure to maintain blob
    arduino = serial.Serial('COM3', 9600)
    #arduino.write('0') # turn vacuum off
    #arduino.write('33') # current far plane
    #arduino.write('28') # current near plane
    arduino.close()

def tuneLED(ledCam=None,bounds=(0,9000,0,9000)):
    ledCam = dc.Camera(0) if ledCam is None else ledCam
    blob.setup()
    #frames = []
    prev = (0,0)
    while(True):
        params = blob.update()
        coord, frame = trackLED(ledCam,params=params,bounds=bounds)
        cv2.imshow('tracking',frame)
        #print coord
        ch = cv2.waitKey(100)
        if ch & 0xFF == 27:         # escape
            break
    dc.cv2CloseWindow('tracking')
    print("minArea=%s,maxArea=%s,minThreshold=%s,maxThreshold=%s,bounds=%s"%(minArea,maxArea,minThreshold,maxThreshold,bounds))
    return minArea,maxArea,minThreshold,maxThreshold

def stressTestMembrane(camera, port, frequency=600):
    dir = 'C:/workspace/membraneStress/'
    pZero = '0'
    pOne = '90'
    pTwo = '400'
    pThree = '1000'
    cOne = 0
    cTwo = 0
    cThree = 0

    display = DepthDisplay(port,led=False)
    display.sendSignal(pZero)
    ptg = dc.Camera(0,1)
    time.clock()
    currentSignal = pThree

    while True:
        if time.clock() % frequency < 4:
            time.sleep(5)
            ret, frame = camera.read()
            cv2.imwrite('%s%s.png'%(dir,time.ctime().replace(':','-')),frame)
        if currentSignal == pOne and time.clock() > 3600:
            currentSignal = pTwo
        if currentSignal == pTwo and time.clock() > 14400:
            currentSignal = pThree
        cThree += 1
        print(cThree)
        display.sendSignal(currentSignal)
        time.sleep(2)
        display.sendSignal(pZero)
        time.sleep(2)

class VarifocalStereoDisplay(object):
    ''' If using two separate arduinos to control the display '''
    def __init__(self,right={},left={},**kwargs):
        self.currentDepth = 0
        self.goalDepth = 0
        self.lastSignal = 0
        port = '/dev/ttyACM0' if 'port' not in right else right['port']
        baud = 9600 if 'baud' not in right else right['baud']
        try:
            self.arduino = serial.Serial(port, baud)
            #print('Connected port %s'%port)
        except:
            print('FATAL ERROR: Cannot connect to Adruino => will simulate.')
            self.arduino = None
        for eye in [right,left]:
            eye['port'] = port
            eye['baud'] = baud
        self.right = VarifocalDisplay(**right)
        #self.left = VarifocalDisplay(**left)
    def sendSignal(self, signal):
        print('Sending signal %s'%signal)
        self.lastSignal = signal
        self.right.sendSignal(signal)
        if self.arduino:
            self.arduino.write(('%s\n'%self.right.lastSignal).encode('ascii'))
        #self.left.sendSignal(signal)
        #if self.arduino:
        #    self.arduino.write(('%s\n'%-self.left.lastSignal).encode('ascii'))
    def close(self):
        self.sendSignal(0)
        self.sendSignal(0)
        self.sendSignal(0)
        #self.right.close()
        #self.left.close()
        if self.arduino:
            self.arduino.close()
    
    


'''
import sys
sys.path.append(r'../python')

#RIGHT EYE
from dDisplay import varifocal as vf
led = True
camera = 0
points = ((  0, 332.8, 229.4),
          ( 20, 314.2, 301.8),
          (100, 306.0, 323.8),
          (900, 290.1, 362.1),
          
         )
#dist = [p[0] for p in points]
#ledX = [p[1] for p in points]
#ledY = [p[2] for p in points]
dist = (    0,   25,   33,   50,   66,  100,  150,  200,)# 500, 700)
#dist = None
signal=None
ledX = (313.9,298.1,297.4,296.3,294.5,293.9,292.7,292.5,)# 292.7,292.7)
ledY = (227.9,279.7,281.2,283.2,286.9,287.9,290.0,290.5,)# 290.0,290.0)
#ledX = None
#ledY = None
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

display.sendSignal('400')
display.increaseDepth(.02,True)
display.decreaseDepth(.02,True)

i = 0
while i < 20:
    currentDepth = display.updateState(50)
    time.sleep(1)
    i += 1
'''
