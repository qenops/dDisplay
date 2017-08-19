
import cv2, sys
sys.path.append(r'../python')
import numpy as np
import serial
import dCamera as dc
from scipy.interpolate import interp1d, interp2d
import time, random
import dDisplay.blobDetectorParameters as blob

class VarifocalDisplay(object):
    ''' An object to control a varifocal display '''
    def __init__(self,port='/dev/ttyACM0',baud=9600,led=True,camera=0,dist=None,signal=None,ledX=None,ledY=None,bounds=(0,1000,0,1000),minArea=15,maxArea=150,minThreshold=0,maxThreshold=150,thresholdStep=7,minDistance=10,**kwargs):
        self.currentDepth = 0
        self.goalDepth = 0
        self.lastSignal = 0
        try:
            print('trying port %s and baud %s'%(port, baud))
            self.arduino = serial.Serial(port, baud)
            print('Connected port %s'%port)
        except:
            print('FATAL ERROR: Cannot connect to Adruino => will simulate.')
            self.arduino = None
        self.epsilon = 15
        dist = (-1, 0, 50,250,1000,1500) if dist is None else dist
        self.solenoids = True if signal is None else False
        if signal is not None:
            #signal = (0,0,195,230, 250, 300) if signal is None else signal
            self.signalMap = interp1d(dist,signal)
        self.led = led
        if self.led:
            self.ledCam = dc.Camera(camera,0)
            self.ledCam.open()
            self.cameraID = camera
            ledX = (450,382,300,281,226,200) if ledX is None else ledX
            ledY = (100,134,223,247,333,400) if ledY is None else ledY
            self.ledMap = interp2d(ledX, ledY, dist)
            self.ledNoise = np.array((.8,1.)) # Highest measured: [1.974945068359375, 2.407806396484375]
        self.bounds = bounds
        self.minArea = minArea
        self.maxArea = maxArea
        self.minThreshold = minThreshold
        self.maxThreshold = maxThreshold
        self.thresholdStep = thresholdStep
        self.minDistance = minDistance
        self.coord = np.array((0.,0.))
    def getFocus(self):
        self.readLED()
        return self.currentDepth
    def readLED(self):
        coord, frame = trackLED(self.ledCam,minArea=self.minArea,maxArea=self.maxArea,bounds=self.bounds,minThreshold=self.minThreshold,maxThreshold=self.maxThreshold,thresholdStep=self.thresholdStep,minDistance=self.minDistance)
        if coord == []:
            return False
        self.coord = np.array(coord[0])
        self.currentDepth = self.ledMap(*coord[0])[0]
        #print "Current Depth: %s"%self.currentDepth, coord 
        return True
    # @param focus - focus depth in cm.
    # @param speed - speed of refocus in diopters/second. [speed <= 0] => maximum speed.
    def setFocus(self,focus):
        print("Setting display depth to %s"%focus)
        self.goalDepth = focus 
    def updateState(self,focus=None):
        '''read current state and send signal to control the display for the desired depth '''
        if not(focus is None or focus == self.goalDepth):
            self.setFocus(focus)
        if self.solenoids:
            self.readLED()
            self.epsilon = self.goalDepth * .1
            #if self.epsilon < 15:
            #    self.epislon = 15
            if self.currentDepth < (self.goalDepth - self.epsilon):
                print('+%s'%(self.goalDepth-self.currentDepth))
                self.increaseDepth(self.goalDepth-self.currentDepth)
            elif self.currentDepth > (self.goalDepth + self.epsilon):
                print('-%s'%(self.currentDepth-self.goalDepth))
                self.decreaseDepth(self.currentDepth-self.goalDepth)
            #else:
            #    self.arduino.write('%s/n'%self.closeSignal)
        else:
            newSignal = self.signalMap(self.goalDepth)
            if self.lastSignal != newSignal:
                self.sendSignal(newSignal)
                self.lastSignal = newSignal
            if self.led:
                self.readLED()
            else:
                self.currentDepth = self.goalDepth
            time.sleep(.3)
        return self.currentDepth
    def increaseDepth(self, dist, period=False):
        if period:
            mseconds = dist
        else:
            mseconds = 0.07368421*dist + 9.8
        mseconds = max(mseconds, 0.)
        self.sendSignal(mseconds)
    def decreaseDepth(self, dist, period=False):
        if period:
            mseconds = dist
        else:
            mseconds = 0.7368421*dist + 9.8
        seconds = max(mseconds, 0.)
        self.sendSignal(-mseconds)
    def sendSignal(self, signal):
        self.lastSignal = signal
        if self.arduino:
            self.arduino.write(('%s\n'%self.lastSignal).encode('ascii'))
        #print "Sending %s"%self.lastSignal
    def close(self):
        self.decreaseDepth(1000)
        if self.arduino:
            self.arduino.close()
        if self.led:
            self.ledCam.close()
    def tuneLED(self):
        blob.setup(minArea=self.minArea, maxArea=self.maxArea,minThreshold=self.minThreshold,maxThreshold=self.maxThreshold)
        capture = False
        #frames = []
        noise = [0.,0.]
        prev = [0.,0.]
        count = 0
        while(True):
            params = blob.update()
            coord, frame = trackLED(self.ledCam,params=params,bounds=self.bounds)
            coord = [(0.,0.)] if coord == [] else coord
            cv2.imshow('tracking',frame)
            if capture:
                video.write(frame)
            self.currentDepth = self.ledMap(*coord[0])[0]
            #print self.currentDepth, coord[0]
            count += 1
            if count == 30:
                prev = coord[0]
            elif count > 30:
                noise[0] = max(noise[0], abs(prev[0]-coord[0][0]))
                noise[1] = max(noise[1], abs(prev[1]-coord[0][1]))
            ch = cv2.waitKey(100)
            if ch & 0xFF == 27:         # escape
                break
            elif ch == 1113938:         # Upkey
                self.increaseDepth(5)
                count = 0
            elif ch == 1113940:         # DownKey
                self.decreaseDepth(5)
                count = 0
            elif ch == 1113937:         # left
                self.decreaseDepth(20)
                count = 0
            elif ch == 1113939:         # right
                self.increaseDepth(20)
                count = 0
            elif ch & 0xFF == 32:              # space bar
                capture = not capture
                if capture:
                    video = cv2.VideoWriter('temp.avi',-1,self.ledCam.fps,self.ledCam.resolution)
                    print("RECORDING")
                else:
                    video.release()
                    print("END RECORDING")
            elif ch == 1048695:         # w
                self.increaseDepth(50)
                count = 0
            elif ch == 1048691:         # s
                self.decreaseDepth(50)
                count = 0
            elif ch == 1048673:         # a
                self.increaseDepth(500)
                count = 0
            elif ch == 1048676:         # d  
                self.decreaseDepth(500)  
                count = 0
            elif ch == 1048688:         # p
                print(self.currentDepth, coord[0])
        dc.cv2CloseWindow('tracking')
        dc.cv2CloseWindow('Blob Detector')
        print('Noise floor = %s'%noise)
        #self.ledCam.close()
    def getData(self,loops):
        data = np.zeros((loops,5),np.float)
        
        for i in range(loops):
            #self.sendSignal(self.ventSignal)
            #time.sleep(2)
            #self.sendSignal(self.ventCloseSignal)
            self.decreaseDepth(decrease)
            time.sleep(1)
            seconds = random.random()/5.
            a,t,clean = self.measureLatency(24,seconds)
            s = np.argmin(a)
            e = np.argmax(a[s:])
            if clean:
                data[i,:] = (seconds,a[s],a[e],t[s],t[e])
                decrease = 1000
            else:
                decrease = 3000
        #mask = np.invert(np.any(data < 0,axis=1))
        mask = np.any(data!=0,axis=1)
        #return data[mask,:]
        return data
        # If you wanted to know how to do math on the arrays:
        #ready = data.copy()
        #ready[:,2] = data[:,2] - data[:,0]
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
