import cv2
import time
import numpy as np
import handtrackingmodule as htm
import math

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# height and width of cam
wCam, hCam = 720, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0 # previous time

# importing class from another module
detector = htm.handDetector(detectionCon=0.7, maxHands = 1)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(0, None) # 0 is the max volume in this line only
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400  # default value when window opens
volPer = 0 # volume percentage
area = 0    # bounding box area
bounding_box_usable_when = 0
colorVolume = (200, 0, 0)



while True:
    _, img = cap.read()
    img = detector.handsFinder(img)
    lmlist, bbox = detector.positionFinder(img, draw = True)

    if len(lmlist) != 0:
        # print(lmlist[4], lmlist[8])
        '''
        If we run this without the if condition, we get error
        because in the first loop there is no lmlist[2]. It comes after looping two or 
        more times. So, if condition prevents from throwing an error on the first loop because the condition is false.
        '''

        # Filter based on size
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        bounding_box_usable_when = area // 100
        print(bbox)
        if 200 < bounding_box_usable_when < 1000:

            # Find the distance between the Index and the Thumb
            length, image, extra_points = detector.distanceFinder(point1 = 4, point2 = 8, image = img)
            cx, cy = extra_points[4], extra_points[5]

            # Convert volume
            # Hand range ( 50 - 220 )
            # vol = np.interp(length, [0, 180], [minVol, maxVol])  # Actual volume
            volBar = np.interp(length, [50, 180], [400, 150])  # Volume level display on bar
            volPer = np.interp(length, [50, 180], [0, 100])  # Volume percentage

            # Reduce Resolution to make it smoother
            smoothness = 10
            volPer = smoothness * round(volPer/smoothness)
            print(vol, length)

            # Setting master volume

            # volume.SetMasterVolumeLevel(vol, None)
            # Check fingers up
            fingers = detector.fingersUp()
            print(fingers)

            # If distance exceeds maximum or minimum threshold, indicate with a red circle
            if length < 50 or length >= 180:
                cv2.circle(image, (cx, cy), 9, (0, 0, 200), cv2.FILLED)

            # If pinky is down, set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                cv2.circle(image, (cx, cy), 9, (200, 0, 0), cv2.FILLED)
                colorVolume = (0, 200, 0)
                time.sleep(0.04)
            else:
                colorVolume = (200, 0, 0)

            # Frame rate
            cTime = time.time()  # current Time
            fps = 1 / (cTime - pTime)
            pTime = cTime


            # Drawings [ if inside the if statement, shows only if the hand is detected. Else kept outside, shows all the time. ]
            cv2.rectangle(img, (50, 150), (85, 400), (0, 200, 0), 2)  # Creating a bar
            if volPer == 0:
                cv2.rectangle(img, (50, int(volBar)), (85, 405), (0, 0, 200), cv2.FILLED)  # Creating a volume fill
                cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_ITALIC, 1, (0, 0, 200), 2)
            else:
                cv2.rectangle(img, (50, int(volBar)), (85, 402), (0, 200, 0), cv2.FILLED)  # Creating a volume fill
                cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_ITALIC, 1, (0, 200, 0), 2)

            cVol = int(volume.GetMasterVolumeLevelScalar()*100)
            cv2.putText(img, f'Vol: {int(cVol)}%', (400, 50), cv2.FONT_ITALIC, 1, colorVolume, 2)

            cv2.putText(img, f'fps: {int(fps)}', (40, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)




    cv2.imshow("Img", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;


