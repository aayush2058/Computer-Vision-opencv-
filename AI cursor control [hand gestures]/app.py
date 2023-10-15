import cv2
import numpy as np
import handtrackingmodule as htm
import pyautogui
import time

###################################
wCam, hCam = 640, 480
###################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, wCam)

pTime = 0
detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size() # gets the size of the screen
print (wScr, hScr)
frameR = 150 # frame reduction
smoothening = 2.3
plocX, plocY = 0, 0     # previous location of x, y
clocX, clocY = 0, 0     # current location of X, Y

while True:

    # 1. Find hand landmarks
    success, img = cap.read()
    img = detector.handsFinder(img)
    lmlist, bbox = detector.positionFinder(img)

    # 2. Get the tip of the index and the middle finger
    if len(lmlist)!=0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]
        print(x1, y1)
        print(x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)

        # Draws a rectangle for gesture to be used
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (200, 0, 200), 2)

        # 4. Only index finger: moving mode
        if fingers[1] == 1 and fingers[2] == 0:

            # 5. Convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr)) # [normalization] converts wCam into the width of the screen
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 6. smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening


            # 7. Move mouse
            pyautogui.moveTo(wScr - clocX, clocY, 0) # wScr - x3 flips the movement , 0.1 is the speed
            pyautogui.FAILSAFE = False
            cv2.circle(img, (x1, y1), 13, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY


        # 8, Both index and middle fingers are up: Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:

            # 9. Find distance between fingers
            length, img, lineinfo = detector.distanceFinder(8, 12, img)
            print(length)

            # 10. Click mouse if distance short
            if length < 42:
                cv2.circle(img, (lineinfo[4], lineinfo[5]), 13, (0, 0, 200), cv2.FILLED)
                pyautogui.click(button = 'left')


    # 11. Frame rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'fps: {str(int(fps))}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 0), 3)

    # 12. Display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()