import cv2
import numpy as np
import time
import os
import handtrackingmodule as htp

folderPath = "header_img"
myList = os.listdir(folderPath)
print(myList)

overLayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)

header = overLayList[0]
drawColor = (56, 52, 204) # default is red
brushThickness = 15
eraserThickness = 100
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
pTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htp.handDetector(detectionCon=0.7) # This should be kept outside the loop to avoid lag

while True:

    # Import image
    success, img = cap.read()
    img = cv2.flip(img, 1) # flipping image to match directions

    # Find hand landmarks
    img = detector.handsFinder(img, draw=True)
    lmlist, bbox = detector.positionFinder(img, draw = False)

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # If selection mode - two fingers are up
        if fingers[1] and fingers[2]:
            print('selection mode')
            print (x1, y1)
            xp, yp = 0, 0
            if y1 < 115:
                # select red
                if 145 < x1 < 250:
                    header = overLayList[0]
                    drawColor = (56, 52, 204)
                # select green
                elif 275 < x1 < 350:
                    header = overLayList[1]
                    drawColor = (97, 181, 3)
                # select blue
                elif 425 < x1 < 510:
                    header = overLayList[2]
                    drawColor = (202, 97, 65)
                # erase
                elif 1000 < x1 < 1150:
                    header = overLayList[3]
                    drawColor = (0, 0, 0)

            # cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # index finger is up for drawing
        if fingers[1] == True and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print('drawing mode')

            # Drawing into the canvas
            # xp, yp = x previous point, y previous point
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInverse)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting header image
    img[0:103, 0:1280] = header

    # Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'fps: {str(int(fps))}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1.6, (200, 0, 0), 3)

    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()

