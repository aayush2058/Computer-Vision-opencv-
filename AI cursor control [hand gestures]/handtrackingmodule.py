import cv2
import mediapipe as mp
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self, image, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)

            bbox = xmin, ymin, xmax, ymax   # bounding box points

            if draw:
                cv2.rectangle(image, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmlist, bbox


    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


    def distanceFinder(self, point1, point2, image, draw = True):
        x1, y1 = self.lmlist[point1][1], self.lmlist[point1][2]  # x coordinates, y coordinates
        x2, y2 = self.lmlist[point2][1], self.lmlist[point2][2]  # x coordinates, y coordinates
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # center of x coordinates and y coordinates

        length = math.hypot(x2 - x1, y2 - y1)  # length of the line (length between two points)
        # print(length)

        if draw:
            cv2.circle(image, (x1, y1), 12, (0, 200, 0), cv2.FILLED)  # Creates circle at x coordinates
            cv2.circle(image, (x2, y2), 12, (0, 200, 0), cv2.FILLED)  # Creates circle at y coordinates
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)  # creates a line from x to y
            cv2.circle(image, (cx, cy), 9, (0, 200, 0), cv2.FILLED)  # creates a center point circle

        return length, image, [x1, y1, x2, y2, cx, cy]