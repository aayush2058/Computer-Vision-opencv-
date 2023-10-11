# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:15:52 2023

@author: picolo
"""



# ------------------------------------------
## CASCADE works on black and white images
# ------------------------------------------


import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


# Defining a function that will do the detections
def detect(gray, frame): # gray is gray scale image of the original frame image.
    faces = face_cascade.detectMultiScale(image = gray,
                                          scaleFactor = 1.3,
                                          minNeighbors = 5)
    
    # Defining rectangle coordinates for the facial region.
    for (x, y, w, h) in faces:
        cv2.rectangle(img = frame, # want to draw rectangle on the frame
                      pt1 = (x, y), 
                      pt2 = (x+w, y+h), 
                      color = (255, 0, 0), 
                      thickness = 2)
        
        # roi = region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(image = roi_gray, # face is already detected and eyes are inside the face rectangle (helps lower computational cost))
                                            scaleFactor = 1.1,
                                            minNeighbors = 20)
        
        # Defining rectangle coordinates for eyes.
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img = roi_color, # want to draw rectangle on the colour frame
                          pt1 = (ex, ey),
                          pt2 = (ex+ew, ey+eh), 
                          color = (0, 255, 0), 
                          thickness = 2)


        smile = smile_cascade.detectMultiScale(image = roi_gray,
                                               scaleFactor = 1.7,
                                               minNeighbors = 42)

        # Defining rectangle coordinates for smile.
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(img = roi_color, # want to draw rectangle on the colour frame
                          pt1 = (sx, sy),
                          pt2 = (sx+sw, sy+sh),
                          color = (0, 0, 255),
                          thickness = 2)
    
    return frame

# Display live face recognition using webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    canvas = detect(gray = gray, frame = frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
video_capture.release()
cv2.destroyAllWindows()