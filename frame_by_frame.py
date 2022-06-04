import cv2
import numpy as np
import os
cap=cv2.VideoCapture(0)

#face detection
face_cascade=cv2.CascadeClassifier("D:\\Py\\haarcascade_frontalface_alt.xml")

while True:
    ret,frame=cap.read()
    
    if ret==False:
        continue
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    print(faces)
    
    for face in faces:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    
    cv2.imshow("Frame",frame)
    
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


# In the output we are geting the Frame dimension ,along with the online vedion stream
    
