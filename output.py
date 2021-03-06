import cv2
import numpy as np
import os

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("D:\\Py\\haarcascade_frontalface_alt.xml")
skip=0
face_data=[]
dataset_path='D:\\Py\\face-recog'  
labels=[]
class_id=0
names={}  # dic
    
#data preparation
for fx in os.listdir(dataset_path):
    
    if fx.endswith('.npy'):
        
        #create a mapping btw classid and name
        
        names[class_id]=fx[:-4]
        print("Loaded "+fx)
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)
        
        #create labels for the class
        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)

face_dataset=np.concatenate(face_data,axis=0)

face_labels=np.concatenate(labels,axis=0).reshape((-1,1))
print(face_dataset.shape)
print(face_labels.shape)


trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)
            
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("D:\\Py\\haarcascade_frontalface_alt.xml")
skip=0
face_data=[]
dataset_path='D:\\Py\\face-recog'
labels=[]
class_id=0
names={}
    
#data preparation

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        #create a mapping btw classid and name
        names[class_id]=fx[:-4]
        print("Loaded "+fx)
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)
        
        #create labels for the class
        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)

face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))
print(face_dataset.shape)
print(face_labels.shape)


trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)
         
# testing : and calling to KNN         
        
while True:        
        ret,frame=cap.read()
        if ret==False:
            continue
        
        faces=face_cascade.detectMultiScale(frame,1.3,5)


        for face in faces:
            x,y,w,h=face
        
            #get the face ROI
            offset=10
            face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
            face_section=cv2.resize(face_section,(100,100))
        
            #predicted label(out)
            out=knn(trainset,face_section.flatten())
        
        #display on the screen the name rectangle around it
        
            pred_name=names[int(out)]
            cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    
        cv2.imshow("Face",frame)
    
        key=cv2.waitKey(1) & 0xff
        if key==ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
