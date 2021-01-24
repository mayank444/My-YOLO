import cv2
import numpy as np
cap = cv2.VideoCapture(0)

Wht = 320
confThreshold = 0.5
nmsThreshold = 0.3
classesFile = 'coco names.txt'
classNames = []

with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    ht,wt,ct = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence>confThreshold:
                w,h = int(det[2]*wt),int(det[3]*ht)
                x,y = int((det[0]*wt)-w/2),int((det[1]*ht)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    print(indices)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f"{classNames[classIds[i]].upper()}{int(confs[i]*100)}%",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

while True:
    success,img = cap.read()
    blob = cv2.dnn.blobFromImage(img,1/255,(Wht,Wht),[0,0,0],1,crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    # print(layerNames)

    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    # print(net.getUnconnectedOutLayers)

    outputs = net.forward(outputNames)

    findObjects(outputs,img)
    cv2.imshow('Image',img)
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()
