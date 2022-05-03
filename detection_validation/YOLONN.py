import cv2 as cv
import os
import re
import numpy as np
import matplotlib.pyplot as plt

#simple method to rescale pictures
def rescale(pic, scale=0.5):
    width = int(pic.shape[0] * scale)
    height = int(pic.shape[1] * scale)
    dimensions = (width, height)
    return cv.resize(pic, dimensions, interpolation=cv.INTER_AREA)

net = cv.dnn.readNetFromDarknet("yolov3_custom.cfg","yolov3_custom_last_heavy.weights")

#classes = ['Diamond']

testing_dir = "test"
results_dir = "detected"
pic_res_dir = "pictures_with_detections"

testing_set = [] #the complete list of jpg in the test directory
labels = [] #the complete list of txt in the test directory
counter = 0
for files in os.listdir(testing_dir):
    counter += 1
    if re.match(".*\.jpg",files):
        testing_set += [files[:-4]]
print("Files found in " + testing_dir + ": " + str(counter))
if len(testing_set) == counter/2:
    print("testing-set correctly initialized")

for pic in range(len(testing_set)):
    print("Picture " + str(pic+1) + "   of " + str(int(counter/2)))

    picture_filename = testing_set[pic] + ".jpg"
    bbox_filename = testing_set[pic] + ".txt"
    testing_pic = os.path.join(testing_dir, picture_filename)
    pic_with_det = os.path.join(pic_res_dir, picture_filename)
    result_txt = os.path.join(results_dir, bbox_filename)
    
    img = cv.imread(testing_pic)
    width,height,channels = img.shape
    blob = cv.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_name)
    
    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.6: #0.7
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3]* height)
                x = int(cx - w/2)
                y = int(cy - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
                
    indexes = cv.dnn.NMSBoxes(boxes,confidences,.5,.4)
    """
    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.2:
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3]* height)
                x = int(cx - w/2)
                y = int(cy - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes,confidences,.8,.4)
    """
    f = open(result_txt, 'w')
    if  len(indexes) > 0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            confidence = str(round(confidences[i],2))
            cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            f.write(str(int(0)) + " " + str(confidence) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
            #cv.putText(img, confidence, (x,y+400),(0,0,255),2)
    f.close()
    cv.imshow('img',rescale(img))
    cv.imwrite(pic_with_det, img)
    cv.waitKey(1)

cv.destroyAllWindows()
