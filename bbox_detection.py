#Task 1
#DTP
#Goal: localize gems in a picture

import os
import re
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
    
#simple method to rescale pictures
def rescale(pic, scale=0.5):
    width = int(pic.shape[0] * scale)
    height = int(pic.shape[1] * scale)
    dimensions = (width, height)
    return cv.resize(pic, dimensions, interpolation=cv.INTER_AREA)

#this method reads the file in which are saved the positions of the gems and add
#the markers to the picture
#it returns then a list containing coordinates of every marker
def addmarkers(pic, marker_file):
    width = pic.shape[0]
    height = pic.shape[1]
    f = open(marker_file, 'r')
    all_markers = []

    for lines in f:
        markers = np.array(lines.split(' ')).astype(float)
        deltax = int(width*markers[3])
        deltay = int(height*markers[4])
        cx = int(width*markers[1])
        cy = int(height*markers[2])
        all_markers += [(cx, cy, deltax, deltay)]
        cv.rectangle(pic, (cx - int(deltax/2),cy - int(deltay/2)), (cx + int(deltax/2),cy + int(deltay/2)), (0,0,255), thickness=1)
    f.close()
    return all_markers

def savemarkers(markers, filename="markers.txt"):
    f = open(filename, 'w')
    for i in range(2,markers.max()+1,1):
        i_gem = np.zeros(markers.shape)
        i_gem[markers == i] = 1
        bin_pic = np.array(i_gem).astype(int)
        width = bin_pic.shape[0]
        height = bin_pic.shape[1]
        x_proj = bin_pic.sum(axis=0)
        y_proj = bin_pic.sum(axis=1)
        first_x = -1
        first_y = -1
        for j in range(len(x_proj)):
            if first_x == -1 and x_proj[j] != 0:
                first_x = j
            elif first_x != -1 and x_proj[j] == 0:
                last_x = j - 1
                break
        for j in range(len(y_proj)):
            if first_y == -1 and y_proj[j] != 0:
                first_y = j
            elif first_y != -1 and y_proj[j] == 0:
                last_y = j - 1
                break
        cx = ((first_x + last_x)/2)/width
        cy = ((first_y + last_y)/2)/height
        deltax = (last_x-first_x)/width
        deltay = (last_y-first_y)/height
        f.write(str(int(0)) + " " + str(0.5) + " " + str(int(first_x)) + " " + str(int(first_y)) + " " + str(int(last_x-first_x)) + " " + str(int(last_y-first_y)) + "\n") #<class> <confidence> <left> <top> <width> <height>
    f.close()
                
                

testing_dir = os.path.join("dataset", "test") #directory containing the testing-set
results_dir = "results"

testing_set = [] #the complete list of jpg in the test directory
labels = [] #the complete list of txt in the test directory
counter = 0
for files in os.listdir(testing_dir):
    counter += 1
    if re.match(".*\.jpg",files):
        testing_set += [files]
    else:
        labels += [files]
print("Files found in " + testing_dir + ": " + str(counter))
if len(testing_set) == len(labels) & len(testing_set) == counter/2:
    print("testing-set correctly initialized")

for pic in range(len(testing_set)):
    print("Picture " + str(pic+1) + "   of " + str(int(counter/2)))
    
    testing_pic = os.path.join(testing_dir, testing_set[pic])
    result_txt = os.path.join(results_dir, labels[pic])

    img = cv.imread(testing_pic)
    #img = rescale(img)
    #cv.imshow('Original picture', img)

    gray_lvl_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_lvl_img = cv.equalizeHist(gray_lvl_img)

    threshold = 250
    thresh, thresholded = cv.threshold(gray_lvl_img, threshold, 255, cv.THRESH_BINARY)
    smoothed = cv.medianBlur(thresholded, 11)
    #cv.imshow('thresholded', smoothed)

    # opening phase
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(smoothed,cv.MORPH_OPEN,kernel, iterations = 2)
    #cv.imshow('opening', opening)

    # background pixels
    bg = cv.dilate(opening,kernel,iterations=3)
    #cv.imshow('bg', bg)

    # foreground pixels
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,3)
    thresh, fg = cv.threshold(dist_transform,0.65*dist_transform.max(),255,0)
    #cv.imshow('fg', fg)

    # unknown region
    fg = np.uint8(fg)
    unknown = cv.subtract(bg, fg)
    #cv.imshow('unknown', unknown)

    # Marker labelling
    ret, markers = cv.connectedComponents(fg)

    # Add one to all labels in order to make background 1, instead of 0
    markers = markers+1
    # Now, we mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(img,markers)
    savemarkers(markers, filename=result_txt)
