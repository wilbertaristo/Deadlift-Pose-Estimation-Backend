import os
import re
from os.path import isfile, join
import argparse
import math

test_image_path = "./detection/testing"
classes_path = "./detection/classes.txt"
weights_path = "./detection/yolov4-tiny-custom_best.weights"
testing_cfg_path = './detection/yolov4-tiny-custom.cfg'

# sorting test files based on ascending order of names
# images = os.listdir(test_image_path)
# images.sort(key=lambda f : int(re.sub('\D', '', f)))

# testing object detection with a single image
import cv2
import numpy as np
import os
import re
import cv2
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Running yolo_v4 tiny model")
parser.add_argument("--cam", action='store_true')
parser.add_argument("--video", help="input video")
parser.add_argument("--save", action='store_true')

args = parser.parse_args()


net = cv2.dnn.readNet(weights_path, testing_cfg_path)
classes = []
with open(classes_path, "r") as f:
    classes = f.read().splitlines()

font = cv2.FONT_HERSHEY_SIMPLEX
colors = np.random.uniform(0, 255, size=(100, 3))
count_list = []
# test with a single image
#img = cv2.imread("./detection/input/good_0.png")

# test with a video
if args.video:
    print(f'video used: {args.video}')
    cap = cv2.VideoCapture(f'./detection/input/{args.video}')
# test with webcam
if args.cam:
    print(f'live cam on')
    cap = cv2.VideoCapture(0) #0=front-cam, 1=back-cam
    
# to save video
if args.save:
    print(f'saving video')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if args.video:
        out = cv2.VideoWriter(f'./detection/output/out_{args.video}', fourcc, 30.0, (540,960))
    if args.cam:
        out = cv2.VideoWriter(f'./detection/output/live.mp4', fourcc, 30.0, (540,960))

threshold = 0.5
while True:
    _,img = cap.read()
    if img is None:
        break
    height, width, _ = img.shape
    #new_width = math.floor(height/1920*1080)
    #img = img[:, (width-new_width)//2 : new_width + (width-new_width)//2]
    height, width, _ =  img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False) #convert image to fit into the model
    net.setInput(blob) #fit image into the model
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names) #forward pass to get outputs

    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs: # for each box segment
        for detection in output: # for each object detected
            # detection[0:3] first 4 values represents the bounding boxes, detection[4] confidence scores
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > threshold:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.4)
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))

            # set red color for rounded back
            if (label == 'rounded_back'):
                color = [0,0,255]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, f'{label}', (x, y-35), font, 1, color, 2)
                cv2.putText(img, f'{confidence}', (x, y-5), font, 1, color, 2)
            # set green color for straight back
            elif (label == 'straight_back'):
                color = [0,255,0]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, f'{label}', (x, y-35), font, 1, color, 2)
                cv2.putText(img, f'{confidence}', (x, y-5), font, 1, color, 2)
    img = cv2.resize(img,(540,960))
    if args.save:
        out.write(img)
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()



#read img
#height, width, _ = img.shape
#blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
#net.setInput(blob)
#output_layers_names = net.getUnconnectedOutLayersNames()
#layerOutputs = net.forward(output_layers_names)
#boxes = []
#confidences = []
#class_ids = []
#print(layerOutputs)
#for output in layerOutputs:
# for detection in output:
#     scores = detection[5:]
#     class_id = np.argmax(scores)
#     confidence = scores[class_id]
#     if confidence > threshold:
#         center_x = int(detection[0]*width)
#         center_y = int(detection[1]*height)
#         w = int(detection[2]*width)
#         h = int(detection[3]*height)
#
#         x = int(center_x - w/2)
#         y = int(center_y - h/2)
#
#         boxes.append([x, y, w, h])
#         confidences.append((float(confidence)))
#         class_ids.append(class_id)
#
#indexes = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.4)
#if len(indexes)>0:
# for i in indexes.flatten():
#     x, y, w, h = boxes[i]
#     label = str(classes[class_ids[i]])
#     confidence = str(round(confidences[i],2))
#     if (label == 'rounded_back'):
#         color = [0,0,255]
#         cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
#         cv2.putText(img, f'{label}', (x, y-35), font, 1, color, 2)
#         cv2.putText(img, f'{confidence}', (x, y-5), font, 1, color, 2)
#     elif (label == 'straight_back'):
#         color = [0,255,0]
#         cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
#         cv2.putText(img, f'{label}', (x, y-35), font, 1, color, 2)
#         cv2.putText(img, f'{confidence}', (x, y-5), font, 1, color, 2)
#
#cv2.imshow('image',img)
#cv2.waitKey(0)



