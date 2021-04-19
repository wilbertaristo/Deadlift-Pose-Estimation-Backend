# testing object detection with a single image
import cv2
import numpy as np
import tensorflow as tf

# test_image_path = "./input.mp4"
classes_path = "./detection/classes.txt"
weights_path = "./detection/yolov4-tiny-custom_best.weights"
testing_cfg_path = "./detection/yolov4-tiny-custom.cfg"
model_path = "./detection/rep-counter2.h5"

labels = ['up', 'down', 'nothing']
img_size = 256

# Save images with shape (256, 256, 3)
def rescale_frame(frame):
  dim = (img_size, img_size)
  return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def run_yolo_prediction(args): 
  model = tf.keras.models.load_model(model_path)
  net = cv2.dnn.readNet(weights_path, testing_cfg_path)
  classes = []
  with open(classes_path, "r") as f:
    classes = f.read().splitlines()

  # test with a video
  if args['video']:
    print(f"video used: {args['video']}")
    cap = cv2.VideoCapture(f"./detection/input/{args['video']}")
      
  # to save video
  if args['save']:
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      if args['video']:
          out = cv2.VideoWriter(f"./detection/output/out_{args['video']}", fourcc, 30.0, (540,960))

  threshold = 0.5

  ret, frame1 = cap.read()
  prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
  hsv = np.zeros_like(frame1)
  hsv[...,1] = 255

  count = 0
  up_bool = False
  down_bool = False
  first_up = False
  frames = 0
  # good_back = 0
  # bad_back = 0
  num_labels = 0
  total_confidence = 0
  form = 100.0
  rep_form = []
  prvs_labels = []

  # Initialize text style
  height, width, _ = frame1.shape
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 2
  fontColor = (255,255,255)
  lineType = 2
  bottomLeftCornerOfText = (10, int(height - (height / 40 + fontScale * height / 40)))
  formPosition = (10, int(height - height / 40))

  while True:
      _,img = cap.read()
      if img is None:
          break

      # 1. REP COUNTER
      next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
      # Every 10 frames
      if (frames % 10 == 0):
          flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0) # What do these numbers represent?
          mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
          hsv[...,0] = ang*180/np.pi/2
          hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
          rgb1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
          rgb = rescale_frame(rgb1)
          rgb = rgb.reshape((1,256,256,3))

          prediction = model.predict(rgb, batch_size=1)
          predict_label = np.argmax(prediction, axis=-1)
          label = labels[int(predict_label)]
          if len(prvs_labels) >= 3:
            prvs_labels.pop(0)
          prvs_labels.append(label)
          
          # Identify first rep (two up frames in a row)
          if prvs_labels[-2:] == ['up', 'up'] and not first_up:
              first_up = True
          # First down within the new rep
          elif prvs_labels[-2:] == ['down', 'down'] and first_up:
              down_bool = True
          # First up within the new rep
          elif prvs_labels[0] == 'down' and 'down' not in prvs_labels[-2:] and down_bool:
              up_bool = True
          
          # Reset and start a new rep
          if up_bool and down_bool:
              rep_form.append(form)
              count += 1
              up_bool = False
              down_bool = False
              num_labels = 0
              total_confidence = 0
              # good_back = 0
              # bad_back = 0
              form = 100.0
      
      # Add count text
      cv2.putText(img, f'Count: {count}', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
      
      # Good form defined as more than 70% good labels within given rep
      # Good form defined as an average confidence score of more than 0.5 within given rep
      if (form >= 50):
        formColor = [0,255,0]
      else:
        formColor = [0,0,255]

      # Add form (%) text
      cv2.putText(img, f'Form: {form}%', formPosition, font, fontScale, formColor, lineType)

      # Add form summary text
      for i in range(len(rep_form)):
        # Add form for each completed rep
        x = 10
        y = 30 + i * 25 * fontScale
        if rep_form[i] >= 50:
          repColor = [0,255,0]
        else:
          repColor = [0,0,255]

        cv2.putText(img, f'Rep {i+1}: {rep_form[i]}%', (x, y), font, fontScale * 0.5, repColor, lineType)
      
      # 2. ROUNDED BACK DETECTION
      height, width, _ = img.shape
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
      if len(indexes)>0 and first_up:
          for i in indexes.flatten():
              x, y, w, h = boxes[i]
              label = str(classes[class_ids[i]])
              confidence_score = round(confidences[i],2)
              confidence = str(confidence_score)

              # set red color for rounded back
              if (label == 'rounded_back'):
                  # bad_back += 1
                  total_confidence += 1 - confidence_score
                  num_labels += 1
                  color = [0,0,255]
                  cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                  cv2.putText(img, f'{label}', (x, y-35), font, 1, color, 2)
                  cv2.putText(img, f'{confidence}', (x, y-5), font, 1, color, 2)
              # set green color for straight back
              elif (label == 'straight_back'):
                  # good_back += 1
                  total_confidence += confidence_score
                  num_labels += 1
                  color = [0,255,0]
                  cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                  cv2.putText(img, f'{label}', (x, y-35), font, 1, color, 2)
                  cv2.putText(img, f'{confidence}', (x, y-5), font, 1, color, 2)
              # rep form (percentage of frames with good labels)
              # form = round(good_back / (good_back + bad_back) * 100, 2)

              # rep form (average confidence of good back within the rep)
              form = round(total_confidence / num_labels * 100, 2)

      img = cv2.resize(img,(540,960))
      if args['save']:
          out.write(img)
    #   cv2.imshow('Image', img)
      key = cv2.waitKey(1)
      if key==27:
          break
      prvs = next
      frames += 1
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



