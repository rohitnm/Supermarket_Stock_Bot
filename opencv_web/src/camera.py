#!/usr/bin/env python3

import cv2
import rospy
import roslib
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
bridge = CvBridge()
import matplotlib.pyplot as plt

classes = ['Beer_Can','Biscuit_Box','Milk_Carton',
'Salt_Pack',
'Tea_Time',
'Tomato_Sauce',
'Tomato_Soup',
'Volkorn_Toast']

net = cv2.dnn.readNetFromDarknet("/home/rohit/catkin_ws/src/opencv_web/src/yolov3_custom_new.cfg", "/home/rohit/catkin_ws/src/opencv_web/src/yolov3_custom_recent.weights")

def object_callback(img_msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    
    img = cv2.resize(cv_image,(1280,720))
    height,width,_ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_name)

    boxes =[]
    confidences = []
    class_ids = []
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0
    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3]* height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.8,.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size =(len(boxes),3))
    if  len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            #label = str(classes[class_ids[i]])
            label = class_ids[i]
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            if(label==0):
                count0+=1
            if(label==1):
                count1+=1
            if(label==2):
                count2+=1
            if(label==3):
                count3+=1
            if(label==4):
                count4+=1
            if(label==5):
                count5+=1
            if(label==6):
                count6+=1
            if(label==7):
                count7+=1    
        cv2.putText(img,"Beer_Can =" + " " + str(count0), (50,50),font,2,color,2)
        cv2.putText(img,"Biscuit_Box =" + " " + str(count1), (50,100),font,2,color,2)
        cv2.putText(img,"Milk_Carton =" + " " + str(count2), (50,150),font,2,color,2)
        cv2.putText(img,"Salt_Pack =" + " " + str(count3), (350,50),font,2,color,2)
        cv2.putText(img,"Tea_Time =" + " " + str(count4), (350,100),font,2,color,2)
        cv2.putText(img,"Tomato_Sauce =" + " " + str(count5), (750,50),font,2,color,2)
        cv2.putText(img,"Tomato_Soup =" + " " + str(count6), (750,100),font,2,color,2)
        cv2.putText(img,"Volkorn_Toast =" + " " + str(count7), (750,150),font,2,color,2)
    cv2.imshow('img',img)
    cv2.waitKey(1)
    print("Beer_Can = " + str(count0))
    print("Biscuit_Box = " + str(count1))
    print("Milk = " + str(count2))
    print("Salt = " + str(count3))
    print("Tea = " + str(count4))
    print("Sauce = " + str(count5))
    print("Soup = " + str(count6))
    print("Toast = " + str(count7))
    print("====================================")

rospy.init_node('Detection', anonymous=True)
sub_objects = rospy.Subscriber("/orbotox/camera1/image_raw", Image, object_callback)
rospy.spin()