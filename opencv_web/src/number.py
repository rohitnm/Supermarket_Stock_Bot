#!/usr/bin/env python

import sys
import rospy
import roslib
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
bridge = CvBridge()

def callback(img_msg):
    rospy.loginfo(img_msg.header)
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    gray = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    samples =  np.empty((0,100))
    responses = []
    keys = [i for i in range(48,58)]

    for cnt in contours:
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)

            if  h>28:
                cv2.rectangle(cv_image),(x,y),(x+w,y+h),(0,0,255),2
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                cv2.imshow('norm',cv_image)
                key = cv2.waitKey(0)

                if key == 27:  # (escape to quit)
                    sys.exit()
                elif key in keys:
                    responses.append(int(chr(key)))
                    sample = roismall.reshape((1,100))
                    samples = np.append(samples,sample,0)

    responses = np.array(responses,np.float32)
    responses = responses.reshape((responses.size,1))
    print("training complete")

    np.savetxt('generalsamples.data',samples)
    np.savetxt('generalresponses.data',responses)

rospy.init_node('Number_Detection', anonymous=True)
sub_blue = rospy.Subscriber("/orbotox/camera1/image_raw", Image, callback)
rospy.spin()