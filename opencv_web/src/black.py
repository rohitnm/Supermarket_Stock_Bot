#!/usr/bin/env python3

import cv2
import rospy
import roslib
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
bridge = CvBridge()

def callback1(msg):
    value = msg.linear.x
    print(value)

def callback(img_msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    img = cv2.resize(cv_image,(1280,720))
    lwb = np.array([0, 0, 0])
    uwb = np.array([0, 0, 0])
    blue_mask = cv2.inRange(img, lwb, uwb)
    blurred = cv2.GaussianBlur(blue_mask, (5, 5), 4)
    _, threshold = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        cv2.drawContours(img, [approx], 0, (80, 255, 246), 2)
        #cv2.putText(cv_image, str(blue_shape), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 246), 2)
        if len(approx) == 4:
            x1, y1, w, h = cv2.boundingRect(approx)

rospy.init_node('Detection', anonymous=True)
sub = rospy.Subscriber("/orbotox/camera1/image_raw", Image, callback)
sub1 = rospy.Subscriber("/cmd_vel", Twist, callback1, queue_size=1)
rospy.spin()