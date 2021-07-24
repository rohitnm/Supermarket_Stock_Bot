#!/usr/bin/env python3

import cv2
import rospy
import roslib
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
bridge = CvBridge()


def blue_callback(img_msg):
    rospy.loginfo(img_msg.header)
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    
    blue_shape = 0
    lwb = np.array([100, 0, 0])
    uwb = np.array([255, 0, 0])
    blue_mask = cv2.inRange(cv_image, lwb, uwb)
    blurred = cv2.GaussianBlur(blue_mask, (5, 5), 4)
    _, threshold = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        blue_shape += 1
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        cv2.drawContours(cv_image, [approx], 0, (80, 255, 246), 2)
        cv2.putText(cv_image, str(blue_shape), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 246), 2)
        if len(approx) == 4:
            x1, y1, w, h = cv2.boundingRect(approx)
    if blue_shape < 2:
        msg1.data = "Object is below threshold, Please Refill. Count - " + str(blue_shape)
    else:
        msg1.data = str(blue_shape)
    pub1.publish(msg1)


def green_callback(img_msg):
    rospy.loginfo(img_msg.header)
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)

    green_shape = 0
    lwg = np.array([0, 50, 0])
    uwg = np.array([0, 150, 0])
    green_mask = cv2.inRange(cv_image, lwg, uwg)
    blurred = cv2.GaussianBlur(green_mask, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        green_shape += 1
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        cv2.drawContours(cv_image, [approx], 0, (80, 255, 246), 2)
        cv2.putText(cv_image, str(green_shape), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 246), 2)
        if len(approx) == 4:
            x1, y1, w, h = cv2.boundingRect(approx)
    if green_shape < 4:
        msg2.data = "Object is below threshold, Please Refill. Count - " + str(green_shape)
    else:
        msg2.data = str(green_shape)
    pub2.publish(msg2)


def red_callback(img_msg):
    # log some info about the image topic
    rospy.loginfo(img_msg.header)

    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    red_shape = 0
    lwr = np.array([0, 0, 100])
    uwr = np.array([0, 0, 255])
    red_mask = cv2.inRange(cv_image, lwr, uwr)
    blurred = cv2.GaussianBlur(red_mask, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        red_shape += 1
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        cv2.drawContours(cv_image, [approx], 0, (80, 255, 246), 2)
        cv2.putText(cv_image, str(red_shape), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 246), 2)
        if len(approx) == 4:
            x1, y1, w, h = cv2.boundingRect(approx)
    if red_shape < 3:
        msg3.data = "Object is below threshold, Please Refill. Count - " + str(red_shape)
    else:
        msg3.data = str(red_shape)
    pub3.publish(msg3)


msg1 = String()
msg2 = String()
msg3 = String()
rospy.init_node('Detection', anonymous=True)
pub1 = rospy.Publisher('/blue_count', String, queue_size=10)
pub2 = rospy.Publisher('/green_count', String, queue_size=10)
pub3 = rospy.Publisher('/red_count', String, queue_size=10)
sub_blue = rospy.Subscriber("/orbotox/camera1/image_raw", Image, blue_callback)
sub_green = rospy.Subscriber("/orbotox/camera2/image_raw", Image, green_callback)
sub_red = rospy.Subscriber("/orbotox/camera3/image_raw", Image, red_callback)
rospy.spin()