#!/usr/bin/env python3

import cv2
import rospy
import roslib
from geometry_msgs.msg import Twist
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
x1 = x2 = x3 = 0
arr = []
def callback1(msg):
    global x1
    x1 = msg.data
    arr.append(int(x1))

def callback2(msg):
    global x2
    x2 = msg.data
    arr.append(int(x2))

def callback3(msg):
    global x3
    x3 = msg.data
    arr.append(int(x3))
    print(sum(arr))

rospy.init_node('Listener', anonymous=True)
sub1 = rospy.Subscriber('/biscuit_1', String, callback1, queue_size=1)
sub2 = rospy.Subscriber('/biscuit_2', String, callback2, queue_size=1)
sub3 = rospy.Subscriber('/biscuit_3', String, callback3, queue_size=1)
rospy.spin()