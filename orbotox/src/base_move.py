#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import time
import math

a  = 0
roll = pitch = yaw = radian = yaw_round = rad = pose_x = pose_y = 0.0


def get_angle(radians):
    case = {
        '0.0' : -1.57,
        '0.01' : -1.57,
        '-0.0' : -1.57, 
        '-1.56' : 3.13, 
        '-1.57' : 3.13, 
        '-1.58' : 3.13,  
        '3.13' :  1.57,
        '1.57' : 0.0,
        '1.58' : 0.0,
        '1.56' : 0.0,
        '-3.14' : 1.57,
        '-3.13' : 1.57
        }
    return case[str(radians)] 


def get_rotation (msg):
    global roll, pitch, yaw, yaw_round, pose_x, pose_y
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
    yaw_round = round(yaw, 2)
    pose_q = msg.pose.pose.position
    pose_x = round(pose_q.x, 2)
    pose_y = round(pose_q.y, 2)


def callback(msg):
    global a, radian
    kP = 2
    r  =  round(msg.ranges[0], 3)
    m  =  round(msg.ranges[360], 3)
    m1 = round(min(msg.ranges[288:423]), 3)
    l  =  round(msg.ranges[719], 3)

    if(a == 0):
        if(l > 1.200):
            #print("Left  " + str(l) + " | Pose Y  " + str(pose_y))
            move.linear.x = 0
            move.angular.z = 0
            move.linear.y = 0.3
        elif(l < 1.100):
            move.linear.x = 0
            move.angular.z = 0
            move.linear.y = -0.3
        elif(1.100 < l < 1.200):
            print("Pose Y  " + str(pose_y))
            print("A = 1")
            print(l)
            a = 1

    if(a == 1):
        if(m1 > 1.000):
            move.linear.x = 1
            move.angular.z = 0
            move.linear.y = 0
        elif(m1 < 1.000):
            print("A = 2")
            print(m1)
            a = 2

    if(a == 2):
        if(m1 < 1.000):
            move.linear.x = 0
            move.linear.y = 0
            print("Inside a = " + str(m1))
            radian = get_angle(yaw_round)
            print("Radian Fetched")
            print("Entering While Loop")
            move.angular.z = round(kP * (radian-yaw_round), 2)
            while(move.angular.z!=0):
                move.linear.x = 0
                move.linear.y = 0
                print("Z right = " + str(move.angular.z) + "  Radian  " + str(radian) + " Yaw = " + str(yaw_round))
                move.angular.z = round(kP * (radian-yaw_round), 2)
                pub.publish(move)
            print(m1)
            a = 0
        else:
            a = 1
    pub.publish(move)


rospy.init_node('move_node')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
rate = rospy.Rate(2) 
move = Twist()
sub = rospy.Subscriber('/orbotox/laser/scan', LaserScan, callback, queue_size=1)
rot = rospy.Subscriber ('/odom', Odometry, get_rotation)
rospy.spin()
