#!/usr/bin/env python3

import cv2
import rospy
import roslib
from geometry_msgs.msg import Twist
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
bridge = CvBridge()

movex = 0
movey = 0
movez = 0

def callback(msg):
    global movex, movey, movez
    movex = msg.linear.x
    movey = msg.linear.y
    movez = msg.angular.z

def black_detect(input_img):
    shape = 0
    img = cv2.resize(input_img,(1280,720))
    lwb = np.array([0, 0, 0])
    uwb = np.array([0, 0, 0])
    blue_mask = cv2.inRange(img, lwb, uwb)
    blurred = cv2.GaussianBlur(blue_mask, (5, 5), 4)
    _, threshold = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        shape+=1
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            x1, y1, w, h = cv2.boundingRect(approx)
    return shape

classes = ['Beer_Can','Biscuit_Box','Milk_Carton',
'Salt_Pack',
'Tea_Time',
'Tomato_Sauce',
'Tomato_Soup',
'Volkorn_Toast']

net = cv2.dnn.readNetFromDarknet("/home/rohit/catkin_ws/src/opencv_web/src/yolov3_custom_new.cfg", "/home/rohit/catkin_ws/src/opencv_web/src/yolov3_custom_recent.weights")
net2 = cv2.dnn.readNetFromDarknet("/home/rohit/catkin_ws/src/opencv_web/src/yolov3_custom_new.cfg", "/home/rohit/catkin_ws/src/opencv_web/src/yolov3_custom_recent.weights")
net3 = cv2.dnn.readNetFromDarknet("/home/rohit/catkin_ws/src/opencv_web/src/yolov3_custom_new.cfg", "/home/rohit/catkin_ws/src/opencv_web/src/yolov3_custom_recent.weights")
x1 = 0
y1 = 0
z1 = 0
c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = []
value = 0
def camera1_callback(img_msg):
    global value
    global c0, c1, c2, c3, c4, c5, c6, c7
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)
    
    value = black_detect(cv_image)
    if movex == 0.0 and movey == 0.0 and movez == 0.0:
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
                x1 = label
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
        if value == 0:
            if x1 == 0:
                msg1.data = str(count0)
                beer1.publish(msg1)
            elif x1 == 1:
                msg7.data = str(count1)
                bis1.publish(msg7)
            elif x1 == 2:
                msg13.data = str(count2)
                milk1.publish(msg13)
            elif x1 == 3:
                msg19.data = str(count3)
                salt1.publish(msg19)
            elif x1 == 4:
                msg25.data = str(count4)
                tea1.publish(msg25)
            elif x1 == 5:
                msg31.data = str(count5)
                sauce1.publish(msg31)
            elif x1 == 6:
                msg37.data = str(count6)
                soup1.publish(msg37)
            elif x1 == 7:
                msg43.data = str(count7)
                toast1.publish(msg43)
        elif value == 1:
            if x1 == 0:
                msg4.data = str(count0)
                beer4.publish(msg4)
            elif x1 == 1:
                msg10.data = str(count1)
                bis4.publish(msg10)
            elif x1 == 2:
                msg16.data = str(count2)
                milk4.publish(msg16)
            elif x1 == 3:
                msg22.data = str(count3)
                salt4.publish(msg22)
            elif x1 == 4:
                msg28.data = str(count4)
                tea4.publish(msg28)
            elif x1 == 5:
                msg34.data = str(count5)
                sauce4.publish(msg34)
            elif x1 == 6:
                msg40.data = str(count6)
                soup4.publish(msg40)
            elif x1 == 7:
                msg46.data = str(count7)
                toast4.publish(msg46)
        cv2.imshow('img',img)
        cv2.waitKey(1)
    else:
        cv2.destroyAllWindows()
        #print("Robot is Moving")
    

def camera2_callback(img_msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)

    if movex == 0.0 and movey == 0.0 and movez == 0.0:
        img = cv2.resize(cv_image,(1280,720))
        height,width,_ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

        net2.setInput(blob)

        output_layers_name = net2.getUnconnectedOutLayersNames()

        layerOutputs = net2.forward(output_layers_name)

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
                x1 = label
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
        if value == 0:
            if x1 == 0:
                msg2.data = str(count0)
                beer2.publish(msg2)
            elif x1 == 1:
                msg8.data = str(count1)
                bis2.publish(msg8)
            elif x1 == 2:
                msg14.data = str(count2)
                milk2.publish(msg14)
            elif x1 == 3:
                msg20.data = str(count3)
                salt2.publish(msg20)
            elif x1 == 4:
                msg26.data = str(count4)
                tea2.publish(msg26)
            elif x1 == 5:
                msg32.data = str(count5)
                sauce2.publish(msg32)
            elif x1 == 6:
                msg38.data = str(count6)
                soup2.publish(msg38)
            elif x1 == 7:
                msg44.data = str(count7)
                toast2.publish(msg44)
        elif value == 1:
            if x1 == 0:
                msg5.data = str(count0)
                beer5.publish(msg5)
            elif x1 == 1:
                msg11.data = str(count1)
                bis5.publish(msg11)
            elif x1 == 2:
                msg17.data = str(count2)
                milk5.publish(msg17)
            elif x1 == 3:
                msg23.data = str(count3)
                salt5.publish(msg23)
            elif x1 == 4:
                msg29.data = str(count4)
                tea5.publish(msg29)
            elif x1 == 5:
                msg35.data = str(count5)
                sauce5.publish(msg35)
            elif x1 == 6:
                msg41.data = str(count6)
                soup5.publish(msg41)
            elif x1 == 7:
                msg47.data = str(count7)
                toast5.publish(msg47)
    else:
        cv2.destroyAllWindows()
        #print("Robot is Moving")

def camera3_callback(img_msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        print(e)

    if movex == 0.0 and movey == 0.0 and movez == 0.0:
        img = cv2.resize(cv_image,(1280,720))
        height,width,_ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

        net3.setInput(blob)

        output_layers_name = net3.getUnconnectedOutLayersNames()

        layerOutputs = net3.forward(output_layers_name)

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
                x1 = label
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
        if value == 0:
            if x1 == 0:
                msg49.data = "NO"
                msg3.data = str(count0)
                beer3.publish(msg3)
                shelf15.publish(msg49)
            elif x1 == 1:
                msg50.data = "NO"
                msg9.data = str(count1)
                bis3.publish(msg9)
                shelf1.publish(msg50)
            elif x1 == 2:
                msg51.data = "NO"
                msg15.data = str(count2)
                milk3.publish(msg15)
                shelf3.publish(msg51)
            elif x1 == 3:
                msg52.data = "NO"
                msg21.data = str(count3)
                salt3.publish(msg21)
                shelf5.publish(msg52)
            elif x1 == 4:
                msg53.data = "NO"
                msg27.data = str(count4)
                tea3.publish(msg27)
                shelf7.publish(msg53)
            elif x1 == 5:
                msg54.data = "NO"
                msg33.data = str(count5)
                sauce3.publish(msg33)
                shelf9.publish(msg54)
            elif x1 == 6:
                msg55.data = "NO"
                msg39.data = str(count6)
                soup3.publish(msg39)
                shelf11.publish(msg55)
            elif x1 == 7:
                msg56.data = "NO"
                msg45.data = str(count7)
                toast3.publish(msg45)
                shelf13.publish(msg56)
        elif value == 1:
            if x1 == 0:
                msg57.data = "YES"
                msg6.data = str(count0)
                beer6.publish(msg6)
                shelf16.publish(msg56)
            elif x1 == 1:
                msg58.data = "YES"
                msg12.data = str(count1)
                bis6.publish(msg12)
                shelf2.publish(msg58)
            elif x1 == 2:
                msg59.data = "YES"
                msg18.data = str(count2)
                milk6.publish(msg18)
                shelf4.publish(msg59)
            elif x1 == 3:
                msg60.data = "YES"
                msg24.data = str(count3)
                salt6.publish(msg24)
                shelf6.publish(msg60)
            elif x1 == 4:
                msg61.data = "YES"
                msg30.data = str(count4)
                tea6.publish(msg30)
                shelf8.publish(msg61)
            elif x1 == 5:
                msg62.data = "YES"
                msg36.data = str(count5)
                sauce6.publish(msg36)
                shelf10.publish(msg62)
            elif x1 == 6:
                msg63.data = "YES"
                msg42.data = str(count6)
                soup6.publish(msg42)
                shelf12.publish(msg63)
            elif x1 == 7:
                msg64.data = "YES"
                msg48.data = str(count7)
                toast6.publish(msg48)
                shelf14.publish(msg64)
    else:
        cv2.destroyAllWindows()
        #print("Robot is Moving")

rospy.init_node('Detection', anonymous=True)

#----------------Beer Messages-------------
msg1 = String()
msg2 = String()
msg3 = String()
msg4 = String()
msg5 = String()
msg6 = String()
#--------------Biscuit Messages------------
msg7 = String()
msg8 = String()
msg9 = String()
msg10 = String()
msg11 = String()
msg12 = String()
#----------------Milk Messages-------------
msg13 = String()
msg14 = String()
msg15 = String()
msg16 = String()
msg17 = String()
msg18 = String()
#----------------Salt Messages------------
msg19 = String()
msg20 = String()
msg21 = String()
msg22 = String()
msg23 = String()
msg24 = String()
#-----------------Tea Messages------------
msg25 = String()
msg26 = String()
msg27 = String()
msg28 = String()
msg29 = String()
msg30 = String()
#--------------Sauce Messages------------
msg31 = String()
msg32 = String()
msg33 = String()
msg34 = String()
msg35 = String()
msg36 = String()
#--------------Soup Messages-------------
msg37 = String()
msg38 = String()
msg39 = String()
msg40 = String()
msg41 = String()
msg42 = String()
#-------------Toast Messages-------------
msg43 = String()
msg44 = String()
msg45 = String()
msg46 = String()
msg47 = String()
msg48 = String()
#-------------Toast Messages-------------
msg49 = String()
msg50 = String()
msg51 = String()
msg52 = String()
msg53 = String()
msg54 = String()
msg55 = String()
msg56 = String()
msg57 = String()
msg58 = String()
msg59 = String()
msg60 = String()
msg61 = String()
msg62 = String()
msg63 = String()
msg64 = String()

bis1 = rospy.Publisher('/biscuit_1', String, queue_size=10)
bis2 = rospy.Publisher('/biscuit_2', String, queue_size=10)
bis3 = rospy.Publisher('/biscuit_3', String, queue_size=10)
bis4 = rospy.Publisher('/biscuit_4', String, queue_size=10)
bis5 = rospy.Publisher('/biscuit_5', String, queue_size=10)
bis6 = rospy.Publisher('/biscuit_6', String, queue_size=10)

milk1 = rospy.Publisher('/milk_1', String, queue_size=10)
milk2 = rospy.Publisher('/milk_2', String, queue_size=10)
milk3 = rospy.Publisher('/milk_3', String, queue_size=10)
milk4 = rospy.Publisher('/milk_4', String, queue_size=10)
milk5 = rospy.Publisher('/milk_5', String, queue_size=10)
milk6 = rospy.Publisher('/milk_6', String, queue_size=10)

salt1 = rospy.Publisher('/salt_1', String, queue_size=10)
salt2 = rospy.Publisher('/salt_2', String, queue_size=10)
salt3 = rospy.Publisher('/salt_3', String, queue_size=10)
salt4 = rospy.Publisher('/salt_4', String, queue_size=10)
salt5 = rospy.Publisher('/salt_5', String, queue_size=10)
salt6 = rospy.Publisher('/salt_6', String, queue_size=10)

tea1 = rospy.Publisher('/tea_1', String, queue_size=10)
tea2 = rospy.Publisher('/tea_2', String, queue_size=10)
tea3 = rospy.Publisher('/tea_3', String, queue_size=10)
tea4 = rospy.Publisher('/tea_4', String, queue_size=10)
tea5 = rospy.Publisher('/tea_5', String, queue_size=10)
tea6 = rospy.Publisher('/tea_6', String, queue_size=10)

sauce1 = rospy.Publisher('/sauce_1', String, queue_size=10)
sauce2 = rospy.Publisher('/sauce_2', String, queue_size=10)
sauce3 = rospy.Publisher('/sauce_3', String, queue_size=10)
sauce4 = rospy.Publisher('/sauce_4', String, queue_size=10)
sauce5 = rospy.Publisher('/sauce_5', String, queue_size=10)
sauce6 = rospy.Publisher('/sauce_6', String, queue_size=10)

soup1 = rospy.Publisher('/soup_1', String, queue_size=10)
soup2 = rospy.Publisher('/soup_2', String, queue_size=10)
soup3 = rospy.Publisher('/soup_3', String, queue_size=10)
soup4 = rospy.Publisher('/soup_4', String, queue_size=10)
soup5 = rospy.Publisher('/soup_5', String, queue_size=10)
soup6 = rospy.Publisher('/soup_6', String, queue_size=10)

toast1 = rospy.Publisher('/toast_1', String, queue_size=10)
toast2 = rospy.Publisher('/toast_2', String, queue_size=10)
toast3 = rospy.Publisher('/toast_3', String, queue_size=10)
toast4 = rospy.Publisher('/toast_4', String, queue_size=10)
toast5 = rospy.Publisher('/toast_5', String, queue_size=10)
toast6 = rospy.Publisher('/toast_6', String, queue_size=10)

beer1 = rospy.Publisher('/beer_1', String, queue_size=10)
beer2 = rospy.Publisher('/beer_2', String, queue_size=10)
beer3 = rospy.Publisher('/beer_3', String, queue_size=10)
beer4 = rospy.Publisher('/beer_4', String, queue_size=10)
beer5 = rospy.Publisher('/beer_5', String, queue_size=10)
beer6 = rospy.Publisher('/beer_6', String, queue_size=10)

shelf1 = rospy.Publisher('shelf_1', String, queue_size=10)
shelf2 = rospy.Publisher('shelf_2', String, queue_size=10)
shelf3 = rospy.Publisher('shelf_3', String, queue_size=10)
shelf4 = rospy.Publisher('shelf_4', String, queue_size=10)
shelf5 = rospy.Publisher('shelf_5', String, queue_size=10)
shelf6 = rospy.Publisher('shelf_6', String, queue_size=10)
shelf7 = rospy.Publisher('shelf_7', String, queue_size=10)
shelf8 = rospy.Publisher('shelf_8', String, queue_size=10)
shelf9 = rospy.Publisher('shelf_9', String, queue_size=10)
shelf10 = rospy.Publisher('shelf_10', String, queue_size=10)
shelf11 = rospy.Publisher('shelf_11', String, queue_size=10)
shelf12 = rospy.Publisher('shelf_12', String, queue_size=10)
shelf13 = rospy.Publisher('shelf_13', String, queue_size=10)
shelf14 = rospy.Publisher('shelf_14', String, queue_size=10)
shelf15 = rospy.Publisher('shelf_15', String, queue_size=10)
shelf16 = rospy.Publisher('shelf_16', String, queue_size=10)


sub_cam1 = rospy.Subscriber("/orbotox/camera1/image_raw", Image, camera1_callback)
sub_cam2 = rospy.Subscriber("/orbotox/camera2/image_raw", Image, camera2_callback)
sub_cam3 = rospy.Subscriber("/orbotox/camera3/image_raw", Image, camera3_callback)
sub_move = rospy.Subscriber("/cmd_vel", Twist, callback, queue_size=1)
rospy.spin()