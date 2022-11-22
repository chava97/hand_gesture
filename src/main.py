# https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/ --> Gestos
# https://github.com/ANANTH-SWAMY/NUMBER-DETECTION-WITH-MEDIAPIPE --> Numeros

from tensorflow.keras.models import load_model
from datetime import datetime
from turtle import width
from threading import Thread
from djitellopy import Tello

import os
import math
import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

################## ROS #######################
from pickletools import uint8
import rospy
import sys
from getkey import getkey, keys
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8
from std_msgs.msg import UInt8
from std_msgs.msg import Empty

class manejoDron:
    
    def __init__(self):
        # UNIVERSAL
        self.menu = True
        self.frame = None
        self.framerbg = None
        self.standby = False
        self.dronDone = False
        self.timeElapsed = 0
        self.startTime = 0
        self.modes = {
            "okay": "Controlador Altura", 
            "peace": "Derecha",
            "smile": "Izquierda",
            "thumbs up": "Subir",
            "thumbs down": "Bajar",
            "stop": "Girar Derecha", 
            "rock": "Girar Izquierda"
        }
        
        # MENU GESTOS
        self.gestoFinal = ''
        self.cargarModelo()
        self.pastCN = 0
        self.timeCapGest = 5
        
        # SELECCION NUMEROS
        self.width = 0
        self.height = 0
        self.cnt = 0
        self.timeCapNum = 5
        self.result = ["x","x","x"]
        self.res_ix = 0
        self.pastCnt = -1
        self.contHeight = 0
        
        # ROS
        self.pub_takeoff = rospy.Publisher('/tello/takeoff', Empty, queue_size=10)
        self.pub_land = rospy.Publisher('/tello/land', Empty, queue_size=10)
        self.pub_flip = rospy.Publisher('/tello/flip', UInt8, queue_size=10)
        self.pub_cmd_vel = rospy.Publisher('/tello/cmd_vel', Twist, queue_size=10)
        self.pub_override = rospy.Publisher('keyboard_control/override', Int8, queue_size=10)
        self.vel_msg = Twist()
        self.flip_msg = 0
        self.ovr_msg = 0
        self.speed_value = 0.5

        
    def run(self):
        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.startTime = datetime.now()

        while True:
            ret, self.frame = cap.read()
            if not ret:
                break

            # Flip the frame vertically
            self.frame = cv2.flip(self.frame, 1)
            self.framergb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            
            #self.timeElapsed = (datetime.now()-self.startTime).seconds
            
            if not self.standby:
                if self.menu:
                    self.manejoGestos()
                    self.dronDone = False
                elif self.gestoFinal=="okay":
                    self.contHeight = self.seleccionNumeros()
                    
                elif self.gestoFinal=="thumbs up":
                    self.standby = True
                    self.vel_msg.linear.x = 0.0
                    self.vel_msg.linear.y = 0.0
                    self.vel_msg.linear.z = round(self.speed_value, 2)
                    self.vel_msg.angular.z = 0.0
                    self.pub_cmd_vel.publish(self.vel_msg)
                    print("Subiendo")
                    time.sleep(3)
                    self.dronDone = True
                elif self.gestoFinal=="thumbs down":
                    self.standby = True
                    self.vel_msg.linear.x = 0.0
                    self.vel_msg.linear.y = 0.0
                    self.vel_msg.linear.z = -round(self.speed_value, 2)
                    self.vel_msg.angular.z = 0.0
                    self.pub_cmd_vel.publish(self.vel_msg)
                    print("Bajando")
                    time.sleep(3)
                    self.dronDone = True
                elif self.gestoFinal=="peace":
                    self.standby = True
                    self.vel_msg.linear.x = 0.0
                    self.vel_msg.linear.y = -round(self.speed_value, 2)
                    self.vel_msg.linear.z = 0.0
                    self.vel_msg.angular.z = 0.0
                    self.pub_cmd_vel.publish(self.vel_msg)
                    time.sleep(3)
                    self.dronDone = True
                elif self.gestoFinal=="smile":
                    self.standby = True
                    self.vel_msg.linear.x = 0.0
                    self.vel_msg.linear.y = round(self.speed_value, 2)
                    self.vel_msg.linear.z = 0.0
                    self.vel_msg.angular.z = 0.0
                    self.pub_cmd_vel.publish(self.vel_msg)
                    time.sleep(3)
                    self.dronDone = True
                elif self.gestoFinal=="stop":
                    self.standby = True
                    self.vel_msg.linear.x = 0.0
                    self.vel_msg.linear.y = 0.0
                    self.vel_msg.linear.z = 0.0
                    self.vel_msg.angular.z = -round(self.speed_value, 2)
                    self.pub_cmd_vel.publish(self.vel_msg)
                    time.sleep(3)
                    self.dronDone = True
                elif self.gestoFinal=="rock":
                    self.standby = True
                    self.vel_msg.linear.x = 0.0
                    self.vel_msg.linear.y = 0.0
                    self.vel_msg.linear.z = 0.0
                    self.vel_msg.angular.z = round(self.speed_value, 2)
                    self.pub_cmd_vel.publish(self.vel_msg)
                    time.sleep(3)
                    self.dronDone = True
                else:
                    self.standby = True
            else:
                cv2.putText(self.frame,"Drone moving...",(30,int(self.height/2)-40),cv2.QT_FONT_NORMAL,2.3,(0,0,255),5)
                cv2.putText(self.frame,"Standby",(int(self.width/2)-150,int(self.height/2)+40),cv2.QT_FONT_NORMAL,2.3,(100,0,255),5)
                cv2.putText(self.frame,self.modes[self.gestoFinal],(30,int(self.height/2)+200),cv2.FONT_ITALIC,2,(0,255,0),5)
                if self.dronDone:
                    self.vel_msg.linear.x = 0.0
                    self.vel_msg.linear.y = 0.0
                    self.vel_msg.linear.z = 0.0
                    self.vel_msg.angular.z = round(self.speed_value, 2)
                    self.pub_cmd_vel.publish(self.vel_msg)
                    self.menu = True
                    self.standby = False
                
            # Show the final output
            cv2.imshow("Output", self.frame) 

            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.result = ["x","x","x"]
                self.res_ix = 0
                self.pastCnt = -1
            elif key == ord('m'):
                self.dronDone = True
            
            self.timeElapsed = (datetime.now()-self.startTime).seconds
            
        cap.release()

        cv2.destroyAllWindows()
        
    def manejoGestos(self):
        x, y, _ = self.frame.shape

        # Get hand landmark prediction
        result = self.hands.process(self.framergb)
        className = ''
        
        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                 # Drawing landmarks on frames
                self.mpDraw.draw_landmarks(self.frame, handslms, self.mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = self.model.predict([landmarks])
               
                classID = np.argmax(prediction)
                className = self.classNames[classID]
                
                if(self.pastCN!=className):
                    self.startTime = datetime.now()
                                     
                self.pastCN = className
                if(className != 0 and self.timeElapsed >= self.timeCapGest):
                    self.menu = False
                    self.gestoFinal = className
                    self.hands = self.mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
                    self.startTime = datetime.now()
                 
            cv2.putText(self.frame,str(self.timeCapGest-self.timeElapsed)+'s',(30,75),cv2.FONT_ITALIC,2.5,(0,255,255),5)
        else:
            self.pastCN = -1
            self.startTime = datetime.now()
        # show the prediction on the frame
        if(className not in self.modes):
            self.startTime = datetime.now()
            return
        cv2.putText(self.frame, self.modes[className], (20, 450), cv2.FONT_ITALIC, 2, (0,0,255), 2, cv2.LINE_AA)
        
    def seleccionNumeros(self):
        dispBtmMargin = 10
        displayCoords = (int(self.width/2),int(self.height-dispBtmMargin))
        numWidth = 50
        displayWidth = numWidth*3
        displayHeight = 75
        
        res = self.hands.process(self.framergb)
    
        tipids1=[4,8,12,16,20]
        tipids2=[25,29,33,37,41]
        lmlist=[]
        
        #list of all landmarks of the tips of fingers
        recPt1 = (displayCoords[0]-int(displayWidth/2),displayCoords[1]-displayHeight)
        recPt2 = (displayCoords[0]+int(displayWidth/2),displayCoords[1])
        cv2.rectangle(self.frame,recPt1,recPt2,(220,220,220),cv2.FILLED)
        cv2.rectangle(self.frame,recPt1,recPt2,(0,0,0),5)
 
        if res.multi_hand_landmarks:
            for handlms in res.multi_hand_landmarks:

                for id,lm in enumerate(handlms.landmark):
                    h,w,_= self.frame.shape
                    cx,cy=int(lm.x * w) , int(lm.y * h)
                    lmlist.append([id,cx,cy])
                    if len(lmlist) != 0 and len(lmlist)>=21:
                        self.cnt = self.countHands(lmlist,tipids1,0)
                        if len(lmlist)==42:
                            self.cnt += self.countHands(lmlist,tipids2,21)
                    
                    self.cnt = min(self.cnt,9)
                    
                    #change color of points and lines
                    self.mpDraw.draw_landmarks(self.frame,handlms,self.mpHands.HAND_CONNECTIONS)
            
            if(self.pastCnt!=self.cnt):
                self.startTime = datetime.now()
                
            self.pastCnt = self.cnt
            
            if(self.res_ix<6 and self.timeElapsed>=self.timeCapNum):
                self.result[self.res_ix] = self.cnt
                self.res_ix+=1
                self.startTime = datetime.now()
                if(self.res_ix>=3):
                    self.standby = True#Esto lo dira el dron
                
            cv2.putText(self.frame,str(self.timeCapNum-self.timeElapsed)+'s',(30,75),cv2.FONT_ITALIC,2.5,(0,255,255),5)
            
            cv2.putText(self.frame,str(self.cnt),(30,recPt1[1]+displayHeight-dispBtmMargin),cv2.FONT_ITALIC,2.5,(0,0,255),5)
    
        finalNum = ''.join(str(i) for i in self.result)
        cv2.putText(self.frame,finalNum,(recPt1[0]+7,recPt1[1]+displayHeight-dispBtmMargin),cv2.FONT_ITALIC,2.5,(0,0,0),5)
        return finalNum
        
    def countHands(self,lmlist,tipids,ofst):
        fingercount=0
        fingerlist=[]  
        #thumb and dealing with flipping of hands
        if lmlist[ofst+12][1] > lmlist[ofst+20][1]:
            if lmlist[tipids[0]][1] > lmlist[tipids[0]-1][1]:
                fingerlist.append(1)
            else:
                fingerlist.append(0)
        else:
            if lmlist[tipids[0]][1] < lmlist[tipids[0]-1][1]:
                fingerlist.append(1)
            else:
                fingerlist.append(0)
        
        #others
        for id in range (1,5):
            if lmlist[tipids[id]][2] < lmlist[tipids[id]-2][2]:
                fingerlist.append(1)
            else:
                fingerlist.append(0)
        if len(fingerlist)!=0:
            fingercount=fingerlist.count(1)
        return fingercount
        
    def cargarModelo(self):
        # UNIVERSAL
        # Initialize mediapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils
        
        # MENU GESTOS 
        # Load the gesture recognizer model
        self.model = load_model('mp_hand_gesture')
        # Load class names
        f = open('gesture.names', 'r')
        self.classNames = f.read().split('\n')
        f.close()
        
manejoDron().run()




