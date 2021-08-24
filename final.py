import cv2 
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

import pygame
import math
pygame.mixer.init()

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

cap = cv2.VideoCapture(0)
video3 = cv2.VideoCapture("ay_t.mp4")
video1 = cv2.VideoCapture("dio.mp4")
video2 = cv2.VideoCapture("thw.mp4")
video4 = cv2.VideoCapture("req.mp4")

pillar = pygame.mixer.Sound('pillar.mp3')
ohh = pygame.mixer.Sound('ohh.mp3')
dio = pygame.mixer.Sound('dio.mp3')
za = pygame.mixer.Sound('za.mp3')
req = pygame.mixer.Sound('req.mp3')


run = False
count = 0
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        
        #cam frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        
        
        #ayeee frame
        ret3, frame3 = video3.read()
        
        if ret3:
            pass
        else:
           video3.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        try:
            frame3 = cv2.resize(frame3, (640, 480))

        except:
            pass
        
        
        #dio frame
        ret1, frame1 = video1.read()
        
        
        if ret1:
            pass
        else:
           video1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        try:
            frame1 = cv2.resize(frame1, (640, 480))
        except:
            pass
        
        #za warudo
        ret2, frame2 = video2.read()
        
        
        if ret2:
            pass
        else:
           video2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        try:
            frame2 = cv2.resize(frame2, (640, 480))
        except:
            pass
        
        #req da
        ret4, frame4 = video4.read()
        
        
        if ret4:
            pass
        else:
           video4.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        try:
            frame4 = cv2.resize(frame4, (640, 480))
        except:
            pass
        
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle = calculate_angle(shoulder, elbow, wrist)
            
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            
            angle_left = calculate_angle(shoulder_r, shoulder, elbow)
            angle_right = calculate_angle(shoulder, shoulder_r, elbow_r)
            
            
            
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(angle_r),
                        tuple(np.multiply(elbow_r, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            if angle > 90 and angle < 120 or angle_r > 90 and angle_r < 120 and angle_left > 100 or angle_right > 100:
                #image = overlay_transparent(image, overlay, 0, 0)
                u_green = np.array([255, 255, 255])
                l_green = np.array([255, 255, 255])

                mask = cv2.inRange(frame3, l_green, u_green)
                res = cv2.bitwise_and(frame3, frame3, mask = mask)

                f = frame - res
                if count >= 60:
                    image = np.where(f == 0, frame, f)
                if(run == True):
                    if count <= 60:
                        count+=1
                    pass
                else:
                    
                    pillar.play()
                    run = True
            elif angle > 40 and angle < 90 or angle_r > 40 and angle_r < 90 and angle_left < 100 and angle_right < 100:
                #image = overlay_transparent(image, overlay, 0, 0)
                u_green = np.array([255, 255, 255])
                l_green = np.array([255, 255, 255])

                mask = cv2.inRange(frame4, l_green, u_green)
                res = cv2.bitwise_and(frame4, frame4, mask = mask)

                f = frame4 - res
                if count >= 90:
                    image = np.where(f == 0, frame, f)
                if(run == True):
                    if count <= 90:
                        count+=1
                    pass
                else:
                    
                    req.play()
                    run = True
                    
                    
                    
            elif angle > 20 and angle < 40 or angle_r > 20 and angle_r < 40 and angle_left > 100 and angle_right > 100:
                
                u_green = np.array([255, 255, 255])
                l_green = np.array([255, 255, 255])

                mask = cv2.inRange(frame1, l_green, u_green)
                res = cv2.bitwise_and(frame1, frame1, mask = mask)

                f = frame1 - res
                image = np.where(f == 0, frame, f)
                if(run == True):
                    pass
                else:
                    
                    dio.play()
                    run = True
                
            else:
                pygame.mixer.stop()
                run = False
                count = 0
                
            
                
        except:
            pygame.mixer.stop()
            run = False
            pass
        finally:
            pygame.mixer.stop()
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Mediapipe |Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            pygame.mixer.music.stop()
            break
cap.release()
cv2.destroyAllWindows()