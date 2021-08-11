import tkinter as tk
from tkinter import Message, Text
import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from pathlib import Path
import faceRecognition as fr
def Facial():
    window = tk.Tk() 
    window.title("Face_Recogniser")
    window.configure(background ='white')
    window.grid_rowconfigure(0, weight = 1)
    window.grid_columnconfigure(0, weight = 1)
    message = tk.Label(
        window, text ="Face-Recognition-System", 
        bg ="green", fg = "white", width = 50, 
        height = 3, font = ('times', 30, 'bold')) 
        
    message.place(x = 200, y = 20)
    
    lbl = tk.Label(window, text = "No.", 
    width = 20, height = 2, fg ="green", 
    bg = "white", font = ('times', 15, ' bold ') ) 
    lbl.place(x = 400, y = 200)
    
    txt = tk.Entry(window, 
    width = 20, bg ="white", 
    fg ="green", font = ('times', 15, ' bold '))
    txt.place(x = 700, y = 215)
    
    lbl2 = tk.Label(window, text ="Name", 
    width = 20, fg ="green", bg ="white", 
    height = 2, font =('times', 15, ' bold ')) 
    lbl2.place(x = 400, y = 300)
    
    txt2 = tk.Entry(window, width = 20, 
    bg ="white", fg ="green", 
    font = ('times', 15, ' bold ')  )
    txt2.place(x = 700, y = 315)
    # The function beow is used for checking 
    # whether the text below is number or not ?   
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass
    
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
    
        return False
    # Take Images is a function used for creating
    # the sample of the images which is used for 
    # training the model. It takes 60 Images of 
    # every new user.  
    def TakeImages():        
        
        # Both ID and Name is used for recognising the Image
        Id =(txt.get()) 
        name =(txt2.get())
        
        # Checking if the ID is numeric and name is Alphabetical
        if(is_number(Id) and name.isalpha()): 
            # Opening the primary camera if you want to access
            # the secondary camera you can mention the number 
            # as 1 inside the parenthesis
        
            cap=cv2.VideoCapture(0)

            count = 0
            for count in range(70):
                ret,test_img=cap.read()
                if not ret :
                    continue
                cv2.imwrite(r"\trainingImages\3\frame%d.jpg" % count, test_img)     # save frame as JPG file
                count += 1
                resized_img = cv2.resize(test_img, (1000, 700))
                cv2.imshow('face detection Tutorial ',resized_img)
                if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
                    break


            cap.release()
            cv2.destroyAllWindows
            res = "Images Saved for ID : " + Id +" Name : "+ name  
            # Creating the entry for the user in a csv file
            row = [Id, name] 
            with open(r'\UserDetails\UserDetails1.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile)
                # Entry of the row in csv file
                writer.writerow(row) 
            csvFile.close()
            message.configure(text = res)
        else:
            if(is_number(Id)):
                res = "Enter Alphabetical Name"
                message.configure(text = res)
            if(name.isalpha()):
                res = "Enter Numeric Id"
                message.configure(text = res)
                
    # Training the images saved in training image folder  
    def TrainImages():
        import cv2
        import os
        import numpy as np
        import faceRecognition as fr

        #This module takes images  stored in diskand performs face recognition
        test_img=cv2.imread('TestImages/kangana.jpg')#test_img path
        faces_detected,gray_img=fr.faceDetection(test_img)
       


        #Comment belows lines when running this program second time.Since it saves training.yml file in directory
        faces,faceID=fr.labels_for_training_data(r'\FaceRecognition-master\trainingImages')
        face_recognizer=fr.train_classifier(faces,faceID)
        face_recognizer.write(r'C:\trainingData.yml')


        #Uncomment below line for subsequent runs
       

        name={0:"Name 1",1:"Name 2", 2:"Name 3", 3:"Name 4 "}#creating dictionary containing names for each label

        for face in faces_detected:
            (x,y,w,h)=face
            roi_gray=gray_img[y:y+h,x:x+h]
            label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
            print("confidence:",confidence)
            print("label:",label)
            fr.draw_rect(test_img,face)
            predicted_name=name[label]
            if(confidence>37):#If confidence more than 37 then don't print predicted face text on screen
                continue
            fr.put_text(test_img,predicted_name,x,y)

        resized_img=cv2.resize(test_img,(1000,1000))
        #cv2.imshow("face dtecetion tutorial",resized_img)
        cv2.waitKey(0)#Waits indefinitely until a key is pressed
        cv2.destroyAllWindows

    def TrackImages():
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read('trainingData.yml')#Load saved training data

        name = {0:"Name 1",1:"Name 2", 2:"Name 3", 3:"Name 4 "}


        cap=cv2.VideoCapture(0)

        while True:
            ret,test_img=cap.read()# captures frame and returns boolean value and captured image
            faces_detected,gray_img=fr.faceDetection(test_img)



            for (x,y,w,h) in faces_detected:
                cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)

            resized_img = cv2.resize(test_img, (1000, 700))
            cv2.imshow('face detection Tutorial ',resized_img)
            cv2.waitKey(10)


            for face in faces_detected:
                (x,y,w,h)=face
                roi_gray=gray_img[y:y+w, x:x+h]
                label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
                print("confidence:",confidence)
                print("label:",label)
                fr.draw_rect(test_img,face)
                predicted_name=name[label]
                if confidence < 39:#If confidence less than 37 then don't print predicted face text on screen
                    fr.put_text(test_img,predicted_name,x,y)


            resized_img = cv2.resize(test_img, (1000, 700))
            cv2.imshow('face recognition tutorial ',resized_img)
            if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
                break


        cap.release()
        cv2.destroyAllWindows
    takeImg = tk.Button(window, text ="Sample", 
    command = TakeImages, fg ="white", bg ="green", 
    width = 20, height = 3, activebackground = "Red", 
    font =('times', 15, ' bold '))
    takeImg.place(x = 200, y = 500)
    trainImg = tk.Button(window, text ="Training", 
    command = TrainImages, fg ="white", bg ="green", 
    width = 20, height = 3, activebackground = "Red", 
    font =('times', 15, ' bold '))
    trainImg.place(x = 500, y = 500)
    trackImg = tk.Button(window, text ="Testing", 
    command = TrackImages, fg ="white", bg ="green", 
    width = 20, height = 3, activebackground = "Red", 
    font =('times', 15, ' bold '))
    trackImg.place(x = 800, y = 500)
    quitWindow = tk.Button(window, text ="Quit", 
    command = window.destroy, fg ="white", bg ="green", 
    width = 20, height = 3, activebackground = "Red", 
    font =('times', 15, ' bold '))
    quitWindow.place(x = 1100, y = 500)
    
    
    window.mainloop()

