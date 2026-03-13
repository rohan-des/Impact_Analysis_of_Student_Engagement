from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.models import User, auth

import os
import pandas as pd
import datetime
import time
import cv2
import numpy as np
import csv
from PIL import Image, ImageTk
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import os
#from twilio.rest import Client
##account_sid = "AC726e5660f9f3de22e3e9be8b2fc90612"
##auth_token = "d6b17f81f67b9dac73a63ed21d2637c9"
##client = Client(account_sid, auth_token)

# Create your views here.
def home(request):
   return render(request, "home.html")

def create_datsets(request):
   if request.method == 'POST':
      Id = request.POST['Id']
      Name = request.POST['Name']
      Phone = request.POST['Phone']
      Email = request.POST['Email']
      Sem = request.POST['Sem']
      Cource = request.POST['Cource']
      Branch = request.POST['Branch']
      print(Id+' '+Name+' '+Phone+' '+Email+' '+Sem+' '+Cource+' '+Branch)

      cam = cv2.VideoCapture(0)
      harcascadePath = "D:\\PROJECT\\Attendence\\haarcascade_frontalface_default.xml"
      detector=cv2.CascadeClassifier(harcascadePath)
      sampleNum=0
      while True:
         ret, img = cam.read()
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         faces = detector.detectMultiScale(gray, 1.3, 5)
         for (x,y,w,h) in faces:
               cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
               #incrementing sample number 
               sampleNum=sampleNum+1
               #saving the captured face in the dataset folder TrainingImage
               cv2.imwrite("D:\\PROJECT\\Attendence\\TrainingImage\\ "+Name +"."+Id +'.'+ str(sampleNum) + ".png", gray[y:y+h,x:x+w])
               #display the frame
         cv2.imshow('frame',img)
         #wait for 100 miliseconds 
         if cv2.waitKey(1) & 0xFF == ord('q'):
               break
         # break if the sample number is morethan 100
         elif sampleNum>100:
               break
      cam.release()
      cv2.destroyAllWindows() 
      msg = "Images Saved for ID : " + Id +" Name : "+ Name 
      row = [Id, Name, Phone, Email, Sem, Cource, Branch,]

      if not os.path.exists('D:\\PROJECT\\Attendence\\StudentDetails\\StudentDetails.csv'):
         row1 = ['Id', 'Name', 'Phone', 'Email', 'Sem', 'Cource', 'Branch']
         with open('D:\\PROJECT\\Attendence\\StudentDetails\\StudentDetails.csv','w',newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row1)
         csvFile.close()

      with open('D:\\PROJECT\\Attendence\\StudentDetails\\StudentDetails.csv','a', newline='') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
      return render(request, "home.html", {'msg' : msg})
   return render(request, "home.html")

def training(request):
   def getImagesAndLabels(path):
      #get the path of all the files in the folder
      imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
      #print(imagePaths)
      
      #create empth face list
      faces=[]
      #create empty ID list
      Ids=[]
      #now looping through all the image paths and loading the Ids and the images
      for imagePath in imagePaths:
         #loading the image and converting it to gray scale
         pilImage=Image.open(imagePath).convert('L')
         #Now we are converting the PIL image into numpy array
         imageNp=np.array(pilImage,'uint8')
         #getting the Id from the image
         Id=int(os.path.split(imagePath)[-1].split(".")[1])
         # extract the face from the training image sample
         faces.append(imageNp)
         Ids.append(Id)        
      return faces,Ids
##    recognizer=cv2.face.LBPHFaceRecognizer_create()
   recognizer=cv2.face_LBPHFaceRecognizer.create()
   harcascadePath = "D:\\PROJECT\\Attendence\\haarcascade_frontalface_default.xml"
   detector =cv2.CascadeClassifier(harcascadePath)
   faces,Id = getImagesAndLabels("D:\\PROJECT\\Attendence\\TrainingImage")
   recognizer.train(faces, np.array(Id))
   recognizer.save("D:\\PROJECT\\Attendence\\TrainingImageLabel\\Trainner.yml")
   msg = "Datsets trained successfully"
 
   return render(request, "home.html", {'msg' : msg})

def create_datsets_hod(request):
   if request.method == 'POST':
      Id = request.POST['Id']
      Name = request.POST['Name']
      Phone = request.POST['Phone']
      Subject = request.POST['Subject']
      print(Id+' '+Name+' '+Phone+' '+Subject)

      cam = cv2.VideoCapture(0)
      harcascadePath = "D:\\PROJECT\\Attendence\\haarcascade_frontalface_default.xml"
      detector=cv2.CascadeClassifier(harcascadePath)
      sampleNum=0
      while True:
         ret, img = cam.read()
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         faces = detector.detectMultiScale(gray, 1.3, 5)
         for (x,y,w,h) in faces:
               cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
               #incrementing sample number 
               sampleNum=sampleNum+1
               #saving the captured face in the dataset folder TrainingImage
               cv2.imwrite("D:\\PROJECT\\Attendence\\TrainingImage_hod\\ "+Name +"."+Id +'.'+ str(sampleNum) + ".png", gray[y:y+h,x:x+w])
               #display the frame
         cv2.imshow('frame',img)
         #wait for 100 miliseconds 
         if cv2.waitKey(1) & 0xFF == ord('q'):
               break
         # break if the sample number is morethan 100
         elif sampleNum>100:
               break
      cam.release()
      cv2.destroyAllWindows() 
      msg = "Images Saved for ID : " + Id +" Name : "+ Name 
      row = [Id, Name, Phone, Subject]

      if not os.path.exists('D:\\PROJECT\\Attendence\\HODDetails\\HODDetails.csv'):
         row1 = ['Id', 'Name', 'Phone', 'Subject']
         with open('D:\\PROJECT\\Attendence\\HODDetails\\HODDetails.csv','w',newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row1)
         csvFile.close()

      with open('D:\\PROJECT\\Attendence\\HODDetails\\HODDetails.csv','a', newline='') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
      return render(request, "home.html", {'msg' : msg})
   return render(request, "home.html")

def training_hod(request):
   def getImagesAndLabels(path):
      #get the path of all the files in the folder
      imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
      #print(imagePaths)
      
      #create empth face list
      faces=[]
      #create empty ID list
      Ids=[]
      #now looping through all the image paths and loading the Ids and the images
      for imagePath in imagePaths:
         #loading the image and converting it to gray scale
         pilImage=Image.open(imagePath).convert('L')
         #Now we are converting the PIL image into numpy array
         imageNp=np.array(pilImage,'uint8')
         #getting the Id from the image
         Id=int(os.path.split(imagePath)[-1].split(".")[1])
         # extract the face from the training image sample
         faces.append(imageNp)
         Ids.append(Id)        
      return faces,Ids
##    recognizer=cv2.face.LBPHFaceRecognizer_create()
   recognizer=cv2.face_LBPHFaceRecognizer.create()
   harcascadePath = "D:\\PROJECT\\Attendence\\haarcascade_frontalface_default.xml"
   detector =cv2.CascadeClassifier(harcascadePath)
   faces,Id = getImagesAndLabels("D:\\PROJECT\\Attendence\\TrainingImage_hod")
   recognizer.train(faces, np.array(Id))
   recognizer.save("D:\\PROJECT\\Attendence\\TrainingImageLabel_hod\\Trainner.yml")
   msg = "Datsets trained successfully"
 
   return render(request, "home.html", {'msg' : msg})

def attendence(request):
   recognizer = cv2.face.LBPHFaceRecognizer_create() 
   #recognizer = cv2.createLBPHFaceRecognizer()#cv2.face.LBPHFaceRecognizer_create()
   recognizer.read("D:\\PROJECT\\Attendence\\TrainingImageLabel_hod\\Trainner.yml")
   harcascadePath = "D:\\PROJECT\\Attendence\\haarcascade_frontalface_default.xml"
   faceCascade = cv2.CascadeClassifier(harcascadePath)
   df=pd.read_csv("D:\\PROJECT\\Attendence\\HODDetails\\HODDetails.csv")
   cam = cv2.VideoCapture(0)
   font = cv2.FONT_HERSHEY_SIMPLEX        
   col_names1 =  ['Id','Name','Date','Time']
   attendance1 = pd.DataFrame(columns = col_names1)
   count = 0
   Subject = ''
   while True:
      ret, im =cam.read()
      gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
      faces=faceCascade.detectMultiScale(gray, 1.2,5)
      
      for(x,y,w,h) in faces:
         Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
         if(conf < 60):
            count += 1
            print(count)
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            Name=df.loc[df['Id'] == Id]['Name'].values
            Name = Name[0]
            print(Name)
            Subject = df.loc[df['Id'] == Id]['Subject'].values
            Subject=Subject[0]
            attendance1.loc[len(attendance1)] = [Id,Name,date,timeStamp]
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            cv2.putText(im,str(Name),(x, y-10), font, 1,(255,255,255),2)
         else:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            cv2.putText(im,'Unknown',(x, y-10), font, 1,(255,255,255),2)
      
      cv2.imshow('im',im)
      if cv2.waitKey(1) & count >= 60:
         break
   cam.release()
   cv2.destroyAllWindows()

   attendance1=attendance1.drop_duplicates(subset=['Id'],keep='first')
   ts = time.time()      
   date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
   timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
   Hour,Minute,Second=timeStamp.split(":")
   fileName="D:\\PROJECT\\Attendence\\HODAttendence\\"+str(Subject)+"\\Attendence_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
   attendance1.to_csv(fileName, index=False)


   # parameters for loading data and images
   # detection_model_path = 'C:\\Users\\surya naik\\Desktop\\Final_project\\StudentAttendenceSystem\\Attendence\\haarcascade_frontalface_default.xml'
   emotion_model_path = 'D:\\PROJECT\\Attendence\\_mini_XCEPTION.102-0.66.hdf5'
   # hyper-parameters for bounding boxes shape
   # face_detection = cv2.CascadeClassifier(detection_model_path)
   emotion_classifier = load_model(emotion_model_path, compile=False)
   EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
   recognizer = cv2.face.LBPHFaceRecognizer_create() 
   #recognizer = cv2.createLBPHFaceRecognizer()#cv2.face.LBPHFaceRecognizer_create()
   recognizer.read("D:\\PROJECT\\Attendence\\TrainingImageLabel\\Trainner.yml")
   harcascadePath = "D:\\PROJECT\\Attendence\\haarcascade_frontalface_default.xml"
   faceCascade = cv2.CascadeClassifier(harcascadePath) 
   df=pd.read_csv("D:\\PROJECT\\Attendence\\StudentDetails\\StudentDetails.csv")
   canvas = np.zeros((250, 300, 3), dtype="uint8")
   cam = cv2.VideoCapture(0)
   font = cv2.FONT_HERSHEY_SIMPLEX        
   col_names =  ['Id','Name','Date','Time', 'Emotions']
   attendance = pd.DataFrame(columns = col_names)
   now = datetime.datetime.now()
   count = 0
   Emotions = []
   if os.path.exists('data.csv'):
      os.remove('data.csv')
      
   col_names11 =  ['name','emotion']
   emotion11 = pd.DataFrame(columns = col_names11)
   while True:
      ret, im =cam.read()
      gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
      faces=faceCascade.detectMultiScale(gray, 1.2,5)

      i = 0
      for(x,y,w,h) in faces:
         roi = gray[y:y+h, x:x+w]
         roi = cv2.resize(roi, (64, 64))
         roi = roi.astype("float") / 255.0
         roi = img_to_array(roi)
         roi = np.expand_dims(roi, axis=0)

         preds = emotion_classifier.predict(roi)
         emotion_probability = np.max(preds)
         label = EMOTIONS[preds.argmax()]
         Emotions.append(label)

         cv2.rectangle(canvas, (7, (i * 35) + 5),
         (w, (i * 35) + 35), (0, 0, 255), -1)
         cv2.putText(canvas, label, (10, (i * 35) + 23),
         cv2.FONT_HERSHEY_SIMPLEX, 0.45,
         (255, 255, 255), 2)
         i += 1

         
         Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
         if(conf < 60):
               count += 1
               print(count)
               ts = time.time()
               date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
               timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
               Name=df.loc[df['Id'] == Id]['Name'].values
               Name = Name[0]
               print(Name)
               attendance.loc[len(attendance)] = [Id,Name,date,timeStamp, label]
               emotion11.loc[len(emotion11)] = [Name,label]
               cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
               cv2.putText(im,str(Name)+'('+str(label)+' '+str(w)+')',(x, y-10), font, 1,(255,255,255),2)
         else:
               cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
               #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
               cv2.putText(im,'Unknown('+str(label)+' '+str(w)+')',(x, y-10), font, 1,(255,255,255),2)
            
      cv2.imshow('im',im)
      cv2.imshow('graph',canvas)
      if cv2.waitKey(1) & count >= 250:
         break
   cam.release()
   cv2.destroyAllWindows()

   attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
   emotion11=emotion11.drop_duplicates(subset=['name'],keep='first')
   ts = time.time()
   date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
   timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
   Hour,Minute,Second=timeStamp.split(":")
   fileName="D:\\PROJECT\\Attendence\\StudentAttendence\\"+str(Subject)+"\\Attendence_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
   attendance.to_csv(fileName, index=False)
   emotion11.to_csv('data.csv', index=False)
   
   pl = attendance['Id'].values
   sl = df['Id'].values
   
   attendence_info = []
   for i in sl:
      if i not in pl:
         nm = df.loc[df['Id'] == i]['Name'].values
         nm = nm[0]
         print(nm+' is absent')
         attendence_info.append(nm+' is absent '+Subject)
##         client.api.account.messages.create(
##                                 to="+918217770651",
##                                 from_="+16074146556",
##                                 body=nm+' is absent in {} class'.format(Subject))

      else:
         nm1 = df.loc[df['Id'] == i]['Name'].values
         nm1 = nm1[0]
         print(nm1+' is present')
         attendence_info.append(nm1+' is present at class '+Subject)

   Active_count = 0
   Active_count += Emotions.count("happy")
   Active_count += Emotions.count("neutral")
   Active_count += Emotions.count("surprised")

   lazy_count = 0
   lazy_count += Emotions.count("sad")
   lazy_count += Emotions.count("scared")
   lazy_count += Emotions.count("disgust")
   lazy_count += Emotions.count("angry")

   active_status =  Active_count/len(Emotions)*100
##   from twilio.rest import Client
##   account_sid = "AC23ef5f7ac0d684499de6d027e99cd859"
##   auth_token = " 2e0734d41f93b18b03bd8b38cc9c48bf"
##   client = Client(account_sid, auth_token)
##   client.api.account.messages.create(
##         to="+919916517139",
##         from_="+16205269336",
##         body='{}% students active in {} class'.format(active_status, Subject))

   f=open('data.csv','r')
   reader = csv.reader(f)
   for row in reader:
      if "sad" in row or "scared" in row or "disgust" in row or "neutral" in row:
         print(row)
##         client.api.account.messages.create(
##                                       to="+918217770651",
##                                       from_="+16074146556",
##                                       body='{}'.format(row))
##
##



   return render(request, "home.html", { 'List': attendence_info,  'subject': Subject, 'date': date, 'time':timeStamp,'msg':'{}% students active in {} class'.format(active_status, Subject)})


def home(request):
   return render(request, "home.html")

def persondetection(request):
  # Start capturing video 
   vid_cam = cv2.VideoCapture(0)

   # Detect object in video stream using Haarcascade Frontal Face
   face_detector = cv2.CascadeClassifier("D:\\PROJECT\\Attendence\\haarcascade_frontalface_default.xml")
   

   detect = 0
   # Start looping
   while(True):

       # Capture video frame
       _, image_frame = vid_cam.read()

       # Convert frame to grayscale
       gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

       # Detect frames of different sizes, list of faces rectangles
       faces = face_detector.detectMultiScale(gray, 1.3, 5)

       # Loops for each faces
       for (x,y,w,h) in faces:

           # Crop the image frame into rectangle
           cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
           

           cv2.imwrite('frame.png', image_frame)
           detect += 1
           #send photo
           # Display the video frame, with bounded rectangle on the person's face
       cv2.imshow('frame', image_frame)

       # To stop taking video, press 'q' for at least 100ms
       if cv2.waitKey(100) & 0xFF == ord('q'):
           break

       # If image taken reach 100, stop taking video
##       if detect > 0:
##           break

   # Stop video
   vid_cam.release()
   cv2.destroyAllWindows()
   return render(request, "home.html")

   
  
