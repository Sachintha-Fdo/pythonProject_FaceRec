import cv2
import numpy as np
import face_recognition

#import images
imgbill = face_recognition.load_image_file('image/jeff1.jpeg')
imgbill = cv2.cvtColor(imgbill,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('image/bill2.jpeg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

#face detect
faceLoc = face_recognition.face_locations(imgbill)[0]
encodBill = face_recognition.face_encodings(imgbill)[0]
cv2.rectangle(imgbill,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#image encoding
faceLocTest = face_recognition.face_locations(imgtest)[0]
encodTest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodBill],encodTest)
print(results)

print(faceLoc)

cv2.imshow('Bill Gates',imgbill)
cv2.imshow('Bill Gates Test',imgtest)
cv2.waitKey(0)

#Message box
import tkinter as tk
from tkinter import messagebox
tk.messagebox.showinfo("Result", results)

