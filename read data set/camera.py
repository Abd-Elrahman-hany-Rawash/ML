import encodings
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime

path = 'Train'
encodings = []
names = []

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'{name}, Time: {time}, date: {date}')
            f.writelines("\n")

def findEcoding(img):

    image = face_recognition.load_image_file(img)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (0,0), None, 0.3,0.3)
#    face_locations = face_recognition.face_locations(image)
    
    face_detection = face_recognition.face_encodings(image, model="small")
    for faceing in face_detection:
         faceing = np.array(faceing).ravel()
         encodings.append(faceing)
         names.append(os.path.basename(img).split('.')[0])

def detectFace(path , encodings , names):
    name = []
    unknown_picture = face_recognition.load_image_file(path)
    unknown_picture = cv2.cvtColor(unknown_picture, cv2.COLOR_BGR2RGB)
    unknown_picture = cv2.resize(unknown_picture, (0,0), None, 0.3,0.3)
    #face_locations = face_recognition.face_locations(unknown_picture)            

    unknown_face_encoding = face_recognition.face_encodings(unknown_picture, model="small")
    unknown_faces = []

    for faces in unknown_face_encoding:
        unknown_faces.append(np.array(faces).ravel())

    for unknown_face in unknown_faces: 
        results = face_recognition.compare_faces(encodings, unknown_face, tolerance=0.5)
        best_face_destination = face_recognition.face_distance(encodings, unknown_face)
        index_of_best_face_destination = np.argmin(best_face_destination) # index of best distance (encodings)
        if results[index_of_best_face_destination]:
            name.append(names[index_of_best_face_destination])
            markAttendance(names[index_of_best_face_destination])
        else:
            name.append("UNKOWN PERSON!")
    return name   

def camModel(encodings , names):
    cap  = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0,0), None, 0.3,0.3)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        unknown_faces_location = face_recognition.face_locations(imgS)
        unknown_faces_encodeings = face_recognition.face_encodings(imgS, unknown_faces_location)

        for encode_face, faceloc in zip(unknown_faces_encodeings,unknown_faces_location):
            matches = face_recognition.compare_faces(encodings, encode_face , tolerance=0.5)
            faceDist = face_recognition.face_distance(encodings, encode_face)
            matchIndex = np.argmin(faceDist)
        
            if matches[matchIndex]:
                name = names[matchIndex]
                print(name)
                y1,x2,y2,x1 = faceloc
                # since we scaled down by 4 times
                y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name)
        cv2.imshow('webcam', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    j = 0 
    print(datetime.now().strftime('%I:%M:%S:%p')) 
    for img in os.listdir(path):
        j+=1
        print("Encoding.... Person num: ",j)
        findEcoding(f'{path}/{img}') 

    print("The testing has been started... ")
    j = 1
    print(datetime.now().strftime('%I:%M:%S:%p')) 
    for student in detectFace("Test\Mohammed & Sayed.jpeg",encodings,names):
        print("Student ", j , " name: ", student)
        j+=1
    print("Finish...")

    camModel(encodings , names)
