# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 07:52:26 2020

@author: Dell
"""
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

subjects = ["", "Ariana Grande", "Selena Gomez","Taylor Swift"]

# function to detect face using OpenCV
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # if no faces are detected then return original img
    if len(faces) == 0:
        return None, None

    # under the assumption that there will be only one face,
    # extract the face area
    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]

def prepare_training_data(data_folder_path):
    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("a"):
            continue;

        label = int(dir_name.replace("a", ""))

        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            # build image path
            # sample image path = training-data/a1/1.jpg
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)

            # detect face
            face, rect = detect_face(image)

            # ------STEP-4--------
            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("train_a")
print("Data prepared")

# print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

print("faces",faces)
print("labels",labels)
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces,np.array(labels))
def draw_rectangle(img,rect):
    (x,y,w,h)=rect
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
    
def draw_text(img,text,x,y):
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)

def predict(test_img):
    img=test_img.copy()
    face,rect=detect_face(img)
    label=face_recognizer.predict(face)
    label_text=subjects[label[0]]
    draw_rectangle(img,rect)
    draw_text(img, label_text, rect[0], rect[1] - 5)
    return img

test_img1=cv2.imread("test_a/test1.jpg")
test_img2=cv2.imread("test_a/test2.jpg")    
test_img3=cv2.imread("test_a/test3.jpg")  
predicted_img1=predict(test_img1)
predicted_img2=predict(test_img2)
predicted_img3=predict(test_img3)
print("prediction complete")
cv2.imshow("Predicted image 1",cv2.resize(predicted_img1,(400,600)))
cv2.imshow("Predicted image 2",cv2.resize(predicted_img2,(400,600)))
cv2.imshow("Predicted image 3",cv2.resize(predicted_img3,(400,600)))

# cv2.imshow(subjects[1],predicted_img1)
# cv2.imshow(subjects[2],predicted_img2)
# cv2.imshow(subjects[3],predicted_img3)
# titles=[subjects[1],subjects[2],subjects[2]]
# images=[predicted_img1,predicted_img2,predicted_img3]
# for i in range(2):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
#     plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
    

