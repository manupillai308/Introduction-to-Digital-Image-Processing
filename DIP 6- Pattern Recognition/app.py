import cv2
import matplotlib.pyplot as plt
import numpy as np
from Preprocess import preprocess
from keras.models import load_model
import json

classes = json.load(open("./classes.json", "r"))

model = load_model("./Model.h5")
cam = cv2.VideoCapture("http://192.168.0.157:8080/video")

while True:

    ret, image = cam.read()

    if ret:
        image = cv2.resize(image, (500, 300))  
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_ERODE, np.ones((3,3)))
        thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray_image, (5,5), 5/6), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 10)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        dr_im = np.copy(image)
        rectangle = []
        images = []
        for contour, hier in zip(contours[1:], hierarchy[0][1:]):
            area = cv2.contourArea(contour)
            if area < 150 or hier[3] > 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            rectangle.append((x,y,w,h))
            im = (thresh[y:y+h, x:x+w] != 255).astype("uint8")
            images.append(im)

        for rect, img in zip(rectangle, images):
            x, y, w, h = rect
            
            pp_im = preprocess(img)
            pp_im = pp_im.reshape(-1, 45, 45, 1)

            pred = model.predict(pp_im)
            cls = classes[str(np.argmax(pred))]

            cv2.putText(image, cls, (x,y-1), 0, 2, (255, 0, 0), 2)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 1)

        cv2.imshow("Output", image)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break