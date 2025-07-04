import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20  # Offset for cropping the image
imgSize = 300  # Size of the cropped image

folder = "Data/Z"
counter = 0

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

while True:
  success, img = cap.read()
  imgOutput = img.copy()
  hands, img = detector.findHands(img)
  if hands:
    hand = hands[0]
    x, y, w, h = hand['bbox']

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255  # Create a white image
    imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]

    imgCropShape = imgCrop.shape

    aspectRatio = h / w

    if aspectRatio > 1:  # Height is greater than width
      k = imgSize / h
      wCal = math.ceil(k * w)
      imgResize = cv2.resize(imgCrop, (wCal, imgSize))
      imgResizeShape = imgResize.shape
      wGap = math.ceil((imgSize - wCal)/2)
      imgWhite[:, wGap:wGap + wCal] = imgResize
      prediction, index = classifier.getPrediction(imgWhite, draw=False)
      print(prediction, index)


    else:  # Width is greater than height
      k = imgSize / w
      hCal = math.ceil(k * h)
      imgResize = cv2.resize(imgCrop, (imgSize, hCal))
      imgResizeShape = imgResize.shape
      hGap = math.ceil((imgSize - hCal)/2)
      imgWhite[hGap:hGap + hCal, :] = imgResize
      prediction, index = classifier.getPrediction(imgWhite, draw=False)

    cv2.putText(imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255),2)
    
    cv2.imshow("ImageCrop", imgCrop)
    cv2.imshow("ImageWhite", imgWhite)

  cv2.imshow("Image", imgOutput)
  cv2.waitKey(1)