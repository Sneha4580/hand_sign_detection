import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20  # Offset for cropping the image
imgSize = 300  # Size of the cropped image

folder = "Data/Z"
counter = 0
while True:
  success, img = cap.read()
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


    else:  # Width is greater than height
      k = imgSize / w
      hCal = math.ceil(k * h)
      imgResize = cv2.resize(imgCrop, (imgSize, hCal))
      imgResizeShape = imgResize.shape
      hGap = math.ceil((imgSize - hCal)/2)
      imgWhite[hGap:hGap + hCal, :] = imgResize

    
    cv2.imshow("ImageCrop", imgCrop)
    cv2.imshow("ImageWhite", imgWhite)

  cv2.imshow("Image", img)
  key = cv2.waitKey(1)
  if key == ord('s'):
    counter += 1
    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
    print(counter)