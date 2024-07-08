import math

import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

#Inialize the video
cap = cv2.VideoCapture('Videos/vid (4).mp4')

# create the color finder obj
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 5, 'smin': 111, 'vmin': 106, 'hmax': 15, 'smax': 255, 'vmax': 245}

# variables
posListX, posListY = [], []
xList = [item for item in range(0, 1300)]
prediction = False

while True:
   # Grab the image
   success, img = cap.read()
   # img = cv2.imread("Ball.png") #temporary to capture color of the ball and later use same hsv values for video.
   img = img[0:900, :] #corp the image

   #find the color of the ball
   imgColor, mask = myColorFinder.update(img, hsvVals)
   imgColor = imgColor[0:900, :]

   #Find location of ball using countours
   imgContours ,contours =cvzone.findContours(img, mask, minArea=200)

   if contours:
      posListX.append(contours[0]['center'][0])
      posListY.append(contours[0]['center'][1])

   if posListX:
      # polynomial regression y = a^2 + bx + c
      # find coff
      a, b, c = np.polyfit(posListX, posListY, 2)

      for i, (posX, posY) in enumerate(zip(posListX,posListY)):
         pos = (posX, posY)
         cv2.circle(imgContours, pos, 10, (0, 255, 0), cv2.FILLED) # to display position as points
         if i == 0:
            cv2.line(imgContours, pos, pos, (0, 255, 0), 8)
         else:
            cv2.line(imgContours, pos, (posListX[i - 1],posListY[i - 1]), (0, 255, 0), 8)

      for x in xList:
         y = int((a*(x**2)) + (b*x) + c)
         cv2.circle(imgContours,(x, y), 1, (255, 0, 255), cv2.FILLED) # to display position as points

      if len(posListX)<10:
         # Prediction
         # x = values 330 to 430
         # y = 595
         A = a
         B = b
         C = c - 590

         x = int((-B -math.sqrt((b**2)-4*A*C))/(2*A))
         prediction = 330 < x < 430
      if prediction:
         cvzone.putTextRect(imgContours, "Basket", (50, 100), scale=7, thickness= 5, colorR=(0, 255, 0), offset= 20)
      else:
         cvzone.putTextRect(imgContours, "No Basket", (50, 100), scale=7, thickness= 5, colorR=(0, 0, 255), offset= 20)


   #Display
   imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7) #resize the output window.
   # cv2.imshow("Image", img)
   cv2.imshow("ImageColor", imgContours)
   cv2.waitKey(100) #to control the speed of the output video.

   # print(posList)