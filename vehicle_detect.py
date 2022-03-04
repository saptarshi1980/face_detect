# -*- coding: utf-8 -*-

import cv2


width = 600
height = 600
print('Project Topic : Vehicle Classification')

print('By SAPTARSHI SANYAL')

cascade_src = 'cars.xml'

video_src = 'video.mp4'

cap = cv2.VideoCapture(video_src)

cv2.namedWindow('cap',cv2.WINDOW_NORMAL)

car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, img = cap.read()

    if (type(img) == type(None)):
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 2)

    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.resizeWindow('cap', width, height)
    cv2.imshow('cap', img)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()