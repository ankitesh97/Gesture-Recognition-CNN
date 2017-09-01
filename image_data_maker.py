
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
count = 1
while count != 1001:
    ret, frame = cap.read()
    cv2.rectangle(frame, (300,300), (100,100), (0,255,0),0)
    crop_img = frame[100:300, 100:300]
    value = (33, 33)
    hsv = cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv,value,0)
    blurred = cv2.GaussianBlur(grey, value, 0)
    lower_green = np.array([80,50,30])
    upper_green = np.array([255,255,255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    gaussian = cv2.GaussianBlur(mask, (11,11), 0)
    image, contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 255, 255), 0)
    erosion = cv2.erode(mask, None, iterations = 1)
    dilated = cv2.dilate(erosion,None,iterations = 1)
    median = cv2.medianBlur(dilated, 7)
    cv2.imshow('cropped', frame)
    cv2.imshow('mask', median)

    # #
    # write_img = cv2.resize(median, (50,50))
    # cv2.imwrite('images_data/peace/'+str(count)+'.jpg',write_img)
    # print count
    # count += 1
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
