import cv2
import numpy as np
import matplotlib.pyplot as plt
 
print("Start programm...")
img = cv2.imread("C:/Users/Jean/Pictures/Capture.PNG")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fast = cv2.FastFeatureDetector_create()


cv2.VideoCapture()

while True:
    kp = fast.detect(img,None)
    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(True)
    fast.setThreshold(2)

    cv2.drawKeypoints(img, kp, img, color=(255,255,255))

    print ("Threshold: ", fast.getThreshold())
    #print ("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
    #print ("neighborhood: ", fast.getInt('type'))
    print ("Total Keypoints with nonmaxSuppression: ", len(kp))

    cv2.imwrite('fast_true_2.png',img)

    cv2.waitKey()