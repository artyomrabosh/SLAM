import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image
import random

def SaveImgFromArray(img):
    # takes an image object and saves it to images folder with random hash as name
    im = Image.fromarray(img.astype("uint8"))
    im.save("images/" + "%x" % random.getrandbits(128) + ".jpeg")


os.chdir(r"C:\Users\moore\PycharmProjects\AutonomousPlane\data")
try:
    if not os.path.exists('data'):
        os.makedirs('data2')
except OSError:
    print('Error: Creating directory of data')

currentFrame = 0

directory = r"C:\Users\moore\PycharmProjects\AutonomousPlane\data"
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        img = cv2.imread(filename, 0)
        orb = cv2.ORB_create()
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        plt.imshow(img2), plt.show()
        SaveImgFromArray(img2)
        continue


currentFrame += 1
cap.release()
cv2.destroyAllWindows()