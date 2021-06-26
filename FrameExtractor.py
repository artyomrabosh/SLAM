import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)


try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0

while(True):
    ret, frame = cap.read()
    name = './data/frame' + str(currentFrame) + '.jpg'

    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    if ret:
        cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

    currentFrame += 1


cap.release()
cv2.destroyAllWindows()
