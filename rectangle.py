import cv2
import numpy as np

img = np.ones((150, 500), dtype=np.uint8)*255
cv2.line(img, (350,0),(350,150), (0), 3)
cv2.line(img, (350,75),(500,75), (0), 3)

font = cv2.FONT_HERSHEY_SIMPLEX
text1 = "Add"
text2 = "Reset"



cv2.imshow("mat", img)
cv2.waitKey()


