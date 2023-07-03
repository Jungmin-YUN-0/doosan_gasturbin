import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt

img = cv.imread('C:/Users/cho40/doosan/output_ex/output1.png', cv.IMREAD_GRAYSCALE)
hh, ww = img.shape[:2]

# threshold
_,thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY)
rect=cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dilation = cv.dilate(thresh,rect,iterations = 3)
erosion = cv.erode(dilation, rect, iterations=3)

plt.imshow(erosion, cmap='gray')
plt.axis('off')
plt.show()