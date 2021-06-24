import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

img_rgb = cv.imread("C:/Users/kbang/what-do-these-images-mean/data/white-background/white-with-black-icons-4.png")

img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

drawing = svg2rlg('C:/Users/kbang/what-do-these-images-mean/data/symbols-svg/iron-medium.svg')
renderPM.drawToFile(drawing, 'iron-medium.png', fmt='PNG')
template = cv.imread("C:/Users/kbang/what-do-these-images-mean/iron-medium.png",0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.4
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv.imwrite('res.png',img_rgb)
