import torch
import cv2
import numpy as np

img = cv2.imread('edge_output/real_edges.png')
# img = cv2.imread('H://class12model.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lines = cv2.HoughLines(gray, 0.9, np.pi / 180, 80)
print(len(lines))
for i in range(len(lines)):
    for rho, theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('dd', img)
cv2.waitKey()
