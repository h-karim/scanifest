import cv2
import numpy as np

def detect_skew_angle(image):
    height, width= image.shape
    lines = cv2.HoughLinesP(i, rho=1, theta = np.pi/180, 
            threshold = 100, minLineLength=width/2, maxLineGap=20)
    angle = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle += np.arctan2( (y2 - y1), (x2 - x1))
    print(angle)
    mean_ang = angle/(len(lines))
    print(mean_ang*180/np.pi)
    return (mean_ang*180/np.pi, lines)

def draw_lines(image, lines):
    for line in lines:
        print(line[0])
        x1,y1,x2,y2 = line[0]
        cv2.line(image,(x1,y1),(x2,y2), (255,0,0))
def correct_skew(image, rho):
    height, width = image.shape[:2]
    center = (width/2, height/2)
    M = cv2.getRotationMatrix2D(center, -rho, 1)
    rotated = cv2.warpAffine(image, M, (height, width))
    cv2.imshow('rotated', rotated)
    cv2.waitKey(0)
    pass

i = cv2.imread('./data/m20.jpg', cv2.CV_8UC1)
img_not = cv2.bitwise_not(i)
angle, lines = detect_skew_angle(img_not)
correct_skew(i, angle) 
