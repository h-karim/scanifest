import cv2
import numpy as np

def detect_skew_angle(image):
    height, width = image.shape
    edges         = cv2.Canny(image, 50, 200)
    cv2.imshow('edges', edges)
    cv2.waitKey(0)
    cv2.imwrite('edges.jpg', edges)
    lines = cv2.HoughLinesP(edges, rho=1, theta = np.pi/180, 
            threshold = 100, minLineLength=width/2, maxLineGap=10)
    angle = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle         += np.arctan2( (y2 - y1), (x2 - x1))
    print(angle)
    mean_ang = angle/(len(lines))
    print(mean_ang*180/np.pi)
    return (mean_ang*180/np.pi, lines)

def draw_lines(image, lines):
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(image,(x1,y1),(x2,y2), (255,0,0))
    cv2.imshow('lines', image)
    cv2.waitKey(0)
    cv2.imwrite('lines.jpg',image)
def correct_skew(image, rho):
    height, width = image.shape[:2]
    center        = (width/2, height/2)
    M             = cv2.getRotationMatrix2D(center, rho, 1)
    cos, sin      = (np.abs(M[0,0]), np.abs(M[1,0]))
    new_width     = int(width*cos + height*sin)
    new_height    = int(height*cos + width*sin)
    M[0,2]       += new_width/2 - width/2
    M[1,2]       += new_height/2 - height/2
    rotated       = cv2.warpAffine(image, M, (new_width, new_height))
    cv2.imshow('rotated', rotated)
    cv2.waitKey(0)
    cv2.imwrite('rotated.jpg',rotated)
    return rotated

i            = cv2.imread('./data/p16.jpg', cv2.CV_8UC1)
img_not      = cv2.bitwise_not(i)
angle, lines = detect_skew_angle(img_not)
draw_lines(img_not, lines)
correct_skew(i, angle) 
