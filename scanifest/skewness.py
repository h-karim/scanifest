import cv2 as cv
import numpy as np
import logging 
logger = logging.getLogger(__name__)

def detect_skew_angle(image):
    height, width = image.shape
    edges         = cv.Canny(image, 50, 200)
    cv.imshow('edges', edges)
    cv.waitKey(0)
    cv.imwrite('edges.jpg', edges)
    lines = cv.HoughLinesP(edges, rho=1, theta = np.pi/180, 
            threshold = 100, minLineLength=width/2, maxLineGap=10)
    angle = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle         += np.arctan2( (y2 - y1), (x2 - x1))
    mean_ang = angle/(len(lines))
    logging.debug(f'mean skewness angle: {mean_ang} rad')
    return (mean_ang*180/np.pi, lines)

def draw_lines(image, lines, *args):
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(image,(x1,y1),(x2,y2), (255,0,0))
    cv.imshow('lines', image)
    cv.waitKey(0)
    if args:
        destination = args[0]
        logging.debug(f'writing hough lines to: {destination}')
        destination = args[0]
        cv.imwrite(('hough_'+destination) ,image)

def correct_skew(image, rho, *args):
    height, width = image.shape[:2]
    center        = (width/2, height/2)
    M             = cv.getRotationMatrix2D(center, rho, 1)
    cos, sin      = (np.abs(M[0,0]), np.abs(M[1,0]))
    new_width     = int(width*cos + height*sin)
    new_height    = int(height*cos + width*sin)
    M[0,2]       += new_width/2 - width/2
    M[1,2]       += new_height/2 - height/2
    rotated       = cv.warpAffine(image, M, (new_width, new_height))
    cv.imshow('rotated', rotated)
    cv.waitKey(0)
    if args:
        destination = args[0]
        logging.debug(f'writing rotated image to: {destination}')
        cv.imwrite(('rotated_'+destination),rotated)
    return rotated

