import cv2 as cv
import numpy as np
import logging 
logger = logging.getLogger(__name__)

def detect_skew_angle(image):
    """Calculate mean skew angle and hough lines of edges
    
    Hough lines are determined after detecting the edges of the image using the Canny algorithm.
    Angle is calculated by taking the mean of all the lines' angles with respect to the horizontal.
    Positional argument:
    image -- the image matrix
    Return: A tuple containing the mean skew angle and the Hough lines;
    """
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

def draw_lines(image, lines, destination = None):
    """Draw lines over an image and save it.

    Draws given lines over given image and optionally saves the image under a target name
    Positional arguments:
    image       -- the image to draw onto
    lines       -- the lines to be drawn
    destination -- (optional) the output file for the new image. Must end with valid image extension. 
                    If not given image won't be saved.
    """
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(image,(x1,y1),(x2,y2), (255,0,0))
    cv.imshow('lines', image)
    cv.waitKey(0)
    if destination is not None:
        logging.debug(f'writing hough lines to: hough_{destination}')
        cv.imwrite(('hough_'+destination) ,image)

def correct_skew(image, rho, destination = None):
    """Rotate image around its center while preserving original dimensions

    Positional arguments:
    image       -- The image matrix to be rotated
    rho         -- Angle of rotation
    destination -- (optional) the output file for the new image. Must end with valid image extension. 
                    If not given image won't be saved.
    return: 
        The rotated image
    """
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
    if destination is not None:
        logging.debug(f'writing rotated image to: {destination}')
        cv.imwrite(destination, rotated)
    return rotated

