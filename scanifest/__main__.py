import skewness as sk
import cv2 as cv 
i            = cv.imread('./data/p16.jpg', cv.CV_8UC1)
img_not      = cv.bitwise_not(i)
angle, lines = sk.detect_skew_angle(img_not)
sk.draw_lines(img_not, lines)
sk.correct_skew(i, angle) 
