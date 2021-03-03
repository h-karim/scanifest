import sys
import skewness as sk
import cv2 as cv 
import argparse 
import logging
import os

parser = argparse.ArgumentParser(description = 'Skewness and shear correcting tool')
parser.add_argument('image', help = 'image file path')
parser.add_argument('-o', '--output', help = 'output file name -- must have a valid image extension')
parser.add_argument('-d', '--debug', help  = 'enable debug logging', action = 'store_true', default = False)
args   = parser.parse_args()
if args.debug:
    logging.basicConfig(level = logging.DEBUG, format = '%(message)s')
logging.debug(f'arguments passed: {vars(args)}' )

i_path = args.image
destination = args.output
if not os.path.isfile(i_path):
    raise OSError('image file does not exist')
img          = cv.imread(i_path)
gray         = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
bw_inv = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11,2) 
if args.debug:
    cv.imshow('img',bw_inv)
    cv.waitKey(0)
    cv.imwrite(destination +'bw_inv.jpg', bw_inv)
angle, lines = sk.detect_skew_angle(bw_inv)
try:
    sk.draw_lines(bw_inv, lines, destination)
    sk.correct_skew(img, angle, destination) 
except cv.error as e:
    print(e)
    print('You likely did not specify a valid image extension for your output file')
    sys.exit(1)
