import skewness as sk
import cv2 as cv 
import argparse 
import logging
import os

parser = argparse.ArgumentParser(description = 'Skewness and shear correcting tool')
parser.add_argument('image', help = 'image file path')
parser.add_argument('-o', '--output', help = 'output file name')
parser.add_argument('-d', '--debug', help  = 'enable debug logging', action = 'store_true', default = False)
args   = parser.parse_args()
if args.debug:
    logging.basicConfig(level = logging.DEBUG, format = '[%(levelname)s] %(message)s')
logging.debug(f'arguments passed: {vars(args)}' )

i_path = args.image
if not os.path.isfile(i_path):
    raise OSError('image file does not exist')
img          = cv.imread(i_path, cv.CV_8UC1)
img_not      = cv.bitwise_not(img)
angle, lines = sk.detect_skew_angle(img_not)
sk.draw_lines(img_not, lines)
sk.correct_skew(i, angle) 
