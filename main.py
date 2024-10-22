import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image_path = 'images/the_berry_farms_sunflower_field.jpeg'
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
cv.imshow("berry", image_path)