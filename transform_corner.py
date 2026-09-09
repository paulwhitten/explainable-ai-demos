import os
import sys
import math
import skimage
import numpy as np
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from load_mnist_data import load_mnist, write_partial_mnist_data

# this function returns true if x, y is valid in the image
def valid_pixel(img, x, y):
    max_y = img.shape[0]
    max_x = img.shape[1]
    if (x >= 0 and x < max_x and y >= 0 and y < max_y):
        return True
    else:
        return False


# this function sets the neighborhood of the passed pixel in the image a distance of pixels_away
def set_neighborhood(img, x, y, pixels_away, intensity):
    for yi in range(y-pixels_away, y+pixels_away+1):
        for xi in range(x-pixels_away, x+pixels_away+1):
            if (valid_pixel(img, xi, yi)):
                dist = math.sqrt(abs(x-xi) + abs(y-yi))
                pixel_intensity = int(intensity / (dist + 2))
                if (img[yi, xi] < pixel_intensity):
                    img[yi, xi] = pixel_intensity


N, rows, columns, digits, labels = load_mnist(sys.argv[1], sys.argv[2])

corners = []

digit_count = 0
for d in digits:
    #raw = np.reshape(d, (-1, 28))// not working
    #print("Shape:", raw.shape)
    #print("raw:", raw)
    raw = np.zeros([rows,columns], np.uint8)
    pixel = 0
    for y in range(rows):
        for x in range(columns):
            raw[y,x] = d[pixel]
            pixel += 1
    
    # find corners
    coords = corner_peaks(corner_harris(raw), min_distance=5, threshold_rel=0.02)
    corner_image = np.zeros((rows, columns), np.uint8)
    for coord in coords:
        set_neighborhood(corner_image, coord[1], coord[0], 1, 255)
    
    corners.append(corner_image.flatten())
    digit_count += 1
    if digit_count % 1000 == 0:
        print("count:", digit_count)

write_partial_mnist_data(corners, labels, N, "corners-image",  "corners-labels")


                                              