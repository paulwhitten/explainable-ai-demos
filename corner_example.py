import os
import sys
import math
import skimage
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import corner_harris, corner_subpix, corner_peaks

# from example https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_corner.html#sphx-glr-auto-examples-features-detection-plot-corner-py

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
                pixel_intensity = intensity
                if dist > 0:
                        pixel_intensity = int(intensity / (dist + 2))
                if (img[yi, xi] < pixel_intensity):
                    img[yi, xi] = pixel_intensity

# load image
print("opening file:", sys.argv[1])
filename = os.path.join("./", sys.argv[1])
image = skimage.io.imread(filename)

# find corners
coords = corner_peaks(corner_harris(image), min_distance=3, threshold_rel=0.02)
coords_subpix = corner_subpix(image, coords, window_size=7)

print("corner coords:", coords)

fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
ax.plot(coords[:, 1], coords[:, 0], color='red', marker='o',
        linestyle='None', markersize=6)
#ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=10)
ax.axis((0, 30, 30, 0))

new_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
for coord in coords:
     set_neighborhood(new_image, coord[1], coord[0], 2, 255)
skimage.io.imsave("test.png", new_image, check_contrast=False)

plt.show()



