#!/usr/bin/env python

import argparse
import struct # https://docs.python.org/3/library/struct.html#struct.unpack
import math
import numpy as np
from scipy import fftpack
import skimage
from skimage import io
from skimage.draw import line
from skimage.transform import probabilistic_hough_line, hough_circle, hough_circle_peaks
from skimage.morphology import medial_axis, skeletonize
from load_mnist_data import load_mnist, write_partial_mnist_data, get_num_classes
from skimage.transform import hough_ellipse
from skimage.feature import corner_harris, corner_peaks
from skimage.draw import ellipse_perimeter, circle_perimeter
from skimage.morphology import convex_hull_image
from datetime import datetime
from multiprocessing import shared_memory, Pool, Value, cpu_count
from enum import IntEnum
from sklearn.decomposition import PCA
import pickle
import os
from load_emnist_resnet_data import emnist_classes, emnist_cap_letter_classes, emnist_lower_letter_classes, emnist_digit_classes

num_processed = None
pca_model = None
pca_model_path = None

# Not using lock as we are only touching distinct
# regions of shared memory in each worker.

def convert_array_to_image(pixels, rows, columns):
    img = np.zeros((rows, columns), np.uint8)
    pixel_count = 0
    for y in range(columns):
        for x in range(rows):
            img[y, x] = pixels[pixel_count]
            pixel_count += 1
    return img

# returns an array that is normalized
def convert_to_normalized(pixels, max_val):
    norm = []
    for pixel in pixels:
        norm.append(pixel / max_val)
    return norm

def label_to_norm_array(label, item_count):
    labels = np.zeros(10)
    labels[label] = 1.0
    return labels

def get_neighbors(loc, img):
    rows = img.shape[0]
    columns = img.shape[1]
    neighbors = []
    if (loc[1] - 1 >= 0): # check north
        neighbors.append((loc[0], loc[1] - 1))
    if (loc[1] + 1 < rows): # check south
        neighbors.append((loc[0], loc[1] + 1))
    if (loc[0] + 1 < columns): # check east
        neighbors.append((loc[0] + 1, loc[1]))
    if (loc[0] - 1 >= 0): # check west
        neighbors.append((loc[0] - 1, loc[1]))
    return neighbors

def fill_set_contains_border(img, fill_set):
    rows = img.shape[0]
    columns = img.shape[1]
    if ((0, 0) in fill_set or (0, rows - 1) in fill_set or (columns - 1, 0) in fill_set or (columns - 1, rows - 1) in fill_set) :
        return True
    else:
        for loc in fill_set:
            if (loc[0] == 0 or loc[0] == columns - 1 or loc[1] == 0 or loc[1] == rows - 1):
                return True
    return False

def flood_fill_loops(img, threshold):
    visited = dict()
    loops = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if (not (x, y) in visited.keys()):
                if (threshold > img[y, x]):
                    to_fill = [(x, y)]
                    filled = []
                    while len(to_fill) > 0:
                        #print("len(to_fill):", len(to_fill))
                        loc = to_fill.pop(0)
                        #print("len(to_fill):", len(to_fill))
                        if (img[loc[1], loc[0]] < threshold):
                            filled.append(loc)
                            visited[loc] = True
                            neighbors = get_neighbors(loc, img)
                            for neighbor in neighbors:
                                if (not neighbor in visited.keys() and not neighbor in to_fill):
                                    #print("adding", neighbor)
                                    to_fill.append(neighbor)
                        else:
                            visited[loc] = True
                    if (not fill_set_contains_border(img, filled)):
                        loops.append(filled)
                else:
                    visited[(x, y)] = True
    fill_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for loop_set in loops:
        for pixel in loop_set:
            fill_img[pixel[1], pixel[0]] = 255
    return fill_img

# this function returns true if x, y is valid in the image
def valid_pixel(img, x, y):
    max_y = img.shape[0]
    max_x = img.shape[1]
    if (x >= 0 and x < max_x and y >= 0 and y < max_y):
        return True
    else:
        return False

# this function counts neighbors >= the passed threshold to x,y
def get_neighbor_count(img, x, y, threshold):
    count = 0
    skip = 0
    #print("getting neighbors for", x, y)
    for yi in range(y-1, y+2):
        for xi in range(x-1, x+2):
            #print(xi, yi)
            if (x == xi and y == yi):
                skip += 1
            elif (valid_pixel(img, xi, yi) and img[yi, xi] >= threshold):
                count += 1
    #print("count", count)
    return count

# this function sets the neighborhood of the passed pixel in the image a distance of pixels_away
def set_neighborhood(img, x, y, pixels_away, intensity):
    for yi in range(y-pixels_away, y+pixels_away+1):
        for xi in range(x-pixels_away, x+pixels_away+1):
            if (valid_pixel(img, xi, yi)):
                dist = math.sqrt(abs(x-xi) + abs(y-yi))
                pixel_intensity = int(intensity / (dist + 2))
                if (img[yi, xi] < pixel_intensity):
                    img[yi, xi] = pixel_intensity


# this function gets the endpoints, pixels with only 1 neighbor
# output:
#  the number of endpoints
#  an image with only endpoints set
#  the modified input image with endpoins set to 100
def get_endpoints(img, threshold):
    new_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    mod_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    endpoint_count = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            mod_image[y, x] = img[y,x]
            if (img[y, x] >= threshold and get_neighbor_count(img, x, y, threshold) == 1):
                endpoint_count += 1
                set_neighborhood(mod_image, x, y, 1, 100)
                set_neighborhood(new_image, x, y, 1, 255)
    return (endpoint_count, new_image, mod_image)

# this function tells if x2, y2 is in the neighborhood of x1, y1
def in_neighborhood(x1, y1, x2, y2):
    if (x2 >= x1-1 and x2 <= x1+1 and y2 >= y1-1 and y2 <= y1+1):
        return True
    else:
        return False

# this function identifies crossings
def get_crossings(img, threshold):
    new_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    mod_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    crossings = [] # array to save potential crossings
    # get pixels with more than 2 neighbors
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            mod_image[y, x] = img[y, x]
            if (img[y, x] >= threshold and get_neighbor_count(img, x, y, threshold) > 2):
                crossings.append((x, y))
    # loop through the crossings finding potential neighbors
    # we will remove the neighbors of the pixel with the max neighbors each time through the while loop
    while True:
        max_index = -1          #index of the max
        max_neighbors = 0       #number of neighbors of the max
        neighbors_of_max = []   #the neighbors to remove
        for i in range(len(crossings)):
            neighbor_count = 0
            neighbors = []
            for j in range(len(crossings)):
                if (i != j):
                    if (in_neighborhood(crossings[i][0], crossings[i][1], crossings[j][0], crossings[j][1])):
                        neighbor_count += 1
                        neighbors.append(crossings[j])
            if (neighbor_count > max_neighbors):
                max_neighbors = neighbor_count
                max_index = i
                neighbors_of_max = neighbors
        if (max_index != -1):
            # remove the neighbors of the max
            for rem in neighbors_of_max:
                crossings.remove(rem)
        else:
            break #nothing to remove so bail
    for crossing in crossings:
        set_neighborhood(mod_image, crossing[0], crossing[1], 1, 175) # set the crossing pixels to a lower intensity
        set_neighborhood(new_image, crossing[0], crossing[1], 1, 255)
    return (len(crossings), new_image, mod_image)

def get_lines(image):
    lines = probabilistic_hough_line(image, threshold=0, line_length=4, line_gap=0)
    new_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for l in lines:
        rr, cc = line(l[0][0], l[0][1], l[1][0], l[1][1])
        new_image[cc,rr] = 255
    return new_image

# gets the difference between images
# returns a tuple containing (d, c)
#  where d is the number of pixels that differ
#  and c is the count of pixels in the original image
# TODO: try an or rather than xor
#    also look at scaling to minimize differences
def image_xor(original, new_image):
    if (original.shape[0] != new_image.shape[0]):
        return "images not the same shape"
    if (original.shape[1] != new_image.shape[1]):
        return "images not the same shape"
    acum = 0 # difference accumulator
    original_count = 0
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            original_pixel = 0
            new_pixel = 0
            if (original[y, x] != 0):
                original_count += 1
                original_pixel = 1
            if (new_image[y, x] != 0):
                new_pixel = 1
            if (original_pixel == 1 and original_pixel != new_pixel) :
                acum += 1
    return (acum, original_count)

# TODO parameterize this better
# calculates the image difference in a neighborhood around a pixel
# pixels in the new image close to the original, according to x coordinate
# will give a fractional distance while pixels outside of the neighborhood
# cause a full difference 
def image_diff_neighborhood(original, new_image):
    if (original.shape[0] != new_image.shape[0]):
        return "images not the same shape"
    if (original.shape[1] != new_image.shape[1]):
        return "images not the same shape"
    acum = 0.0 # difference accumulator
    original_count = 0
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            original_pixel = 0
            new_pixel = 0
            neighbor_pixel = 0
            ext_neighbor_pixel = 0
            sec_ext_neighbor_pixel = 0
            if (original[y, x] != 0):
                original_count += 1
                original_pixel = 1
            if (original_pixel == 1):
                if (new_image[y, x] != 0):
                    new_pixel = 1
                elif (y-1 > 0 and new_image[y-1, x] != 0 or y+1 < original.shape[0] and new_image[y+1, x] != 0):
                    neighbor_pixel = 1
                elif (y-2 > 0 and new_image[y-2, x] != 0 or y+2 < original.shape[0] and new_image[y+2, x] != 0):
                    ext_neighbor_pixel = 1
                elif (y-3 > 0 and new_image[y-3, x] != 0 or y+3 < original.shape[0] and new_image[y+3, x] != 0):
                    sec_ext_neighbor_pixel = 1
                if (original_pixel != new_pixel) :
                    # TODO try a non linear increase
                    if (original_pixel == neighbor_pixel):
                        acum += 0.25 # only one pixel off causes 1/4
                    elif (original_pixel == ext_neighbor_pixel):
                        acum += 0.5 # only one pixel off causes 1/2
                    elif (original_pixel == sec_ext_neighbor_pixel):
                        acum += 0.75 # only one pixel off causes 3/4
                    else:
                        acum += 1.0
    return (acum, original_count)

def flood_fill_overlap(fill1, fill2):
    for y in range(fill1.shape[0]):
        for x in range(fill1.shape[1]):
            if (fill1[y, x] == 255 and fill2[y, x] == 255):
                return True
    return False

# gets an ellipse or circles
def get_ellipse(image):
    new_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    # Optimized parameters from tuning: better accuracy and threshold for MNIST digits
    result = hough_ellipse(image, accuracy=2, threshold=3, min_size=5, max_size=20)

    # Estimated parameters for the ellipse
    if (len(result) > 0):
        result.sort(order='accumulator')
        best = list(result[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]

        # only use wih ellipse dimensions > 3
        if (a >= 3 and b >= 3):
            orientation = best[5]

            # Draw the ellipse on the original image
            cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
            # Draw the edge (white) and the resulting ellipse (red)
            # some of the pixels for the ellipse could be out of range
            for i in range(len(cy)):
                if (cy[i] > 0 and cy[i] < 28 and cx[i] > 0 and cx[i] < 28):
                    new_image[cy[i], cx[i]] = 255
    
    return new_image

# gets multiple non overlapping circles
def get_circles(image):
    circle0 = np.zeros([28,28], np.uint8)
    circle1 = np.zeros([28,28], np.uint8)
    circle2 = np.zeros([28,28], np.uint8)
    # get circles
    hough_radii = np.arange(3, 13, 1)
    hough_res = hough_circle(image, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                        total_num_peaks=3)

    circ0y, circ0x = circle_perimeter(cy[0], cx[0], radii[0],
                                shape=circle0.shape)
    circle0[circ0y, circ0x] = 255

    circ1y, circ1x = circle_perimeter(cy[1], cx[1], radii[1],
                                shape=circle0.shape)
    circle1[circ1y, circ1x] = 255

    circ2y, circ2x = circle_perimeter(cy[2], cx[2], radii[2],
                                shape=circle0.shape)
    circle2[circ2y, circ2x] = 255

    acum_threshold = 0.360
    fin_im = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    if (accums[0] > acum_threshold):
        fill0 = flood_fill_loops(circle0, 100)
        fill1 = flood_fill_loops(circle1, 100)
        fill2 = flood_fill_loops(circle2, 100)
        if (not flood_fill_overlap(fill0, fill1) and
            accums[0] > acum_threshold and accums[1] > acum_threshold):
            fin_im[circ0y, circ0x] = 255
            fin_im[circ1y, circ1x] = 255
        elif (not flood_fill_overlap(fill0, fill2) and
            accums[0] > acum_threshold and accums[2] > acum_threshold):
            fin_im[circ0y, circ0x] = 255
            fin_im[circ2y, circ2x] = 255
        elif (not flood_fill_overlap(fill1, fill2) and
            accums[1] > acum_threshold and accums[2] > acum_threshold):
            fin_im[circ1y, circ1x] = 255
            fin_im[circ2y, circ2x] = 255
        elif (accums[0] > acum_threshold):
            fin_im[circ0y, circ0x] = 255
    return fin_im

def get_small_circle(image):
    circle0 = np.zeros([28,28], np.uint8)
    # detect a  circle
    hough_radii = np.arange(3, 7, 1)
    hough_res = hough_circle(image, hough_radii)

    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                        total_num_peaks=1)

    circ0y, circ0x = circle_perimeter(cy[0], cx[0], radii[0],
                                shape=circle0.shape)
    # set threshold for the circle
    acum_threshold = 0.400
    fin_im = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    if (accums[0] > acum_threshold):
        fin_im[circ0y, circ0x] = 255
    return fin_im

def get_large_circle(image):
    circle0 = np.zeros([28,28], np.uint8)
    # detect a  circle
    hough_radii = np.arange(8, 13, 1)
    hough_res = hough_circle(image, hough_radii)

    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                        total_num_peaks=1)

    circ0y, circ0x = circle_perimeter(cy[0], cx[0], radii[0],
                                shape=circle0.shape)
    # set threshold for the circle
    acum_threshold = 0.400
    fin_im = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    if (accums[0] > acum_threshold):
        fin_im[circ0y, circ0x] = 255
    return fin_im

# get one circle
def get_circle(image):
    fin_im = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    circle0 = np.zeros([28,28], np.uint8)
    # Optimized parameters from tuning: smaller radius range and lower threshold for MNIST digits
    hough_radii = np.arange(4, 12, 1)
    hough_res = hough_circle(image, hough_radii)

    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                        total_num_peaks=1)
    
    if len(accums) > 0:
        circ0y, circ0x = circle_perimeter(cy[0], cx[0], radii[0],
                                    shape=circle0.shape)
        # Optimized threshold from tuning (was 0.450, now 0.35)
        acum_threshold = 0.35
        if (accums[0] > acum_threshold):
            fin_im[circ0y, circ0x] = 255
    return fin_im

def get_ellipse_or_circle(image):
    new_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    # Use optimized ellipse parameters from tuning
    result = hough_ellipse(image, accuracy=2, threshold=3, min_size=5, max_size=20)
    if len(result) > 0:
        result.sort(order='accumulator')
        found = False

        # Estimated parameters for the ellipse
        if (len(result) > 0):
            best = list(result[-1])
            yc, xc, a, b = [int(round(x)) for x in best[1:5]]

            # only use wih ellipse dimensions > 3
            if (a >= 3 and b >= 3):
                orientation = best[5]

                # Draw the ellipse on the original image
                cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
                # Draw the edge (white) and the resulting ellipse (red)
                # some of the pixels for the ellipse could be out of range
                for i in range(len(cy)):
                    if (cy[i] > 0 and cy[i] < 28 and cx[i] > 0 and cx[i] < 28):
                        new_image[cy[i], cx[i]] = 255
                found = True
        
        if not found:
            new_image = get_circles(image)

    return new_image

def get_fft(image):
    fft_img = fftpack.fft2(image)
    abs_img = np.abs(fft_img)
    abs_img *= 255 / abs_img.max()
    return abs_img.astype(np.uint8)

def get_convex_hull(image):
    chull = convex_hull_image(image)
    new_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if chull[y,x]:
                    new_image[y,x] = 255
    return new_image

def create_and_save_pca_model(all_images, output_folder, n_components=80):
    """
    Creates PCA model from all images and saves it for reuse.
    """
    global pca_model, pca_model_path
    
    if pca_model is not None:
        return pca_model
    
    print(f"Creating PCA model with {n_components} components from {len(all_images)} images...")
    
    # Flatten all images for PCA training
    flattened_images = np.array([img.flatten().astype(np.float32) / 255.0 for img in all_images])
    
    # Create and fit PCA model
    pca_model = PCA(n_components=n_components)
    pca_model.fit(flattened_images)
    
    # Save the PCA model
    pca_model_path = os.path.join(output_folder, "pca_transformer.pkl")
    os.makedirs(os.path.dirname(pca_model_path), exist_ok=True)
    with open(pca_model_path, 'wb') as f:
        pickle.dump(pca_model, f)
        
    print(f"PCA model saved to {pca_model_path}")
    print(f"Explained variance ratio: {np.sum(pca_model.explained_variance_ratio_):.4f}")
    
    return pca_model

def load_pca_model(model_path):
    """
    Loads a saved PCA model.
    """
    global pca_model
    
    if pca_model is not None:
        return pca_model
        
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            pca_model = pickle.load(f)
            print(f"Loaded PCA model from {model_path}")
            return pca_model
    
    return None

def get_pca_eigendigit_visualization(image, component_weights=None):
    """
    Creates a visualization showing the principal component patterns (eigendigits).
    This shows which pixel patterns contribute most to digit recognition.
    """
    rows, cols = image.shape
    
    # Create synthetic eigendigit patterns based on common PCA patterns for digits
    eigendigits = []
    
    # Component 1: Overall brightness/intensity pattern
    comp1 = np.ones((rows, cols)) * 0.7
    comp1[rows//4:3*rows//4, cols//4:3*cols//4] = 1.0  # Center emphasis
    eigendigits.append(comp1)
    
    # Component 2: Vertical emphasis (good for 1, 4, 7)
    comp2 = np.zeros((rows, cols))
    comp2[:, cols//3:2*cols//3] = 1.0
    eigendigits.append(comp2)
    
    # Component 3: Horizontal emphasis (good for 2, 3, 5, 7)
    comp3 = np.zeros((rows, cols))
    comp3[rows//3:2*rows//3, :] = 1.0
    eigendigits.append(comp3)
    
    # Component 4: Circular pattern (good for 0, 6, 8, 9)
    comp4 = np.zeros((rows, cols))
    center_r, center_c = rows//2, cols//2
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center_r)**2 + (j - center_c)**2)
            if 6 <= dist <= 12:  # Ring pattern
                comp4[i, j] = 1.0 - abs(dist - 9) / 3
    eigendigits.append(comp4)
    
    # Component 5: Diagonal patterns (good for 4, 7)
    comp5 = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if abs(i - j) <= 2:  # Main diagonal
                comp5[i, j] = 1.0
    eigendigits.append(comp5)
    
    # Calculate projection weights based on input image
    flat_image = image.flatten().astype(np.float32) / 255.0
    weights = []
    
    for eigendigit in eigendigits:
        flat_eigen = eigendigit.flatten()
        # Calculate correlation-like weight
        if np.std(flat_eigen) > 0 and np.std(flat_image) > 0:
            weight = np.corrcoef(flat_image, flat_eigen)[0, 1]
            if np.isnan(weight):
                weight = 0.0
        else:
            weight = 0.0
        weights.append(abs(weight))
    
    # Create visualization by combining weighted eigendigits
    result = np.zeros((rows, cols), dtype=np.float32)
    total_weight = sum(weights) if sum(weights) > 0 else 1.0
    
    for i, (eigendigit, weight) in enumerate(zip(eigendigits, weights)):
        normalized_weight = weight / total_weight
        result += normalized_weight * eigendigit * (image.astype(np.float32) / 255.0)
    
    # Enhance contrast and convert to uint8
    if np.max(result) > 0:
        result = (result / np.max(result) * 255).astype(np.uint8)
    else:
        result = image.copy()
    
    return result

def get_pca(image, pca_model_arg=None, n_components=80):
    """
    Main PCA visualization function that shows eigendigit patterns or PCA reconstruction.
    """
    global pca_model
    
    # Use provided model, global model, or fallback to eigendigit visualization
    model_to_use = pca_model_arg or pca_model
    
    if model_to_use is not None:
        try:
            # Use actual PCA model for reconstruction
            flat_image = image.flatten().astype(np.float32) / 255.0
            transformed = model_to_use.transform([flat_image])
            reconstructed = model_to_use.inverse_transform(transformed)
            result = reconstructed[0].reshape(image.shape)
            # Convert back to 0-255 range
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            return result
        except Exception as e:
            print(f"PCA model failed, using eigendigit visualization: {e}")
    
    # Fallback: Use eigendigit visualization
    return get_pca_eigendigit_visualization(image)

class Transform(IntEnum):
    CROSSING = 0
    ENDPOINT = 1
    FILL = 2
    SKEL_FILL = 3
    SKEL = 4
    THRESH = 5
    LINE = 6
    ELLIPSE = 7
    CIRCLE = 8
    ELLIPSE_CIRCLE = 9
    CONVEX_HULL = 10
    RAW = 11
    CORNER = 12
    PCA = 13
    SIZE = 14

TransformNames = [
    "crossing",
    "endpoint",
    "fill",
    "skel-fill",
    "skel",
    "thresh",
    "line",
    "ellipse",
    "circle",
    "ellipse-circle",
    "chull",
    "raw",
    "corner",
    "pca"
]

def set_image(dest, origin):
    count = 0
    for o in origin:
        dest[count] = o

def get_image_name(folder, transform_name, label, d_count, classes, digit_count):
    #uncomment for filename by class 
    #return folder + "/" + transform_name + "/" + classes[label] + "-" + str(d_count) + "-" + str(digit_count) + ".png"
    #uncomment for filename by index
    return folder + "/" + transform_name + "/" + str(digit_count) + "-" + classes[label] + "-" + str(d_count) + ".png"


def get_corners(image):
    coords = corner_peaks(corner_harris(image), min_distance=3, threshold_rel=0.02)
    corner_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for coord in coords:
        set_neighborhood(corner_image, coord[1], coord[0], 1, 255)
    return corner_image

def get_image_from_list(data):
    pixel_count = len(data)
    dimension = int(math.sqrt(pixel_count))
    print("pixels:", pixel_count, "dimension:", dimension)
    raw = np.zeros((dimension,dimension), np.uint8)
    pixel = 0
    for x in range(dimension):
        for y in range(dimension):
            raw[x,y] = data[pixel]
            pixel += 1
    return raw

def image_data_to_list(data):
    #return data.flatten().tolist()
    l = []
    shape = data.shape
    for x in range(shape[0]):
        for y in range(shape[1]):
            l.append(data[x,y])
    return l

# gets the transforms from the passed array of image data
def get_transforms(data, pca_model_path=None):
    min_thresh = 100
    pixel_count = len(data)
    dimension = int(math.sqrt(pixel_count))
    print("pixels:", pixel_count, "dimension:", dimension)
    raw = np.zeros([dimension,dimension], np.uint8)
    thresh = np.zeros([dimension,dimension], np.uint8)
    img_lee = np.zeros([dimension,dimension], np.uint8)
    pixel = 0
    for x in range(dimension):
        for y in range(dimension):
            if (data[pixel] > min_thresh):
                thresh[x, y] = 255
                img_lee[x,y] = 255
            raw[x,y] = data[pixel]
            pixel += 1
    fill_img = flood_fill_loops(thresh, 100)
    lee_skel_raw = skeletonize(img_lee, method='lee')
    
    # Ensure skeleton is in proper 0-255 range (skeletonize may return 0-255 or boolean)
    if lee_skel_raw.dtype == bool or np.max(lee_skel_raw) <= 1:
        lee_skel = (lee_skel_raw * 255).astype(np.uint8)
        lee_skel_binary = lee_skel_raw
    else:
        lee_skel = lee_skel_raw.astype(np.uint8)  # Already in 0-255 range
        lee_skel_binary = lee_skel > 0
    
    # Process with binary skeleton for algorithms that work with binary data
    corners = get_corners(lee_skel_binary)
    ellipse = get_ellipse(lee_skel_binary)
    circle = get_circle(lee_skel_binary)
    ellipse_circle = get_ellipse_or_circle(lee_skel_binary)
    fill_skel = flood_fill_loops(lee_skel, 100)
    
    # Process with uint8 skeleton for algorithms that expect intensity values
    crossing_count, crossings, mod_crossings = get_crossings(lee_skel, 100)
    print("crossing count:", crossing_count)
    endpoint_count, endpoints, mod_endpoints = get_endpoints(lee_skel, 100)
    print("endpoint count:", endpoint_count)
    lines = get_lines(lee_skel)
    chull = get_convex_hull(thresh)
    
    # Load PCA model if path provided, or use default path
    if pca_model_path:
        load_pca_model(pca_model_path)
    elif pca_model is None:
        # Try default locations
        for default_path in ["transforms/mnist/train/002/pca_transformer.pkl", 
                            "transforms/mnist/train/001/pca_transformer.pkl"]:
            if os.path.exists(default_path):
                load_pca_model(default_path)
                break
    
    pca_vis = get_pca(raw)
    return (raw, thresh, lee_skel, fill_img, corners,
            ellipse, circle, ellipse_circle, fill_skel,
            crossings, endpoints, lines, chull, pca_vis)

def process_image(data):
    global num_processed
    min_thresh = 100
    #(digit, label, shm.name, N, digit_count, count, rows, columns, save, save_location, save_number, classes)
    digit = data[0]
    label = data[1]
    share_name = data[2]
    N = data[3]
    digit_count = data[4]
    count = data[5]
    rows = data[6]
    columns = data[7]
    save_file = data[8]
    save_folder = data[9]
    save_number = data[10]
    classes = data[11]
    folder = "./images/"
    if save_folder != None:
        folder = save_folder
    # get shared memory
    existing_shm = shared_memory.SharedMemory(name=share_name)
    transform_data = np.ndarray((Transform.SIZE, N, rows * columns), dtype=np.uint8, buffer=existing_shm.buf)

    raw = np.zeros([rows,columns], np.uint8)
    img = np.zeros([rows,columns], np.uint8)
    #img_zhang = np.zeros([rows,columns], np.uint8)
    img_lee = np.zeros([rows,columns], np.uint8)
    pixel = 0
    for x in range(rows):
        for y in range(columns):
            #img_zhang[y,x] = 1 if digit[pixel] > 100 else 0
            if (digit[pixel] > min_thresh):
                img[x, y] = 255
                img_lee[x,y] = 255
            raw[x,y] = digit[pixel]
            pixel += 1
    #dil = skimage.morphology.dilation(img)
    #dil2 = skimage.morphology.dilation(dil)
    #opening = skimage.morphology.opening(img)
    fill_img = flood_fill_loops(img, 100)
    bin_skel = skeletonize(img_lee, method='lee')
    
    # Handle skeleton conversion properly - skeletonize may return boolean or 0-255 values
    if bin_skel.dtype == bool:
        lee_skel = (bin_skel * 255).astype(np.uint8)
        bin_skel_binary = bin_skel
    else:
        # Already in 0-255 range, just ensure uint8
        lee_skel = bin_skel.astype(np.uint8)
        bin_skel_binary = bin_skel > 0
        
    corners = get_corners(lee_skel)
    ellipse = get_ellipse(bin_skel_binary)
    circle = get_circle(bin_skel_binary)
    ellipse_circle = get_ellipse_or_circle(bin_skel_binary)
    fill_skel = flood_fill_loops(lee_skel, 100)
    crossing_count, crossings, mod_crossings = get_crossings(lee_skel, 100)
    endpoint_count, endpoints, mod_endpoints = get_endpoints(lee_skel, 100)
    lines = get_lines(lee_skel)
    chull = get_convex_hull(img)
    pca_vis = get_pca(raw)

    if save_file and ( save_number == None or count < save_number):
        io.imsave(get_image_name(folder, "ellipse", label, count, classes, digit_count), ellipse, check_contrast=False)
        io.imsave(get_image_name(folder, "circle", label, count, classes, digit_count), circle, check_contrast=False)
        io.imsave(get_image_name(folder, "ellipse_circle", label, count, classes, digit_count), ellipse_circle, check_contrast=False)
        io.imsave(get_image_name(folder, "skel", label, count, classes, digit_count), skimage.util.img_as_ubyte(lee_skel), check_contrast=False)
        io.imsave(get_image_name(folder, "fill", label, count, classes, digit_count), fill_img, check_contrast=False)
        io.imsave(get_image_name(folder, "skel_fill", label, count, classes, digit_count), fill_skel, check_contrast=False)
        io.imsave(get_image_name(folder, "crossing", label, count, classes, digit_count), crossings, check_contrast=False)
        io.imsave(get_image_name(folder, "endpoint", label, count, classes, digit_count), endpoints, check_contrast=False)
        io.imsave(get_image_name(folder, "lines", label, count, classes, digit_count), lines, check_contrast=False)
        io.imsave(get_image_name(folder, "chull", label, count, classes, digit_count), chull, check_contrast=False)
        io.imsave(get_image_name(folder, "thresh", label, count, classes, digit_count), img, check_contrast=False)
        io.imsave(get_image_name(folder, "raw", label, count, classes, digit_count), raw, check_contrast=False)
        io.imsave(get_image_name(folder, "corner", label, count, classes, digit_count), corners, check_contrast=False)
        io.imsave(get_image_name(folder, "pca", label, count, classes, digit_count), pca_vis, check_contrast=False)

    transform_data[Transform.ELLIPSE][digit_count] = ellipse.flatten()
    transform_data[Transform.CIRCLE][digit_count] = circle.flatten()
    transform_data[Transform.ELLIPSE_CIRCLE][digit_count] = ellipse_circle.flatten()
    transform_data[Transform.SKEL][digit_count] = lee_skel.flatten()
    transform_data[Transform.FILL][digit_count] = fill_img.flatten()
    transform_data[Transform.SKEL_FILL][digit_count] = fill_skel.flatten().astype(np.uint8)
    transform_data[Transform.CROSSING][digit_count] = crossings.flatten()
    transform_data[Transform.ENDPOINT][digit_count] = endpoints.flatten()
    transform_data[Transform.LINE][digit_count] = lines.flatten()
    transform_data[Transform.CONVEX_HULL][digit_count] = chull.flatten()
    transform_data[Transform.THRESH][digit_count] = img.flatten()
    transform_data[Transform.RAW][digit_count] = raw.flatten()
    transform_data[Transform.CORNER][digit_count] = corners.flatten()
    transform_data[Transform.PCA][digit_count] = pca_vis.flatten()

    existing_shm.close()
    current_count = 0
    with num_processed.get_lock():
        num_processed.value += 1
        current_count = num_processed.value
    if current_count % 1000 == 0:
        print("processed image number", current_count, flush=True)

def create_shared_block(n, rows, columns):
    a = np.zeros((Transform.SIZE, n, rows * columns), dtype=np.uint8)
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    # # Now create a NumPy array backed by shared memory
    np_array = np.ndarray(a.shape, dtype=np.uint8, buffer=shm.buf)
    np_array[:] = a[:]  # Copy the original data into shared memory
    return shm, np_array

def init(args):
    global num_processed
    num_processed = args

if __name__ == '__main__':
    start_time = datetime.now()
    num_processed = Value("i", 0)
    parser = argparse.ArgumentParser(description='Saves transformed data and optionally writes images')
    parser.add_argument('-s', '--saveImages', action='store_const',
                    const=True, default=False, 
                    help='Flag to save images, default does not save.')
    parser.add_argument('-i', '--imageFile', 
                        help='The mnist image input file')
    parser.add_argument('-l', '--labelFile', 
                        help='The mnist label input file')
    parser.add_argument('-o', '--outputFolder', 
                        help='The folder to output')
    parser.add_argument('-a', '--saveLocation', 
                        help='The folder to output saved images. Only used if saveImages is set -s.')
    parser.add_argument('-n', '--numberSaved', 
                        help='The number of saved images', type=int)
    parser.add_argument('-d', '--digits', action='store_const',
                    const=True, default=False, 
                    help='Flag to use digits.')
    parser.add_argument('-c', '--cap', action='store_const',
                    const=True, default=False, 
                    help='Flag to use capital letters.')
    parser.add_argument('-w', '--lower', action='store_const',
                    const=True, default=False, 
                    help='Flag to use lowercase letters.')
    parser.add_argument('--column', action='store_const',
                    const=True, default=False, 
                    help='Flag to use column ordering when reading data.')
    args = parser.parse_args()

    ordering = "row"
    if args.column:
        ordering = "column"
    classes = emnist_classes
    if args.cap:
        classes = emnist_cap_letter_classes
    elif args.digits:
        classes = emnist_digit_classes
    elif args.lower:
        classes = emnist_lower_letter_classes
 
    N, rows, columns, digits, labels = load_mnist(args.imageFile, args.labelFile, ordering)
    shm, transform_data = create_shared_block(N, rows, columns)

    # Create PCA model from raw images for consistent transforms
    print("Creating PCA model from raw images...")
    raw_images = []
    for digit in digits[:min(10000, len(digits))]:  # Use subset for efficiency
        raw_img = np.zeros([rows, columns], np.uint8)
        pixel = 0
        for x in range(rows):
            for y in range(columns):
                raw_img[x, y] = digit[pixel]
                pixel += 1
        raw_images.append(raw_img)
    
    # Create and save PCA model
    create_and_save_pca_model(raw_images, args.outputFolder)

    digit_count = 0
    digit_counts = np.zeros(get_num_classes(labels), int)
    i = 0
    params = []
    for digit in digits:
        label = labels[digit_count]
        count = digit_counts[label]
        digit_counts[label] += 1
        params.append((digit, label, shm.name, N, digit_count, count, rows, columns, args.saveImages, args.saveLocation, args.numberSaved, classes))
        digit_count += 1

    num_cpus = cpu_count()
    pool = Pool(num_cpus -1, initializer=init, initargs=(num_processed,))
    pool.map(process_image, params)

    write_partial_mnist_data(transform_data[Transform.CROSSING], labels, N, args.outputFolder + "/crossing-image", args.outputFolder + "/crossing-labels")
    write_partial_mnist_data(transform_data[Transform.ENDPOINT], labels, N, args.outputFolder + "/endpoint-image", args.outputFolder + "/endpoint-labels")
    write_partial_mnist_data(transform_data[Transform.FILL], labels, N, args.outputFolder + "/fill-image", args.outputFolder + "/fill-labels")
    write_partial_mnist_data(transform_data[Transform.SKEL_FILL], labels, N, args.outputFolder + "/skel-fill-image", args.outputFolder + "/skel-fill-labels")
    write_partial_mnist_data(transform_data[Transform.SKEL], labels, N, args.outputFolder + "/skel-image", args.outputFolder + "/skel-labels")
    write_partial_mnist_data(transform_data[Transform.THRESH], labels, N, args.outputFolder + "/thresh-image", args.outputFolder + "/thresh-labels")
    write_partial_mnist_data(transform_data[Transform.LINE], labels, N, args.outputFolder + "/line-image", args.outputFolder + "/line-labels")
    write_partial_mnist_data(transform_data[Transform.ELLIPSE], labels, N, args.outputFolder + "/ellipse-image", args.outputFolder + "/ellipse-labels")
    write_partial_mnist_data(transform_data[Transform.CIRCLE], labels, N, args.outputFolder + "/circle-image", args.outputFolder + "/circle-labels")
    write_partial_mnist_data(transform_data[Transform.ELLIPSE_CIRCLE], labels, N, args.outputFolder + "/ellipse_circle-image", args.outputFolder + "/ellipse_circle-labels")
    write_partial_mnist_data(transform_data[Transform.CONVEX_HULL], labels, N, args.outputFolder + "/chull-image", args.outputFolder + "/chull-labels")
    write_partial_mnist_data(transform_data[Transform.RAW], labels, N, args.outputFolder + "/raw-image", args.outputFolder + "/raw-labels")
    write_partial_mnist_data(transform_data[Transform.CORNER], labels, N, args.outputFolder + "/corner-image", args.outputFolder + "/corner-labels")
    write_partial_mnist_data(transform_data[Transform.PCA], labels, N, args.outputFolder + "/pca-image", args.outputFolder + "/pca-labels")

    # TODO, consider taking threshold and doing all processing on binary image

    end_time = datetime.now()
    shm.close()
    shm.unlink()
    pool.terminate()
    pool.join()
    pool.close()
    print("total processed:", num_processed.value, "in:", end_time - start_time)
