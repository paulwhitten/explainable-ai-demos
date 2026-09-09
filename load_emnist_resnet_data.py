import struct
import numpy as np
import tensorflow as tf
from load_mnist_data import load_mnist
from skimage import io

emnist_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                  "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                  "U", "V", "W", "X", "Y", "Z", "a", "b", "d", "e",
                  "f", "g", "h", "n", "q", "r", "t"]

# 0 - 9
emnist_digit_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# 10 - 46
emnist_all_letter_classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                  "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                  "U", "V", "W", "X", "Y", "Z", "a", "b", "d", "e",
                  "f", "g", "h", "n", "q", "r", "t"]

# 10 - 35
emnist_cap_letter_classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                  "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                  "U", "V", "W", "X", "Y", "Z"]

# 36 - 46
emnist_lower_letter_classes = ["a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"]

# processes the input images so they can be used
# with resnet50 for tensorflow
#
#  images a numpy array of 784 784 pixel grayscale mnist
#         images as unsigned 8 bit integers
#
#  output will be an array of 32x32 color images
#         using float32
def process_mnist_images_for_resnet(images):
    # resize the flattened image arrays as (28,28)
    # maintaining the original dimension
    images.resize((images.shape[0],28,28))
    # expand new axis for color channels
    expanded_images = np.expand_dims(images, axis=-1)
    # put the grayscale data in each of the channels
    grayscale_images = np.repeat(expanded_images, 3, axis=-1)
    # normalize on [0.0 - 1.0]
    float_images = grayscale_images.astype('float32') / 255
    # resize to 32x32
    resized_float_images = tf.image.resize(float_images, [32,32])
    del images
    del expanded_images
    del grayscale_images
    del float_images
    return resized_float_images


# combines the input images so they can be used
# with resnet50 for tensorflow
#
#  im1, im2, im3 numpy arrays of 784 784 pixel grayscale mnist
#         images as unsigned 8 bit integers
#
#  output will be an array of 32x32 color images
#         using float32
def combine_mnist_images_for_resnet(im1, im2, im3):
    # resize the flattened image arrays as (28,28)
    # maintaining the original dimension
    im1.resize((im1.shape[0],28,28))
    im2.resize((im2.shape[0],28,28))
    im3.resize((im3.shape[0],28,28))
    # expand new axis for color channels
    expanded_images = np.expand_dims(im1, axis=-1)
    # put the grayscale data in each of the channels
    color_images = np.repeat(expanded_images, 3, axis=-1)
    for img_ix, im in enumerate(color_images):
        for row_ix, row in enumerate(im):
            for col_ix, col in enumerate(row):
                col[1] = im2[img_ix][row_ix][col_ix]
                col[2] = im3[img_ix][row_ix][col_ix]

    for i in range(25):
        io.imsave("./images/" + str(len(im1)) + "-" + str(i) + ".png", color_images[i]) #check_contrast=False
    # normalize on [0.0 - 1.0]
    float_images = color_images.astype('float32') / 255
    # resize to 32x32
    resized_float_images = tf.image.resize(float_images, [32,32])
    del expanded_images
    del color_images
    del float_images
    return resized_float_images

# Convert from a class to a one hot representation
# assume we have input of integer array.  Each element
# is converted to an array with the index of the label
# as hot.
def process_labels_for_resnet(labels):
    min_label = min(labels)
    max_label = max(labels)
    labels = tf.keras.utils.to_categorical(labels , num_classes=max_label+1)
    return labels, max_label + 1


def load_emnist_resnet(image_file, label_file):
    num_digits, rows, columns, images, labels = load_mnist(image_file, label_file)
    images = np.array(images).astype("uint8")
    images = process_mnist_images_for_resnet(images)
    labels = np.array(labels).astype("uint8")
    cat_labels, num_labels = process_labels_for_resnet(labels)
    return images, cat_labels, num_labels


def load_mnist_labels(label_file):
    label_fd = open(label_file, "rb")
    image_labels = label_fd.read()

    # initial position to read to for header
    labelPosition = 8

    # read big endian header
    (label_magic, numLabels) = struct.unpack(">ii", image_labels[:labelPosition])
    print('label magic num:', hex(label_magic), 'numLabels:', numLabels)

    labels = []
    count = 0
    while count < numLabels:
        # read a byte buffer for the label
        label = struct.unpack("B", image_labels[labelPosition:labelPosition+1])
        # advance the position
        labelPosition += 1
        labels.append(label[0])
        count += 1
    label_fd.close()
    return labels

def array_to_image(pixels, rows, columns):
    img = np.zeros((rows, columns), np.uint8)
    pixel_count = 0
    for x in range(rows):
        for y in range(columns):
            img[x,y] = pixels[pixel_count]
            pixel_count += 1
    return img