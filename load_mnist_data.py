import struct # https://docs.python.org/3/library/struct.html#struct.unpack
import numpy as np

COLUMN_ORDERING = "column"
ROW_ORDERING = "row"

def array_to_image(pixels, rows, columns):
    img = np.zeros((rows, columns), np.uint8)
    pixel_count = 0
    for x in range(rows):
        for y in range(columns):
            img[x,y] = pixels[pixel_count]
            pixel_count += 1
    return img

# loads the mnist image and label file for processing in ml
# output is:
#   the number digits
#   the number of rows per digit
#   the number of columns per digit
#   an array of normalized pixel data
#   an array of floating point label data
def load_mnist_float(image_file, label_file):
    image_data = open(image_file, "rb").read()
    image_labels = open(label_file, "rb").read()

    # initial position to read to for header
    image_position = 16
    label_position = 8

    # read big endian header
    (image_magic, N, rows, columns) = struct.unpack(">iiii", image_data[:image_position])
    (label_magic, num_labels) = struct.unpack(">ii", image_labels[:label_position])
    if (N != num_labels):
        print("number of labels does not correspond to digits")

    pixels_in_image =  rows * columns

    digits = []
    labels_integer = np.zeros(N).astype('uint8')
    image_count = 0
    while image_count < N:
        # read a byte buffer for the label and then the image
        label = struct.unpack("B", image_labels[label_position:label_position+1])
        pixels = struct.unpack("B" * pixels_in_image, image_data[image_position: image_position + pixels_in_image])
        normPixels = []
        # advance the position
        image_position += rows * columns
        label_position += 1
        for pixel in pixels:
            normPixels.append(pixel/255)
        digits.append(normPixels)
        labels_integer[image_count] = label[0]
        image_count += 1
    labels_one_hot = []
    largest_label = max(labels_integer)
    labels_one_hot = np.zeros((N, largest_label + 1)).astype('float32')
    labels_one_hot[np.arange(N), labels_integer] = 1.0
    return (N, rows, columns, digits, labels_one_hot)

# loads the mnist image and label file 
# output is:
#   The number of digits
#   the number of rows per digit
#   the number of columns per digit
#   an int array of digit pixel data
#   an array of integer labels
def load_mnist(image_file, label_file, ordering=ROW_ORDERING):
    print("Loading mnist...")
    image_data = open(image_file, "rb").read()
    image_labels = open(label_file, "rb").read()

    # initial position to read to for header
    image_position = 16
    label_position = 8

    # read big endian header
    (image_magic, N, rows, columns) = struct.unpack(">iiii", image_data[:image_position])
    print('image magic num:', hex(image_magic), 'N:', N, 'rows:', rows, 'columns:', columns)
    (label_magic, num_labels) = struct.unpack(">ii", image_labels[:label_position])
    print('label magic num:', hex(label_magic), 'numLabels:', num_labels)
    if (N != num_labels):
        print("number of labels does not correspond to digits")

    pixels_in_image =  rows * columns
    print("pixels in an image:", pixels_in_image, "rows:", rows, "columns:", columns)

    digits = []
    labels = []
    imageCount = 0
    while imageCount < N:
        # read a byte buffer for the label and then the image
        label = struct.unpack("B", image_labels[label_position:label_position+1])
        pixels = struct.unpack("B" * pixels_in_image, image_data[image_position: image_position + pixels_in_image])
        if ordering == COLUMN_ORDERING:
            np_pixels = np.array(pixels)
            pixels = np_pixels.reshape((rows, columns), order="F").flatten()
        else:
            pixels = np.array(pixels)
        # advance the position
        image_position += rows * columns
        label_position += 1
        digits.append(pixels)
        labels.append(label[0])
        imageCount += 1
        #if imageCount % 10000 == 0:
        #    print("count:", imageCount)
    return (N, rows, columns, digits, labels)

# Reads the label file
# Input:
#   The label file name
# Output:
#   A list of labels read from the file
def load_mnist_labels(label_file):
    image_labels = open(label_file, "rb").read()
    label_position = 8
    (label_magic, num_labels) = struct.unpack(">ii", image_labels[:label_position])
    label_count = 0
    labels = []
    while label_count < num_labels:
        label = struct.unpack("B", image_labels[label_position:label_position+1])
        label_position += 1
        label_count += 1
        labels.append(label[0])
    return labels

# this method will write mnist data to file
#  images is the input  array of images to write
#  labels is the input array of image lables to write
#  count is the number of items to write to the file. expected to be <= sice of input arrays
#  image and label filnames are the files to output to.  overwrites existing data
def write_partial_mnist_data(images, labels, count, image_filename, label_filename):
    rows = 28
    columns = 28
    max = count
    image_buffer_pos = 16
    label_buffer_pos = 8
    image_buffer = bytearray(image_buffer_pos + max * rows * columns)
    label_buffer = bytearray(label_buffer_pos + max)

    struct.pack_into(">iiii", image_buffer, 0, 2051, max, 28, 28)
    struct.pack_into(">ii", label_buffer, 0, 2049, max)
    pixels_in_image = rows * columns
    for x in range(max):
        struct.pack_into("B" * pixels_in_image, image_buffer, image_buffer_pos, *images[x])
        struct.pack_into("B", label_buffer, label_buffer_pos, labels[x])
        image_buffer_pos += pixels_in_image
        label_buffer_pos += 1

    f=open(image_filename,"wb")
    f.write(image_buffer)
    f.close()

    f=open(label_filename,"wb")
    f.write(label_buffer)
    f.close()

def get_num_classes(train_labels):
    train_label_min = min(train_labels)
    train_label_max = max(train_labels)
    print("Class labels, min:", train_label_min, "max:", train_label_max)
    # assuming class labels are zero through train_label_max giving train_label_max + 1 classes
    return train_label_max + 1


# loads the mnist image and label file for processing in ml
# output is:
#   the number digits
#   the number of rows per digit
#   the number of columns per digit
#   an array of normalized pixel data
#   an array of floating point label data
# TODO: pass range to normalize on...
def load_mnist_float_tanh(image_file, label_file):
    image_data = open(image_file, "rb").read()
    image_labels = open(label_file, "rb").read()

    # initial position to read to for header
    image_position = 16
    label_position = 8

    # read big endian header
    (imageMagic, N, rows, columns) = struct.unpack(">iiii", image_data[:image_position])
    (labelMagic, num_labels) = struct.unpack(">ii", image_labels[:label_position])
    if (N != num_labels):
        print("number of labels does not correspond to digits")

    pixels_in_image =  rows * columns

    digits = []
    labels = []
    image_count = 0
    while image_count < N:
        # read a byte buffer for the label and then the image
        label = struct.unpack("B", image_labels[label_position:label_position+1])
        pixels = struct.unpack("B" * pixels_in_image, image_data[image_position: image_position + pixels_in_image])
        normPixels = []
        # advance the position
        image_position += rows * columns
        label_position += 1
        for pixel in pixels:
            normPixels.append(pixel/255 * 2.0 - 1.0)
        digits.append(normPixels)
        output = np.zeros(10)
        output[label[0]] = 1.0
        labels.append(output)
        image_count += 1
    return (N, rows, columns, digits, labels)

# loads the mnist image and label file for processing in ml
# output is:
#   the number digits
#   the number of rows per digit
#   the number of columns per digit
#   an array of normalized pixel data
#   an array of floating point label data
def load_mnist_float_old(image_file, label_file):
    image_fd = open(image_file, "rb")
    image_data =  image_fd.read()
    label_fd = open(label_file, "rb")
    image_labels = label_fd.read()

    print("loading float arrays:", image_file, "labels:", label_file)

    # initial position to read to for header
    imagePosition = 16
    labelPosition = 8

    # read big endian header
    (image_magic, N, rows, columns) = struct.unpack(">iiii", image_data[:imagePosition])
    (label_magic, numLabels) = struct.unpack(">ii", image_labels[:labelPosition])
    if (N != numLabels):
        print("number of labels does not correspond to digits")

    pixelsInImage =  rows * columns

    digits = []
    labels = []
    imageCount = 0
    while imageCount < N:
        # read a byte buffer for the label and then the image
        label = struct.unpack("B", image_labels[labelPosition:labelPosition+1])
        pixels = struct.unpack("B" * pixelsInImage, image_data[imagePosition: imagePosition + pixelsInImage])
        imagePosition += rows * columns
        labelPosition += 1
        digits.append(pixels)
        labels.append(label[0])
        imageCount += 1
    image_fd.close()
    label_fd.close()
    return (N, rows, columns, np.array(digits).astype("float32") / 255, np.array(labels).astype("uint8"))

def load_mnist_float_images(image_file, label_file):
    image_fd = open(image_file, "rb")
    image_data =  image_fd.read()
    label_fd = open(label_file, "rb")
    image_labels = label_fd.read()

    print("loading float arrays:", image_file, "labels:", label_file)

    # initial position to read to for header
    imagePosition = 16
    labelPosition = 8

    # read big endian header
    (image_magic, N, rows, columns) = struct.unpack(">iiii", image_data[:imagePosition])
    (label_magic, numLabels) = struct.unpack(">ii", image_labels[:labelPosition])
    if (N != numLabels):
        print("number of labels does not correspond to digits")

    pixelsInImage =  rows * columns

    digits = []
    labels = []
    imageCount = 0
    while imageCount < N:
        # read a byte buffer for the label and then the image
        label = struct.unpack("B", image_labels[labelPosition:labelPosition+1])
        pixels = struct.unpack("B" * pixelsInImage, image_data[imagePosition: imagePosition + pixelsInImage])
        imagePosition += rows * columns
        labelPosition += 1
        digits.append(pixels)
        labels.append(label[0])
        imageCount += 1
    image_fd.close()
    label_fd.close()
    images = []
    for digit in digits:
        images.append(array_to_image(digit, columns, rows))
    return (N, rows, columns, np.array(images).astype("float32") / 255, np.array(labels).astype("uint8"))