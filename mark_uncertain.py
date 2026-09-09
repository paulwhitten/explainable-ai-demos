#!/usr/bin/env python

import os
import argparse
import numpy as np
from datetime import datetime
from load_mnist_data import get_num_classes, load_mnist, write_partial_mnist_data, load_mnist_labels

IMAGE_SUFFIX = "-image"
LABEL_SUFFIX = "-labels"

#N, train_rows, train_columns, train_images, train_labels = load_mnist_float(batch[1], batch[2])
#label_array = load_mnist_labels(batch[2])

transforms = [
    "crossing",
    "endpoint",
    "fill",
    "skel-fill",
    "skel",
    "thresh",
    "line",
    "ellipse",
    "circle",
    "ellipse_circle",
    "chull",
    "raw",
    "corner"
]

if __name__ == '__main__':
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description='Converts transformed data to include uncertain transforms')
    parser.add_argument('-f', '--folder', 
                        help='The folder to process')
    parser.add_argument('-o', '--output_folder', 
                        help='The folder to output to')
    args = parser.parse_args()


    for transform in transforms:
        print("Transform:", transform)
        count = 0
        num_changed = 0
        # load
        num_samples, num_rows, num_cols, images, one_hot_labels = load_mnist(os.path.join(args.folder, transform + IMAGE_SUFFIX), os.path.join(args.folder, transform + LABEL_SUFFIX))
        label_array = load_mnist_labels(os.path.join(args.folder, transform + LABEL_SUFFIX))
        # get num labels
        # assuming first class starts with zero
        num_labels = get_num_classes(label_array)
        print("Number of classes:", num_labels)
        for ix, image in enumerate(images):
            if max(image) == 0:
                label_array[ix] = num_labels
                num_changed += 1
            count += 1
        write_partial_mnist_data(images, label_array, num_samples, os.path.join(args.output_folder, transform + IMAGE_SUFFIX), os.path.join(args.output_folder, transform + LABEL_SUFFIX))
        print("\tcount:", count, "changed:", num_changed)

    end_time = datetime.now()
    print("Took:", end_time - start_time)
