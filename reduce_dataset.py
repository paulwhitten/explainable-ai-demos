#!/usr/bin/env python

import argparse
import sys
from ast import And, Not
from load_mnist_data import load_mnist, write_partial_mnist_data

def check_for_done(digit_counts, start, end, count):
    for i in range(start, end+1):
        if digit_counts[i] != count:
            return False
    return True

parser = argparse.ArgumentParser(description='reduces a dataset')
parser.add_argument('-i', '--image_file', 
                    help='The mnist image input file')
parser.add_argument('-l', '--label_file', 
                    help='The mnist label input file')
parser.add_argument('-o', '--output_folder', 
                    help='The folder to output')
parser.add_argument('-n', '--name', 
                    help='The output name')
parser.add_argument('-c', '--count', 
                    help='The number of each sample', type=int)
parser.add_argument('-s', '--start_class', 
                    help='the class to start with', type=int)
parser.add_argument('-e', '--end_class', 
                    help='The class to end with', type=int)
parser.add_argument('-t', '--start', 
                    help='The number for class 0', type=int)
args = parser.parse_args()

# sample
# reduce_dataset.py -i /Users/pcw/mnist/download/train-images-idx3-ubyte -l /Users/pcw/mnist/download/train-labels-idx1-ubyte -o small -n small_0_4 -c 1500 -s 0 -e 4
# reduce_dataset.py -i /Users/pcw/mnist/download/t10k-images-idx3-ubyte -l /Users/pcw/mnist/download/t10k-labels-idx1-ubyte -o small -n test_small_0_4 -c 250 -s 0 -e 4
# reduce_dataset.py -i /home/pcw/emnist/data/emnist-balanced-train-images-idx3-ubyte -l /home/pcw/emnist/data/emnist-balanced-train-labels-idx1-ubyte -o small -n train_small_0_46 -c 125 -s 0 -e 46
# upper case
# python reduce_dataset.py -i /home/pcw/emnist/data/emnist-balanced-train-images-idx3-ubyte -l /home/pcw/emnist/data/emnist-balanced-train-labels-idx1-ubyte -o emnist-cap-letters -n cap-letters -c 5000 -s 10 -e 35 -t 0
# upper case letters test
# python reduce_dataset.py -i /home/pcw/emnist/data/emnist-balanced-test-images-idx3-ubyte -l /home/pcw/emnist/data/emnist-balanced-test-labels-idx1-ubyte -o transforms/emnist-cap-letters-test -n cap-letters -c 5000 -s 10 -e 35 -t 0
# lower case
# python reduce_dataset.py -i /home/pcw/emnist/data/emnist-balanced-train-images-idx3-ubyte -l /home/pcw/emnist/data/emnist-balanced-train-labels-idx1-ubyte -o transforms/emnist-lower-letters -n lower-letters -c 5000 -s 36 -e 46 -t 0
# lower case test
# python reduce_dataset.py -i /home/pcw/emnist/data/emnist-balanced-test-images-idx3-ubyte -l /home/pcw/emnist/data/emnist-balanced-test-labels-idx1-ubyte -o transforms/emnist-lower-letters-test -n lower-letters -c 5000 -s 36 -e 46 -t 0
# digits
# python reduce_dataset.py -i /home/pcw/emnist/data/emnist-balanced-train-images-idx3-ubyte -l /home/pcw/emnist/data/emnist-balanced-train-labels-idx1-ubyte -o transforms/emnist-digits -n digits -c 5000 -s 0 -e 9 -t 0
# digits test
# python reduce_dataset.py -i /home/pcw/emnist/data/emnist-balanced-test-images-idx3-ubyte -l /home/pcw/emnist/data/emnist-balanced-test-labels-idx1-ubyte -o transforms/emnist-digits-test -n digits -c 5000 -s 0 -e 9 -t 0


N, rows, columns, digits, labels = load_mnist(args.image_file, args.label_file)

# get the range of classes
label_min = min(labels)
label_max = max(labels)
counts = [0] * (label_max + 1)

# make sure classes are sane
assert args.end_class >= args.start_class
assert args.end_class <= label_max
assert args.start_class >= label_min

output_digits = []
output_labels = []

i = 0
max_relabel = 0
min_relabel = sys.maxsize
while i < N and not check_for_done(counts, args.start_class, args.end_class, args.count):
    if labels[i] >= args.start_class and labels[i] <= args.end_class and counts[labels[i]] < args.count:
        relabel = labels[i]
        if args.start != None:
            relabel = relabel - args.start_class + args.start
        if max_relabel < relabel:
            max_relabel = relabel
        if min_relabel > relabel:
            min_relabel = relabel
        output_digits.append(digits[i])
        output_labels.append(relabel)
        counts[labels[i]] += 1
    i += 1

print("counts", counts)
print("min label:", min_relabel, "max label:", max_relabel)

image_path = "{}/{}/images".format(args.output_folder, args.name)
label_path = "{}/{}/labels".format(args.output_folder, args.name)

total = sum(counts)
print("image file:", image_path, "label file:", label_path, "count:", total)

write_partial_mnist_data(output_digits, output_labels, total, image_path, label_path)
