#!/bin/bash

#mkdir -p ./transforms/emnist/train

#python transform_parallel.py -i /archive/xai/emnist/downloads/emnist-balanced-train-images-idx3-ubyte -l /archive/xai/emnist/downloads/emnist-balanced-train-labels-idx1-ubyte -o ./transforms/emnist/train

#mkdir -p ./transforms/emnist/test

#python transform_parallel.py -i /archive/xai/emnist/downloads/emnist-balanced-test-images-idx3-ubyte -l /archive/xai/emnist/downloads/emnist-balanced-test-labels-idx1-ubyte -o ./transforms/emnist/test

mkdir -p ./models/mnist/svm

python train_svm_transforms.py -r ./transforms/mnist/train -e ./transforms/mnist/test -o ./models/mnist/svm > out-svm-mnist-training.txt

mkdir -p ./kb/mnist/svm

cp ./models/mnist/svm/*.json kb/mnist/svm

python build_kb.py -r ./transforms/mnist/train -e ./transforms/mnist/test -m ./models/mnist/svm -o kb/mnist/svm

node process_kb.js -k kb/mnist/svm > output-mnist-svm-result.txt

