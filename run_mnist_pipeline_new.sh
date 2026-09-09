#!/bin/bash
set -ex

mkdir -p ./new_transforms/mnist/train

python transform_parallel.py -i ~/mnist/download/train-images-idx3-ubyte -l ~/mnist/download/train-labels-idx1-ubyte -o ./new_transforms/mnist/train

mkdir -p ./new_transforms/mnist/test

python transform_parallel.py -i ~/mnist/download/t10k-images-idx3-ubyte -l ~/mnist/download/t10k-labels-idx1-ubyte -o ./new_transforms/mnist/test

mkdir -p ./new_models/mnist/svm

python train_svm_transforms.py -r ./new_transforms/mnist/train -e ./new_transforms/mnist/test -o ./new_models/mnist/svm > out-new-svm-mnist-training.txt

mkdir -p ./new_kb/mnist/svm

cp ./new_models/mnist/svm/*.json new_kb/mnist/svm

python build_kb.py -r ./new_transforms/mnist/train -e ./new_transforms/mnist/test -m ./new_models/mnist/svm -o new_kb/mnist/svm

python process_kb_conf_matrix.py -k new_kb/mnist/svm

node process_kb.js -k new_kb/mnist/svm -e product > new_output-mnist-svm-result-02.txt

