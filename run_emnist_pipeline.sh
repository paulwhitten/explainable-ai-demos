#!/bin/bash

set -xe

#########################
# full emnist pipelines
./pipeline.py -c pipeline_config_emnist_full_cnn.json
./pipeline.py -c pipeline_config_emnist_full_nn.json
./pipeline.py -c pipeline_config_emnist_full_resnet.json
./pipeline.py -c pipeline_config_emnist_full_svm.json

#########################
# reduce cap emnist
mkdir -p ./transforms/emnist_reduced/train/cap

./reduce_dataset.py -i /home/pcw/emnist/data/emnist-balanced-train-images-idx3-ubyte -l /home/pcw/emnist/data/emnist-balanced-train-labels-idx1-ubyte -o ./transforms/emnist_reduced/train -n cap -c 10000 -s 10 -e 35 -t 0

mkdir -p ./transforms/emnist_reduced/test/cap

./reduce_dataset.py -i /home/pcw/emnist/data/emnist-balanced-test-images-idx3-ubyte -l /home/pcw/emnist/data/emnist-balanced-test-labels-idx1-ubyte -o ./transforms/emnist_reduced/test -n cap -c 10000 -s 10 -e 35 -t 0

# run emnist cap pipelines
./pipeline.py -c pipeline_config_emnist_cap_cnn.json
./pipeline.py -c pipeline_config_emnist_cap_nn.json
./pipeline.py -c pipeline_config_emnist_cap_resnet.json
./pipeline.py -c pipeline_config_emnist_cap_svm.json

#########################
# reduce lower emnist
mkdir -p ./transforms/emnist_reduced/train/lower

./reduce_dataset.py -i /home/pcw/emnist/data/emnist-balanced-train-images-idx3-ubyte -l /home/pcw/emnist/data/emnist-balanced-train-labels-idx1-ubyte -o ./transforms/emnist_reduced/train -n lower -c 10000 -s 36 -e 46 -t 0

mkdir -p ./transforms/emnist_reduced/test/lower

./reduce_dataset.py -i /home/pcw/emnist/data/emnist-balanced-test-images-idx3-ubyte -l /home/pcw/emnist/data/emnist-balanced-test-labels-idx1-ubyte -o ./transforms/emnist_reduced/test -n lower -c 10000 -s 36 -e 46 -t 0

# run emnist lower pipelines
./pipeline.py -c pipeline_config_emnist_lower_cnn.json
./pipeline.py -c pipeline_config_emnist_lower_nn.json
./pipeline.py -c pipeline_config_emnist_lower_resnet.json
./pipeline.py -c pipeline_config_emnist_lower_svm.json

#########################
# reduce digit emnist
mkdir -p ./transforms/emnist_reduced/train/digit

./reduce_dataset.py -i /home/pcw/emnist/data/emnist-balanced-train-images-idx3-ubyte -l /home/pcw/emnist/data/emnist-balanced-train-labels-idx1-ubyte -o ./transforms/emnist_reduced/train -n digit -c 10000 -s 0 -e 9 -t 0

mkdir -p ./transforms/emnist_reduced/test/digit

./reduce_dataset.py -i /home/pcw/emnist/data/emnist-balanced-test-images-idx3-ubyte -l /home/pcw/emnist/data/emnist-balanced-test-labels-idx1-ubyte -o ./transforms/emnist_reduced/test -n digit -c 10000 -s 0 -e 9 -t 0

# run emnist digit pipelines
./pipeline.py -c pipeline_config_emnist_digit_cnn.json
./pipeline.py -c pipeline_config_emnist_digit_nn.json
./pipeline.py -c pipeline_config_emnist_digit_resnet.json
./pipeline.py -c pipeline_config_emnist_digit_svm.json
