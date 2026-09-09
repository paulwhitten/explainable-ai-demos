#!/usr/bin/env python

import argparse
from load_mnist_data import load_mnist, load_mnist_float_images
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

#["crossing", f'{args.train_folder}/crossing-image', f'{args.train_folder}/crossing-labels',
#f'{args.test_folder}/crossing-image', f'{args.test_folder}/crossing-labels', args.output_folder],

def train_cnn(batch_data):
    print("Loading original training data for", batch_data[0], flush=True)
    batch_size = 128
    num_epochs = 40

    start_time = datetime.now()

    # load training data
    train_n, train_rows, train_columns, train_images, train_labels = load_mnist_float_images(
        batch_data[1],
        batch_data[2]
    )
    train_images = train_images.reshape(-1, 28, 28, 1)
    print("Training shape:", train_images.shape, flush=True)
    #N, rows, columns, w_digits, w_labels = load_mnist(
    #    batch_data[1],
    #    batch_data[2]
    #)
    #train_images, train_labels, num_train_labels = load_emnist_resnet(
    #    "/home/pcw/mnist/download/train-images-idx3-ubyte",
    #    "/home/pcw/mnist/download/train-labels-idx1-ubyte"
    #)

    test_n, test_rows, test_columns, test_images, test_labels = load_mnist_float_images(
        batch_data[3],
        batch_data[4]
    )
    test_images = test_images.reshape(-1, 28, 28, 1)
    print("Testing shape:", test_images.shape, flush=True)
    # load test data
    #test_images, test_labels, num_test_labels = load_emnist_resnet(
    #    "/home/pcw/mnist/download/t10k-images-idx3-ubyte",
    #    "/home/pcw/mnist/download/t10k-labels-idx1-ubyte"
    #)

    train_label_list = train_labels.tolist()
    train_label_max = np.max(train_label_list)
    train_label_min = np.min(train_label_list)
    num_classes = train_label_max - train_label_min + 1

    CNN = keras.Sequential(
        [
            keras.Input(shape=(28,28,1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    CNN.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # fit 
    cat_train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
    history = CNN.fit(train_images, cat_train_labels, batch_size=batch_size, epochs=num_epochs, verbose=2)

    print("History:", flush=True)
    for key in history.history.keys():
        print("Key:", key, "value:", history.history[key], flush=True)

    cat_test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)
    scores = CNN.evaluate(test_images, cat_test_labels, verbose=1)
    print("Test loss:", scores[0], flush=True)
    print("Test accuracy:", scores[1], flush=True)
    print(scores, flush=True)
    save_path = batch_data[5] + "/" + batch_data[0] + ".keras"
    CNN.save(save_path)
    CNN.summary()
    del CNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trains transforms for a set')
    parser.add_argument('-r', '--train_folder', 
                        help='The training input folder')
    parser.add_argument('-e', '--test_folder', 
                        help='The test input folder')
    parser.add_argument('-o', '--output_folder', 
                        help='The folder to output')
    args = parser.parse_args()

    batches = [
        ["crossing", f'{args.train_folder}/crossing-image', f'{args.train_folder}/crossing-labels',
        f'{args.test_folder}/crossing-image', f'{args.test_folder}/crossing-labels', args.output_folder],

        ["endpoint", f'{args.train_folder}/endpoint-image', f'{args.train_folder}/endpoint-labels',
        f'{args.test_folder}/endpoint-image', f'{args.test_folder}/endpoint-labels', args.output_folder],

        ["fill", f'{args.train_folder}/fill-image', f'{args.train_folder}/fill-labels',
        f'{args.test_folder}/fill-image', f'{args.test_folder}/fill-labels', args.output_folder],

        ["skel-fill", f'{args.train_folder}/skel-fill-image', f'{args.train_folder}/skel-fill-labels',
        f'{args.test_folder}/skel-fill-image', f'{args.test_folder}/skel-fill-labels', args.output_folder],

        ["skel", f'{args.train_folder}/skel-image', f'{args.train_folder}/skel-labels',
        f'{args.test_folder}/skel-image', f'{args.test_folder}/skel-labels', args.output_folder],

        ["thresh", f'{args.train_folder}/thresh-image', f'{args.train_folder}/thresh-labels',
        f'{args.test_folder}/thresh-image', f'{args.test_folder}/thresh-labels', args.output_folder],

        ["line", f'{args.train_folder}/line-image', f'{args.train_folder}/line-labels',
        f'{args.test_folder}/line-image', f'{args.test_folder}/line-labels', args.output_folder],

        ["ellipse", f'{args.train_folder}/ellipse-image', f'{args.train_folder}/ellipse-labels',
        f'{args.test_folder}/ellipse-image', f'{args.test_folder}/ellipse-labels', args.output_folder],

        ["circle", f'{args.train_folder}/circle-image', f'{args.train_folder}/circle-labels',
        f'{args.test_folder}/circle-image', f'{args.test_folder}/circle-labels', args.output_folder],

        ["ellipse-circle", f'{args.train_folder}/ellipse_circle-image', f'{args.train_folder}/ellipse_circle-labels',
        f'{args.test_folder}/ellipse_circle-image', f'{args.test_folder}/ellipse_circle-labels', args.output_folder],

        ["chull", f'{args.train_folder}/chull-image', f'{args.train_folder}/chull-labels',
        f'{args.test_folder}/chull-image', f'{args.test_folder}/chull-labels', args.output_folder],

        ["raw", f'{args.train_folder}/raw-image', f'{args.train_folder}/raw-labels',
        f'{args.test_folder}/raw-image', f'{args.test_folder}/raw-labels', args.output_folder],

        ["corner", f'{args.train_folder}/corner-image', f'{args.train_folder}/corner-labels',
        f'{args.test_folder}/corner-image', f'{args.test_folder}/corner-labels', args.output_folder]
    ]

    start_time = datetime.now()
    
    for batch in batches:
        train_cnn(batch)
    end_time = datetime.now()
    #print("transform names:", TransformNames)
    #print("accuracies:", accuracies)
    print("Completed training", len(batches), "transforms in:", end_time - start_time, flush=True)

    # python train_transforms.py -r ./transforms/mnist -e ./transforms/mnist-test -o models/mnist-svc-rbf

