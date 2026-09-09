#!/usr/bin/env python

import argparse
from load_emnist_resnet_data import load_emnist_resnet
#from plot_history import plot_model_history
from datetime import datetime
from tensorflow import keras
import tensorflow as tf

"""
This program trains all of the transform resnet 50 NNAs
"""

def train_batch(batch):
    print("====", batch[0], "====", flush=True)
    batch_start_time = datetime.now()
    keras.backend.clear_session()
    train_images, train_labels, num_train_labels = load_emnist_resnet(batch[1], batch[2])
    test_images, test_labels, num_test_labels = load_emnist_resnet(batch[3], batch[4])
    # create the model
    input = tf.keras.Input(shape=(32, 32, 3))
    efnet = tf.keras.applications.ResNet50(weights=None, include_top=False, input_tensor=input)

    # Now that we apply global max pooling.
    gap = tf.keras.layers.GlobalMaxPooling2D()(efnet.output)

    # Finally, we add a classification layer.
    output = tf.keras.layers.Dense(num_train_labels, activation='softmax', use_bias=True)(gap)

    # tie together
    resnet_model = tf.keras.Model(efnet.input, output)

    resnet_model.compile(
          loss  = tf.keras.losses.CategoricalCrossentropy(),
          metrics = [tf.keras.metrics.CategoricalAccuracy()],
          optimizer = tf.keras.optimizers.Adam())

    # fit 
    history = resnet_model.fit(train_images, train_labels, batch_size=128, epochs=40, verbose=2)

    scores = resnet_model.evaluate(test_images, test_labels, verbose=2)
    print("Test loss:", scores[0], flush=True)
    print("Test accuracy:", scores[1], flush=True)
    print(scores, flush=True)
    resnet_model.save(batch[5] + "/" + batch[0] + ".keras")
    #plot_model_history(history, "categorical_accuracy", batch[0])
    batch_end_time = datetime.now()
    print(batch[0], "done in:", batch_end_time - batch_start_time, flush=True)
    # clean up
    del resnet_model
    del history
    del train_images
    del train_labels
    del test_images
    del test_labels
    del input
    del efnet
    del output
    del gap

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
        train_batch(batch)
    end_time = datetime.now()
    print("Completed training", len(batches), "transforms in:", end_time - start_time, flush=True)