import pickle
import argparse
from multiprocessing import Process, Pool
from datetime import datetime
import numpy as np
#from sklearn.metrics import confusion_matrix
from load_mnist_data import load_mnist_float, load_mnist_labels
import json

"""
This program reprocesses all of the transform data to build a knowledgebase in a
parallel fashion. In doing so, we read all transforms and load models,
outputting all predictions for the transforms based on training and then test
images. Data is output in json format to files for each of the datasets along
with json containing the labels.
"""

def eval_images(batch):
    print("====", batch[0], "====")
    batch_start_time = datetime.now()
    # load the image data to process
    n, rows, columns, train_images, train_labels = load_mnist_float(batch[1], batch[2])
    test_n, test_rows, test_columns, test_images, test_labels = load_mnist_float(batch[3], batch[4])
    train_label_list = load_mnist_labels(batch[2])
    test_label_list = load_mnist_labels(batch[4])
    train_label_max = np.max(train_label_list)
    train_label_min = np.min(train_label_list)
    print("label min:", train_label_min, "label max:", train_label_max)
    # create the model

    model = pickle.load(open(batch[6] + "/" + batch[0] + ".model", 'rb')) #keras.models.load_model(batch[6] + "/" + batch[0] + ".model")

    # get the predictions and probabilities
    train_predictions = model.predict(train_images)
    train_proba = model.predict_proba(train_images)
    test_predictions = model.predict(test_images)
    test_proba = model.predict_proba(test_images)

    # convert the one-hot to numeric label encoding
    if len(train_predictions.shape) > 1 and train_predictions.shape[1] > 1:
        train_predictions = np.argmax(train_predictions, axis=-1)
    if len(test_predictions.shape) > 1 and test_predictions.shape[1] > 1:
        test_predictions = np.argmax(test_predictions, axis=-1)
    print("train predictions shape:", train_predictions.shape)
    print("test predictions shape:", test_predictions.shape)

    # train_matrix = confusion_matrix(train_label_list, train_predictions)
    # test_matrix = confusion_matrix(test_label_list, test_predictions)

    with open(batch[5] + "/" + batch[0] + ".train.json", 'w') as outfile:
        json.dump(train_predictions.tolist(), outfile)
    with open(batch[5] + "/" + batch[0] + ".test.json", 'w') as outfile:
        json.dump(test_predictions.tolist(), outfile)
    with open(batch[5] + "/" + batch[0] + ".train-labels.json", 'w') as outfile:
        json.dump(train_label_list, outfile)
    with open(batch[5] + "/" + batch[0] + ".test-labels.json", 'w') as outfile:
        json.dump(test_label_list, outfile)

    with open(batch[5] + "/" + batch[0] + ".train-proba.json", 'w') as outfile:
        json.dump(train_proba.tolist(), outfile)
    with open(batch[5] + "/" + batch[0] + ".test-proba.json", 'w') as outfile:
        json.dump(test_proba.tolist(), outfile)

    batch_end_time = datetime.now()
    print(batch[0], "done in:", batch_end_time - batch_start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='save the probabilities to a knowledgebase')
    parser.add_argument('-r', '--train_folder', 
                        help='The training input folder')
    parser.add_argument('-e', '--test_folder', 
                        help='The test input folder')
    parser.add_argument('-o', '--output_folder', 
                        help='The folder to output')
    parser.add_argument('-m', '--model_folder', 
                        help='The folder containing the models')
    args = parser.parse_args()

    batches = [
        ["raw", f'{args.train_folder}/raw-image', f'{args.train_folder}/raw-labels',
        f'{args.test_folder}/raw-image', f'{args.test_folder}/raw-labels', args.output_folder, args.model_folder],

        ["crossing", f'{args.train_folder}/crossing-image', f'{args.train_folder}/crossing-labels',
        f'{args.test_folder}/crossing-image', f'{args.test_folder}/crossing-labels', args.output_folder, args.model_folder],

        ["endpoint", f'{args.train_folder}/endpoint-image', f'{args.train_folder}/endpoint-labels',
        f'{args.test_folder}/endpoint-image', f'{args.test_folder}/endpoint-labels', args.output_folder, args.model_folder],

        ["fill", f'{args.train_folder}/fill-image', f'{args.train_folder}/fill-labels',
        f'{args.test_folder}/fill-image', f'{args.test_folder}/fill-labels', args.output_folder, args.model_folder],

        ["skel-fill", f'{args.train_folder}/skel-fill-image', f'{args.train_folder}/skel-fill-labels',
        f'{args.test_folder}/skel-fill-image', f'{args.test_folder}/skel-fill-labels', args.output_folder, args.model_folder],

        ["skel", f'{args.train_folder}/skel-image', f'{args.train_folder}/skel-labels',
        f'{args.test_folder}/skel-image', f'{args.test_folder}/skel-labels', args.output_folder, args.model_folder],

        ["thresh", f'{args.train_folder}/thresh-image', f'{args.train_folder}/thresh-labels',
        f'{args.test_folder}/thresh-image', f'{args.test_folder}/thresh-labels', args.output_folder, args.model_folder],

        ["line", f'{args.train_folder}/line-image', f'{args.train_folder}/line-labels',
        f'{args.test_folder}/line-image', f'{args.test_folder}/line-labels', args.output_folder, args.model_folder],

        ["ellipse", f'{args.train_folder}/ellipse-image', f'{args.train_folder}/ellipse-labels',
        f'{args.test_folder}/ellipse-image', f'{args.test_folder}/ellipse-labels', args.output_folder, args.model_folder],

        ["circle", f'{args.train_folder}/circle-image', f'{args.train_folder}/circle-labels',
        f'{args.test_folder}/circle-image', f'{args.test_folder}/circle-labels', args.output_folder, args.model_folder],

        ["ellipse-circle", f'{args.train_folder}/ellipse_circle-image', f'{args.train_folder}/ellipse_circle-labels',
        f'{args.test_folder}/ellipse_circle-image', f'{args.test_folder}/ellipse_circle-labels', args.output_folder, args.model_folder],

        ["chull", f'{args.train_folder}/chull-image', f'{args.train_folder}/chull-labels',
        f'{args.test_folder}/chull-image', f'{args.test_folder}/chull-labels', args.output_folder, args.model_folder],

        ["corner", f'{args.train_folder}/corner-image', f'{args.train_folder}/corner-labels',
        f'{args.test_folder}/corner-image', f'{args.test_folder}/corner-labels', args.output_folder, args.model_folder]
    ]

    start_time = datetime.now()

    #processes = []
    #for batch in batches:
    #    eval_images(batch)
    with Pool(10) as p:
        p.map(eval_images, batches)

    end_time = datetime.now()
    print("Program done in:", end_time - start_time)
        
