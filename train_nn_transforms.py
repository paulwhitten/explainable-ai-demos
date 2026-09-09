#!/usr/bin/env python

import sys
import json
import pickle
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, RocCurveDisplay
from load_mnist_data import load_mnist_float, get_num_classes, load_mnist_labels
from transform_parallel import Transform, TransformNames
#from joblib import dump, load # more efficient serialization

def analyze_results(batch, labels, predictions):
    print("\n################################", flush=True)
    print(batch, "accuracy analysis ", flush=True)
    num_classes = get_num_classes(labels)
    class_counts = [0.0] * num_classes
    class_correct = [0.0] * num_classes
    conf_matrix = []
    class_header = []
    for i in range(num_classes):
        class_header.append(i)
        conf_matrix.append([0] * num_classes)
    for pred_ix, p in enumerate(predictions):
        if labels[pred_ix] == p:
            class_correct[labels[pred_ix]] += 1.0
        class_counts[labels[pred_ix]] += 1.0
        conf_matrix[labels[pred_ix]][p] += 1
    # output results
    print("\n\nAccuracy per class:\n| class | accuracy |\n| :---: | :---: |", flush=True)
    for i in range(num_classes):
        print("|", i, " | ", class_correct[i] / class_counts[i], " |", flush=True)
    print("\n", flush=True)
    print("confusion matrix:", flush=True)
    conf_matrix_string = str(class_header) + "\n"
    for row_ix, row in enumerate(conf_matrix):
        conf_matrix_string += str(row_ix)
        for n in row:
            conf_matrix_string += ", " + str(n)
        conf_matrix_string += "\n"
    print(conf_matrix_string, flush=True)
    print("\n################################", flush=True)

"""
This program trains all of the transform models
"""

# Gets a PCA object fit to the passed data withe the passed number of components.
# Returns the PCA object and the transformed data
def do_pca(num_components, data, pca):
    print("Principal Components", num_components)
    if pca == None:
        pca = PCA(num_components)
    transformed_data = pca.fit_transform(data)
    return pca, transformed_data


def train_batch(batch, id):
    print("====", batch[0], "====", flush=True)
    batch_start_time = datetime.now()

    print("type:", type(batch), "length:", len(batch), "element 0:", batch[0], flush=True)

    N, train_rows, train_columns, train_images, train_labels = load_mnist_float(batch[1], batch[2])
    label_array = load_mnist_labels(batch[2])
    n, test_rows, test_columns, test_images, test_labels = load_mnist_float(batch[3], batch[4])

    if batch == "pca":
        # Load the pre-created PCA model from transforms
        import os
        pca_model_path = os.path.join(batch[5], "pca_transformer.pkl")  # batch[5] is output_folder
        if os.path.exists(pca_model_path):
            print(f"Loading pre-created PCA model from {pca_model_path}")
            with open(pca_model_path, 'rb') as f:
                pca = pickle.load(f)
            # Transform the data using the loaded model
            train_images = pca.transform(train_images)
            test_images = pca.transform(test_images)
        else:
            print(f"PCA model not found at {pca_model_path}, creating new one")
            pca, train_images = do_pca(80, train_images)
            pca, test_images = do_pca(80, test_images, pca)

    max_label = max(label_array)
    min_label = min(label_array)
    print("max_label:", max_label, "min_label:", min_label, flush=True)

    # create the mlp model
    hidden_layers = (128, 256, 128)
    act_func = "relu"
    start_time = datetime.now()
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=act_func, max_iter=500, alpha=1e-4,
                        solver='sgd', verbose=10, random_state=1,
                        learning_rate_init=.1)

    print("Fitting MLP", flush=True)
    print("Hidden layers:", hidden_layers, flush=True)
    mlp.fit(train_images, train_labels)
    end_time = datetime.now()
    print("Fit MLP in:", end_time - start_time, flush=True)

    #test MLP
    start_time = datetime.now()
    mlp_predict = mlp.predict(test_images)
    end_time = datetime.now()
    print("Test MLP in:", end_time - start_time, flush=True)
    mlp_p = np.argmax(mlp_predict, axis=-1)
    t_labels = np.argmax(test_labels, axis=-1)
    score = accuracy_score(t_labels, mlp_p)
    batch[6][id] = score
    print("MLP Accuracy:", score, flush=True)
    y_score = mlp.predict_proba(test_images)
    # aucs = []
    # for label in range(min_label, max_label):
    #     auc = roc_auc_score(test_labels[:, label], y_score[:, label], multi_class="ovr")
    #     aucs.append(auc)
    #     print("label:", label, "AUC:", auc)
    #     RocCurveDisplay.from_predictions(
    #         test_labels[:, label],
    #         y_score[:, label],
    #         name=f"{label} vs the rest",
    #         color="darkorange"
    #     )
    #     plt.axis("square")
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.title(batch[0] + " One-vs-Rest ROC curve:\n" + str(label) + " vs rest")
    #     plt.savefig(batch[5] + "/" + batch[0]+"_auc_" + str(label) + ".png")
    #     plt.close()
    # with open(batch[5] + "/" + batch[0]+ "_auc.json", "w") as outfile:
    #     json.dump(aucs, outfile)
    micro = {}
    micro["micro_averaged_auc"] = roc_auc_score(test_labels, y_score, multi_class="ovr", average="micro")
    with open(batch[5] + "/" + batch[0]+ "_auc_avg.json", "w") as outfile:
        json.dump(micro, outfile)
    analyze_results(batch[0], t_labels, mlp_p)

    # save https://scikit-learn.org/stable/model_persistence.html
    # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    pickle.dump(mlp, open(batch[5] + "/" + batch[0] + ".model", 'wb'))
    # open by running loaded_model = pickle.load(open(filename, 'rb'))

    batch_end_time = datetime.now()
    print(batch[0], "done in:", batch_end_time - batch_start_time, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trains transforms for a set')
    parser.add_argument('-r', '--train_folder', 
                        help='The training input folder')
    parser.add_argument('-e', '--test_folder', 
                        help='The test input folder')
    parser.add_argument('-o', '--output_folder', 
                        help='The folder to output')
    
    args = parser.parse_args()

    accuracies = [0.0] * 20  # Make room for all transforms

    batches = [

        ["crossing", f'{args.train_folder}/crossing-image', f'{args.train_folder}/crossing-labels',
        f'{args.test_folder}/crossing-image', f'{args.test_folder}/crossing-labels', args.output_folder, accuracies],

        ["endpoint", f'{args.train_folder}/endpoint-image', f'{args.train_folder}/endpoint-labels',
        f'{args.test_folder}/endpoint-image', f'{args.test_folder}/endpoint-labels', args.output_folder, accuracies],

        ["fill", f'{args.train_folder}/fill-image', f'{args.train_folder}/fill-labels',
        f'{args.test_folder}/fill-image', f'{args.test_folder}/fill-labels', args.output_folder, accuracies],

        ["skel-fill", f'{args.train_folder}/skel-fill-image', f'{args.train_folder}/skel-fill-labels',
        f'{args.test_folder}/skel-fill-image', f'{args.test_folder}/skel-fill-labels', args.output_folder, accuracies],

        ["skel", f'{args.train_folder}/skel-image', f'{args.train_folder}/skel-labels',
        f'{args.test_folder}/skel-image', f'{args.test_folder}/skel-labels', args.output_folder, accuracies],

        ["thresh", f'{args.train_folder}/thresh-image', f'{args.train_folder}/thresh-labels',
        f'{args.test_folder}/thresh-image', f'{args.test_folder}/thresh-labels', args.output_folder, accuracies],

        ["line", f'{args.train_folder}/line-image', f'{args.train_folder}/line-labels',
        f'{args.test_folder}/line-image', f'{args.test_folder}/line-labels', args.output_folder, accuracies],

        ["ellipse", f'{args.train_folder}/ellipse-image', f'{args.train_folder}/ellipse-labels',
        f'{args.test_folder}/ellipse-image', f'{args.test_folder}/ellipse-labels', args.output_folder, accuracies],

        ["circle", f'{args.train_folder}/circle-image', f'{args.train_folder}/circle-labels',
        f'{args.test_folder}/circle-image', f'{args.test_folder}/circle-labels', args.output_folder, accuracies],

        ["ellipse-circle", f'{args.train_folder}/ellipse_circle-image', f'{args.train_folder}/ellipse_circle-labels',
        f'{args.test_folder}/ellipse_circle-image', f'{args.test_folder}/ellipse_circle-labels', args.output_folder, accuracies],

        ["chull", f'{args.train_folder}/chull-image', f'{args.train_folder}/chull-labels',
        f'{args.test_folder}/chull-image', f'{args.test_folder}/chull-labels', args.output_folder, accuracies],

        ["raw", f'{args.train_folder}/raw-image', f'{args.train_folder}/raw-labels',
        f'{args.test_folder}/raw-image', f'{args.test_folder}/raw-labels', args.output_folder, accuracies],

        ["corner", f'{args.train_folder}/corner-image', f'{args.train_folder}/corner-labels',
        f'{args.test_folder}/corner-image', f'{args.test_folder}/corner-labels', args.output_folder, accuracies],

        ["pca", f'{args.train_folder}/raw-image', f'{args.train_folder}/raw-labels',
        f'{args.test_folder}/raw-image', f'{args.test_folder}/raw-labels', args.output_folder, accuracies]


    ]

    start_time = datetime.now()
    transform_id = 0
    for batch in batches:
        train_batch(batch, transform_id)
        transform_id += 1
    end_time = datetime.now()
    print("transform names:", TransformNames, flush=True)
    print("accuracies:", accuracies, flush=True)
    print("Completed training", len(batches), "transforms in:", end_time - start_time, flush=True)

    # python train_transforms.py -r ./transforms/mnist -e ./transforms/mnist-test -o models/mnist-svc-rbf