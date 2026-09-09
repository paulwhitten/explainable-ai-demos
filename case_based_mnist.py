import os
import sys
import json
import argparse
import pickle
import numpy as np
from enum import Enum
from sklearn import svm
from datetime import datetime
from scipy.spatial import distance
from io import BytesIO
import imageio
import base64
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# TensorFlow/Keras are imported lazily inside do_cnn/do_resnet so the serving
# path (CaseObject) never loads them.

MODEL_FOLDER = "cbe_models"
NN_MODEL_NAME = "nn.model"
TRAIN_LABEL_NAME = "train_labels.json"
TRAIN_DATA_NAME = "train_data.json"
TRAIN_DATA_NPY_NAME = "train_data.npy"

class CaseObject:
    
    def __init__(self, folder):
        self.nn = pickle.load(open(os.path.join(folder, NN_MODEL_NAME), 'rb'))
        with open(os.path.join(folder, TRAIN_LABEL_NAME)) as f:
            self.train_labels = json.load(f)
        npy_path = os.path.join(folder, TRAIN_DATA_NPY_NAME)
        if os.path.exists(npy_path):
            # memory-mapped float32 array avoids the ~1.5 GB Python-list blowup
            self.train_data = np.load(npy_path, mmap_mode='r')
        else:
            with open(os.path.join(folder, TRAIN_DATA_NAME)) as f:
                self.train_data = np.asarray(json.load(f), dtype=np.float32)


    def get_prediction(self, img):
        img = (img/255.0).astype(float)
        pred = self.nn.predict([img])
        pred = np.argmax(pred, axis=-1)
        return int(pred[0])


    def get_neighbors(self, img, k):
        img = (img/255.0).astype(float)
        distances = []
        for ix, n in enumerate(self.train_data):
            dist = {}
            dist["distance"] = distance.euclidean(img, n)
            dist["index"] = ix
            dist["label"] = int(self.train_labels[ix])
            distances.append(dist)
        distances.sort(key=get_distance)
        neighbors = []
        for i in range(k):
            n = distances[i]
            buf = BytesIO()
            img = (np.array(self.train_data[n["index"]]) * 255).astype(np.uint8)
            imageio.imwrite(buf, img.reshape((28, 28)), format='png')
            b64_img = base64.b64encode(buf.getbuffer()).decode('utf-8')
            n["b64_image"] = b64_img
            neighbors.append(n)
        weights = [0.0] * 10
        for n in neighbors:
            weights[n["label"]] += 1.0 / (n["distance"] * n["distance"])
        total_weights = sum(weights)
        correspondence = []
        for w in weights:
            correspondence.append(w/total_weights)
        return neighbors, correspondence
    

def create_dir(path):
    if os.path.isdir(path):
        print(f"Directory '{path}' exists.")
        return False
    else:
        os.makedirs(path, exist_ok=True)
        print(f"Created directory '{path}'.")
        return True


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def get_distance(d):
    return d["distance"]


def get_num_classes(preds):
    max_class = max(preds)
    min_class = min(preds)
    return max_class - min_class + 1


def analyze_results(labels, predictions):
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
    print("\n\nAccuracy per class:\n| class | accuracy |\n| :---: | :---: |")
    for i in range(num_classes):
        print("|", i, " | ", class_correct[i] / class_counts[i], " |")
    print("\n")
    print("confusion matrix:")
    conf_matrix_string = str(class_header) + "\n"
    for row_ix, row in enumerate(conf_matrix):
        conf_matrix_string += str(row_ix)
        for n in row:
            conf_matrix_string += ", " + str(n)
        conf_matrix_string += "\n"
    print(conf_matrix_string)


class InferenceEngine(Enum):
    svm = 'svm'
    nn = 'nn'
    cnn = 'cnn'
    resnet = 'resnet'

    def __str__(self):
        return self.value
    

def do_svm(train_images, train_labels, test_images, test_labels):
    tr_labels = np.argmax(train_labels, axis=-1)
    start_time = datetime.now()
    rbf_svc = svm.SVC(kernel='rbf')
    rbf_svc.fit(train_images, tr_labels)
    end_time = datetime.now()
    print("Fit SVM in:", end_time - start_time)
    # process the test dataset
    start_time = datetime.now()
    pred = rbf_svc.predict(test_images)
    end_time = datetime.now()
    print("Test SVM in:", end_time - start_time)
    return pred


def do_cnn(train_images, train_labels, test_images, test_labels):
    from tensorflow import keras
    from keras import layers
    batch_size = 128
    num_epochs = 40
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)
    num_classes = len(train_labels[0])
    print("number of classes:", num_classes)
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
    #cat_train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
    history = CNN.fit(train_images, train_labels, batch_size=batch_size, epochs=num_epochs, verbose=2)
    #cat_test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)
    scores = CNN.evaluate(test_images, test_labels, verbose=1)
    pred = CNN.predict(test_images)
    print("Test loss:", scores[0])
    print("Test accuracy:", scores[1])
    print(scores)
    pred = np.argmax(pred, axis=-1)
    return pred


def do_nn(train_images, train_labels, test_images, test_labels):
    # create the mlp model
    hidden_layers = (128, 256, 128)
    act_func = "relu"
    tr_labels = np.argmax(train_labels, axis=-1)
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
    pred = mlp.predict(test_images)
    end_time = datetime.now()
    print("Test MLP in:", end_time - start_time, flush=True)
    pred = np.argmax(pred, axis=-1)
    pickle.dump(mlp, open(os.path.join(MODEL_FOLDER, NN_MODEL_NAME), 'wb'))
    return pred


def do_resnet(train_images, train_labels, test_images, test_labels):
    import tensorflow as tf
    from load_emnist_resnet_data import process_labels_for_resnet, process_mnist_images_for_resnet
    train_images = np.array(train_images*255.0).astype("uint8")
    train_images = process_mnist_images_for_resnet(train_images)
    train_labels = np.argmax(train_labels, axis=-1)
    train_labels, num_train_labels = process_labels_for_resnet(train_labels)
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
    print("History:")
    for key in history.history.keys():
        print("Key:", key, "value:", history.history[key])
    test_images = np.array(test_images*255.0).astype("uint8")
    test_images = process_mnist_images_for_resnet(test_images)
    scores = resnet_model.evaluate(test_images, test_labels, verbose=1)
    pred = resnet_model.predict(test_images)
    pred = np.argmax(pred, axis=-1)
    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs the case-based method on MNIST')
    parser.add_argument('-m', '--mnist_folder', 
                        help='The path to mnist')
    parser.add_argument('-n', '--neighbor_folder', 
                        help='The path to already stored neighbor data')
    parser.add_argument('-o', '--output', 
                        help='The path to output')
    parser.add_argument('-t', '--type', 
                        help='The type of inference engine', type=InferenceEngine, choices=list(InferenceEngine))
    args = parser.parse_args()

    from load_mnist_data import load_mnist_float

    start = datetime.now()
    train_N, rows, columns, train_images, train_labels = load_mnist_float(
        os.path.join(args.mnist_folder, "train-images-idx3-ubyte"),
        os.path.join(args.mnist_folder, "train-labels-idx1-ubyte")
    )

    test_N, test_columns, test_rows, test_images, test_labels = load_mnist_float(
        os.path.join(args.mnist_folder, "t10k-images-idx3-ubyte"),
        os.path.join(args.mnist_folder, "t10k-labels-idx1-ubyte")
    )

    end = datetime.now()
    print("Elapsed time to load:", end - start)

    print("N:", train_N)
    print("Images length:", len(train_images))
    print("Labels length:", len(train_labels))

    create_dir(MODEL_FOLDER)

    if args.type == InferenceEngine.svm:
        pred= do_svm(train_images, train_labels, test_images, test_labels)
    elif args.type == InferenceEngine.cnn:
        pred = do_cnn(train_images, train_labels, test_images, test_labels)
    elif args.type == InferenceEngine.nn:
        pred = do_nn(train_images, train_labels, test_images, test_labels)
    elif args.type == InferenceEngine.resnet:
        pred = do_resnet(train_images, train_labels, test_images, test_labels)
    else:
        print("Valid inference type not provided.")
        sys.exit(1)

    t_labels = np.argmax(test_labels, axis=-1)
    tr_labels = np.argmax(train_labels, axis=-1)

    # output results
    analyze_results(t_labels, pred)
    print("Accuracy:", accuracy_score(t_labels, pred))

    # save data
    with open(os.path.join(args.output, "test_predictions.json"), 'w') as outfile:
        json.dump(pred.tolist(), outfile)

    with open(os.path.join(args.output, "test_labels.json"), 'w') as outfile:
        json.dump(t_labels.tolist(), outfile)

    with open(os.path.join(args.output, TRAIN_LABEL_NAME), 'w') as outfile:
        json.dump(tr_labels.tolist(), outfile)

    with open(os.path.join(args.output, TRAIN_DATA_NAME), 'w') as outfile:
        json.dump(train_images, outfile)

    # also emit a compact float32 array the serving path can memory-map
    np.save(os.path.join(args.output, TRAIN_DATA_NPY_NAME), np.asarray(train_images, dtype=np.float32))

    NEIGHBOR_COUNT = 25
    loaded_neighbors = False
    neighbor_file_path = os.path.join(args.neighbor_folder, "neighbors.json")
    if os.path.exists(neighbor_file_path):
        with open(neighbor_file_path, 'r') as outfile:
            training_neighbors = json.load(outfile)
            loaded_neighbors = True
    if not loaded_neighbors:
        start_time = datetime.now()
        training_neighbors = []
        for test_ix, test_label in enumerate(test_labels):
            distances = []
            for ix, d in enumerate(train_images):
                dist = {}
                dist["distance"] = distance.euclidean(test_images[test_ix], d)
                dist["index"] = ix
                dist["label"] = int(tr_labels[ix])
                distances.append(dist)
            distances.sort(key=get_distance)
            # slice does not seem to work here
            #distances.append(dist[:NEIGHBOR_COUNT])
            neigh = []
            for i in range(NEIGHBOR_COUNT+1):
                neigh.append(distances[i])
            #print("test label:", t_lab
            #for ix in range(10):
            #    print("distance:", distances[ix], "tr_label:", tr_labels[distances[ix]["index"]])
            training_neighbors.append(neigh)
            if test_ix % 100 == 0:
                print("test index:", test_ix)
        end_time = datetime.now()
        print("Found NN in :", end_time - start_time)
        with open(os.path.join(args.neighbor_folder, "neighbors.json"), 'w') as outfile:
            json.dump(training_neighbors, outfile)

    NEIGHBOR_COUNT = 10
    correct = 0
    corr_count = 0
    corr_total = 0.0
    count = 0
    case_based = []
    start_time = datetime.now()
    for test_ix, test_label in enumerate(test_labels):
        weights = [0.0] * 10
        neighbors = []
        for neighbor_ix in range(NEIGHBOR_COUNT+1):
            distance = training_neighbors[test_ix][neighbor_ix]["distance"]
            # TODO: divide by zero
            weights[tr_labels[training_neighbors[test_ix][neighbor_ix]["index"]]] += 1.0 / (distance * distance)
            neighbors.append(training_neighbors[test_ix][neighbor_ix])
        data = {}
        data["index"] = test_ix
        data["prediction"] = pred[test_ix]
        data["label"] = t_labels[test_ix]
        data["weights"] = weights
        suggested_neighbor = np.argmax(weights)
        data["contradiction"] = suggested_neighbor != t_labels[test_ix]
        
        weight_sum = sum(weights)
        correspondence = []
        for w_ix, w in enumerate(weights):
            correspondence.append(w / weight_sum)
        max_corr = max(correspondence)
        max_index = correspondence.index(max_corr)
        corr_string = "Contradicts"
        if max_index == data["prediction"]:
            corr_count += 1
            corr_total += max_corr
            corr_string = "Corresponds"
        data["pred_comp_to_neighbors"] = corr_string
        data["correspondence"] = correspondence
        data["neighbors"] = neighbors
        if data["prediction"] == data["label"]:
            correct += 1
        case_based.append(data)
        count += 1
    end_time = datetime.now()
    print("Completed weight and correspondence calculation :", end_time - start_time)

    print("Count: ", count, "Correct:", correct, "Correspondence count:", corr_count)
    print("Correct rate:", correct / count, "Correspondence count:", corr_count / count )
    print("Average correspondence: ", corr_total / corr_count)

    with open(os.path.join(args.output, "case-based_results.json"), 'w') as outfile:
        json.dump(case_based, outfile, indent=4, default=np_encoder)
