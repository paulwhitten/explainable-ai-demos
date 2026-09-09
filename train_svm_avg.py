from load_mnist_data import load_mnist_float
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime
from sklearn.neural_network import MLPClassifier
import sklearn

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

print('The scikit-learn version is {}.'.format(sklearn.__version__))

home_folder = "/Users/pcw"

#linear_svc = svm.SVC(kernel='linear')
#polynomial
#sigmoid

# original
print("Loading training data")
N, rows, columns, training_digits, training_labels = load_mnist_float(
    home_folder + "/mnist/combined_image", home_folder + "/mnist/combined_label")
print("loaded combined training", N)

print("Loading test data.")
n, rows, columns, test_digits, test_labels = load_mnist_float(
    home_folder + "/mnist/download/t10k-images-idx3-ubyte", home_folder + "/mnist/download/t10k-labels-idx1-ubyte")
print("loaded test", n)

print("Fitting SVM")

tr_labels = np.argmax(training_labels, axis=-1)
start_time = datetime.now()
rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(training_digits, tr_labels)
end_time = datetime.now()
print("Fit SVM in:", end_time - start_time)

# test SVM
start_time = datetime.now()
pred = rbf_svc.predict(test_digits)
end_time = datetime.now()
print("Test SVM in:", end_time - start_time)

t_labels = np.argmax(test_labels, axis=-1)

print("SVM Radial Bias Function Accuracy:", accuracy_score(t_labels, pred))

analyze_results(t_labels, pred)

### Try with MLP

hidden_layers = (128, 128)
act_func = "relu"

print("Fitting MLP")
start_time = datetime.now()
mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=act_func, max_iter=500, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)
mlp.fit(training_digits, training_labels)
end_time = datetime.now()
print("Fit MLP in:", end_time - start_time)

#test MLP
start_time = datetime.now()
mlp_predict = mlp.predict(test_digits)
end_time = datetime.now()
print("Test MLP in:", end_time - start_time)
mlp_p = np.argmax(mlp_predict, axis=-1)
print("MLP Accuracy:", accuracy_score(t_labels, mlp_p))

analyze_results(t_labels, mlp_p)
