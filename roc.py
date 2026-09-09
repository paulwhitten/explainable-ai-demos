from sklearn.preprocessing import LabelBinarizer
from load_mnist_data import load_mnist_float
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import json
import pickle

#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

t = open("./kb_svm_mnist/corner.train-labels.json")
train_labels = json.load(t)
p = open("./kb_svm_mnist/corner.test.json")
pred = json.load(p)
p.close()
l = open("./kb_svm_mnist/corner.test-labels.json")
labels = json.load(l)
l.close()

#print(pred)
#print(labels)

label_binarizer = LabelBinarizer().fit(train_labels)
y_onehot_test = label_binarizer.transform(labels)
print(y_onehot_test.shape)  # (n_samples, n_classes)

#print(y_onehot_test)

class_id = 1
class_of_interest = "one"

model = pickle.load(open("./models/mnist_svm/corner.model", 'rb'))
n, rows, columns, train_images, train_labels = load_mnist_float("./transforms_mnist_test/corner-image", "./transforms_mnist_test/corner-labels")
y_score = model.predict_proba(train_images)
#print("Y_SCORE:", y_score)
#print("Y_SCORE shape:", y_score.shape)

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    y_score[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange"
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\n1 vs [0, 2-9]")
plt.legend()
plt.show()


RocCurveDisplay.from_predictions(
    y_onehot_test.ravel(),
    y_score.ravel(),
    name="micro-average OvR",
    color="darkorange"
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Micro-averaged One-vs-Rest\nReceiver Operating Characteristic")
plt.legend()
plt.show()