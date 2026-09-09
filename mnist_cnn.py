from load_mnist_data import load_mnist_float_images, load_mnist
from datetime import datetime
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

batch_size = 128
num_epochs = 40

start_time = datetime.now()

# load training data
train_n, train_rows, train_columns, train_images, train_labels = load_mnist_float_images(
    "/home/pcw/mnist/download/train-images-idx3-ubyte",
    "/home/pcw/mnist/download/train-labels-idx1-ubyte"
)
N, rows, columns, w_digits, w_labels = load_mnist(
    "/home/pcw/mnist/download/train-images-idx3-ubyte",
    "/home/pcw/mnist/download/train-labels-idx1-ubyte"
)
#train_images, train_labels, num_train_labels = load_emnist_resnet(
#    "/home/pcw/mnist/download/train-images-idx3-ubyte",
#    "/home/pcw/mnist/download/train-labels-idx1-ubyte"
#)

test_n, test_rows, test_columns, test_images, test_labels = load_mnist_float_images(
    "/home/pcw/mnist/download/t10k-images-idx3-ubyte",
    "/home/pcw/mnist/download/t10k-labels-idx1-ubyte"
)
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

print("History:")
for key in history.history.keys():
    print("Key:", key, "value:", history.history[key])

cat_test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)
scores = CNN.evaluate(test_images, cat_test_labels, verbose=1)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])
print(scores)
CNN.save("./models/cnn_mnist.raw.model")

end_time = datetime.now()

print("done in:", end_time - start_time)
