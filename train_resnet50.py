import argparse
from load_emnist_resnet_data import load_emnist_resnet
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

start_time = datetime.now()

parser = argparse.ArgumentParser(description='trains transforms for a set')
parser.add_argument('-t', '--train_folder', 
                    help='The training input folder')
parser.add_argument('-e', '--test_folder', 
                    help='The test input folder')
args = parser.parse_args()

# load training data
train_images, train_labels, num_train_labels = load_emnist_resnet(
    #"/home/pcw/mnist/download/train-images-idx3-ubyte", #mnist
    #"/home/pcw/mnist/download/train-labels-idx1-ubyte" #mnist
    #"transforms/emnist/train/raw-image",
    #"transforms/emnist/train/raw-labels"
    args.train_folder + "/images",
    args.train_folder + "/labels"
)

# load test data
test_images, test_labels, num_test_labels = load_emnist_resnet(
    #"/home/pcw/mnist/download/t10k-images-idx3-ubyte", #mnist
    #"/home/pcw/mnist/download/t10k-labels-idx1-ubyte" #mnist
    #"transforms/emnist/test/raw-image",
    #"transforms/emnist/test/raw-labels"
    args.test_folder + "/images",
    args.test_folder + "/labels"
)

print(train_images.shape, train_labels.shape, num_train_labels)

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
          metrics = tf.keras.metrics.CategoricalAccuracy(),
          optimizer = tf.keras.optimizers.Adam())

# fit 
history = resnet_model.fit(train_images, train_labels, batch_size=128, epochs=40, verbose=2)
print("History:")
for key in history.history.keys():
    print("Key:", key, "value:", history.history[key])

scores = resnet_model.evaluate(test_images, test_labels, verbose=1)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])
print(scores)
resnet_model.save("./models/resnet50_mnist.raw.model")

end_time = datetime.now()

print("done in:", end_time - start_time)
