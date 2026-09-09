import sklearn
import numpy as np
import skimage
from sklearn import metrics
import matplotlib.pyplot as plt
from skimage import io
from datetime import datetime
from load_mnist_data import load_mnist_float, load_mnist_labels
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.cluster import KMeans, AffinityPropagation


NUM_CLUSTERS = 5


def convert_array_to_image(pixels, rows, columns):
    img = np.zeros((rows, columns), np.uint8)
    pixel_count = 0
    for y in range(columns):
        for x in range(rows):
            img[y, x] = pixels[pixel_count] * 255
            pixel_count += 1
    return img

def add_image(cl, index, image):
    for ix, px in enumerate(image):
        cl[index][ix] += px

N, train_rows, train_columns, train_images, train_labels = load_mnist_float("/home/pcw/mnist/download/train-images-idx3-ubyte", "/home/pcw/mnist/download/train-labels-idx1-ubyte")
label_array = load_mnist_labels("/home/pcw/mnist/download/train-labels-idx1-ubyte")

# N, train_rows, train_columns, train_images, train_labels = load_mnist_float("/home/pcw/mnist/train-transform/skel-image", "/home/pcw/mnist/train-transform/skel-labels")
# label_array = load_mnist_labels("/home/pcw/mnist/download/train-labels-idx1-ubyte")

#km = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(train_images)

#print(km.labels_.tolist())

#print(label_array)

# for num in range(10):
#     images = []
#     for i, img in  enumerate(train_images):
#         if label_array[i] == num:
#             images.append(img)
#     sse = []
#     r = range(1,11)
#     for i in r:
#         km = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(images)
#         sse.append(km.inertia_)
#
#     plt.figure(figsize=(6, 6))
#     plt.plot(r, sse, '-o')
#     plt.xlabel(r'Number of clusters *k* for ' + str(num))
#     plt.ylabel('Sum of squared distance')
#     plt.show()


# # print graphs
# for num in range(10):
#     images = []
#     for i, img in  enumerate(train_images):
#         if label_array[i] == num:
#             images.append(img)
#     sse = []
#     r = range(1,36)
#     for i in r:
#         km = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(images)
#         sse.append(km.inertia_)

#     plt.figure(figsize = [13, 5])
#     plt.subplot(1, 2, 1)
#     plt.plot(r, sse, '-o')
#     plt.xlabel("Number of clusters $k$ for the digit " + str(num))
#     plt.ylabel('Sum of squared distance')
#     plt.grid(True)
#     plt.subplot(1, 2, 2)
#     plt.plot(r[:-1], np.diff(sse), "-o")
#     plt.xlabel("Number of clusters $k$")
#     plt.ylabel("Change in inertia")
#     plt.grid(True)
#     #plt.show()
#     plt.savefig("cluster/cluster-plot-" + str(num) + ".png")
#     plt.close()



# images = []
# for i, img in  enumerate(train_images):
#     if label_array[i] == 9:
#         images.append(img)
# km = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, n_init="auto").fit(images)

# counts = [0] * NUM_CLUSTERS

# clusters = []
# for j in range(NUM_CLUSTERS):
#     clusters.append([0.0] * 784)

# for i, img in enumerate(images):
#     #l = convert_array_to_image(img, 28, 28)
#     #io.imsave('cluster/' + str(km.labels_[i]) + "_" + str(counts[km.labels_[i]]) + ".png", l, check_contrast=False)
#     counts[km.labels_[i]] += 1
#     add_image(clusters, km.labels_[i], img)

# for i, c in enumerate(clusters):
#     res = [x / counts[i] for x in c]
#     #print(res)
#     #print(c)
#     img = convert_array_to_image(res, 28, 28)
#     io.imsave('cluster/out-' + str(i) + ".png", img, check_contrast=False)
    
    

# # get aggregate images for the clusters
# images = []
# for i, img in  enumerate(train_images):
#     if label_array[i] == 1:
#         images.append(img)
# km = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, n_init="auto").fit(images)

# counts = [0] * NUM_CLUSTERS

# clusters = []
# for j in range(NUM_CLUSTERS):
#     clusters.append([0.0] * 784)

# for i, img in enumerate(images):
#     #l = convert_array_to_image(img, 28, 28)
#     #io.imsave('cluster/' + str(km.labels_[i]) + "_" + str(counts[km.labels_[i]]) + ".png", l, check_contrast=False)
#     counts[km.labels_[i]] += 1
#     add_image(clusters, km.labels_[i], img)

# for i, c in enumerate(clusters):
#     res = [x / counts[i] for x in c]
#     #print(res)
#     #print(c)
#     img = convert_array_to_image(res, 28, 28)
#     io.imsave('cluster/out-' + str(i) + ".png", img, check_contrast=False)




# images = []
# true_labels = []
# for i, img in  enumerate(train_images):
#     if label_array[i] == 1:
#         images.append(img)
#         true_labels.append(label_array[i])
af = AffinityPropagation(preference=-50, random_state=0).fit(train_images)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print("Estimated number of clusters: %d" % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(label_array, labels))
print("Completeness: %0.3f" % metrics.completeness_score(label_array, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(label_array, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(label_array, labels))
print(
    "Adjusted Mutual Information: %0.3f"
    % metrics.adjusted_mutual_info_score(label_array, labels)
)
print(
    "Silhouette Coefficient: %0.3f"
    % metrics.silhouette_score(label_array, labels, metric="sqeuclidean")
)