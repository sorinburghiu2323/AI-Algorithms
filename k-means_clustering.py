from time import time
import numpy as np
from sklearn import metrics

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def bench_k_means(k_means, name, data, k_labels):
    """
    Benchmark to compare the different initialization methods for KMeans.
    :param k_means: KMeans instance.
    :param name: Name given to the strategy to show table results.
    :param data: Data to cluster.
    :param k_labels: Labels to compute clustering metrics.
    :return: None
    """
    initial_time = time()
    estimator = make_pipeline(StandardScaler(), k_means).fit(data)
    fit_time = time() - initial_time
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator labels.
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(k_labels, estimator[-1].labels_) for m in clustering_metrics]
    results += [metrics.silhouette_score(data, estimator[-1].labels_, metric="euclidean", sample_size=300)]

    # Print results adequately.
    formatter_result = "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    print(formatter_result.format(*results))


# Load the dataset.
digits, labels = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = digits.shape, np.unique(labels).size
print(f"Digits: {n_digits} \nSamples: {n_samples} \nFeatures {n_features}")

# Initialize column names.
print('\ninit\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

# Run the benchmark for 3 different approaches:
# 1. Initialization using k-means++
k_means = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(k_means=k_means, name="k-means++", data=digits, k_labels=labels)

# 2. Random initialization
k_means = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(k_means=k_means, name="random", data=digits, k_labels=labels)

# 3. PCA projection based initialization
pca = PCA(n_components=n_digits).fit(digits)
k_means = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
bench_k_means(k_means=k_means, name="PCA-based", data=digits, k_labels=labels)

# Fit data to plot.
reduced_data = PCA(n_components=2).fit_transform(digits)
k_means = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
k_means.fit(reduced_data)

h = 0.05  # Step size of the mesh. Decrease to increase the quality of the VQ.

# Plot the decision boundary.
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot.
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired, aspect="auto", origin="lower",
)

# Plot graph using matplotlib.
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
centroids = k_means.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
            color="w", zorder=10)
plt.title("Hand written digits K-means clustering\n(Centroids are marked with white cross)")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
