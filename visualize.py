import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import sys
import matplotlib.pyplot as plt
from S3VM import S3VM

def tsne(data, n_components, perplexity=30.0, verbose=True):
    """
    tsne : Embed and plot given 10-dimensional data into xy or xyz coordinates

    :param data:
    ndarray, (N, dim)
    :param perplexity:
    float, hyper-parameter for std of t-SNE
    :param n_components:
    int, embedding dimension of t-SNE

    :return:
    None
    """
    data_embedded = TSNE(n_components=n_components, perplexity=perplexity, metric='euclidean').fit_transform(data)

    fig = plt.figure(figsize=(10, 5))
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_embedded[:,0], data_embedded[:,1], data_embedded[:,2])
    elif n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(data_embedded[:, 0], data_embedded[:, 1])

    plt.savefig('./tsne_raw.png')
    plt.title("T-SNE plot of given 10-dimensional matrix")
    if verbose:
        plt.show()
    return data_embedded


def draw_embedded_data_and_boundary(X, svm, n_dim, y=None, perplexity=30.0):
    fig = plt.figure(figsize=(30, 13))

    X_embedded = TSNE(n_components=n_dim, perplexity=perplexity, metric='euclidean').fit_transform(X)
    X_cluster_1 = X_embedded[svm.cluster1, :]
    X_cluster_2 = X_embedded[svm.cluster2, :]
    y_predicted = np.squeeze(svm.decision_function(X))

    if n_dim == 3:
        if y is not None:
            ax, ax2 = fig.add_subplot(121, projection='3d'), fig.add_subplot(121, projection='3d')
            ax2.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=y, cmap=plt.cm.Paired, s=200, lw=1)
        else:
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=y_predicted, cmap=plt.cm.Paired, s=200, lw=1)
        ax.scatter(X_cluster_1[:, 0], X_cluster_1[:, 1], X_cluster_1[:, 2], s=500, lw=1, alpha=0.2)
        ax.scatter(X_cluster_2[:, 0], X_cluster_2[:, 1], X_cluster_2[:, 2], s=500, lw=1, alpha=0.2)
    elif n_dim == 2:
        if y is not None:
            ax, ax2 = fig.add_subplot(121), fig.add_subplot(122)
            ax2.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=plt.cm.Paired, s=200, lw=1)
        else:
            ax = fig.add_subplot(111)
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_predicted, cmap=plt.cm.Paired, s=200, lw=1)
        ax.scatter(X_cluster_1[:, 0], X_cluster_1[:, 1], s=500, lw=1, alpha=0.2)
        ax.scatter(X_cluster_2[:, 0], X_cluster_2[:, 1], s=500, lw=1, alpha=0.2)

    plt.savefig('./tsne_s3vm.png')
    plt.show()

    return

