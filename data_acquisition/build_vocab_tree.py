from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive, pairs_from_retrieval, pairs_from_sequence
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

import os
from pathlib import Path
from sklearn.cluster import KMeans
import numpy as np
import h5py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class HierarchicalKMeans:
    def __init__(self, n_clusters=10, max_depth=5):
        """
        Initialize the Hierarchical KMeans clustering model.

        :param n_clusters: Number of clusters (branching factor) at each level.
        :param max_depth: Maximum depth of the tree.
        """
        self.n_clusters = n_clusters
        self.max_depth = max_depth
        self.tree = []

    def fit(self, X, depth=0):
        """
        Recursively fit the hierarchical k-means model to the data.

        :param X: The data to cluster (numpy array of shape [n_samples, n_features]).
        :param depth: Current depth in the hierarchy.
        """
        if depth < self.max_depth:
            print(f'Fitting hierarchical k-means at depth {depth} with {len(X)} samples')
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=self.n_clusters)
            kmeans.fit(X)

            # Store the cluster centers and labels
            self.tree.append({
                'depth': depth,
                'centers': kmeans.cluster_centers_,
                'labels': kmeans.labels_
            })

            # Recursively cluster each cluster's data
            for i in range(self.n_clusters):
                child_data = X[kmeans.labels_ == i]
                if len(child_data) > 1:
                    self.fit(child_data, depth + 1)
        else:
            # If the maximum depth is reached, stop recursion
            return

    def predict(self, X):
        """
        Predict the cluster assignments for each point in X using the hierarchical k-means model.

        :param X: The data to predict (numpy array of shape [n_samples, n_features]).
        :return: A list of lists containing the path through the tree for each point.
        """
        predictions = []
        for x in X:
            node = 0
            path = []
            while node < len(self.tree) and self.tree[node]['depth'] < self.max_depth:
                centers = self.tree[node]['centers']
                # Assign the point to the nearest cluster center
                cluster = np.argmin(np.linalg.norm(centers - x, axis=1))
                path.append(cluster)
                node = node * self.n_clusters + cluster + 1
            predictions.append(path)
        return predictions
    
    def build_histograms(self, images_descriptors):
        histograms = {}
        for key, descriptors in images_descriptors.items():
            visual_words = self.predict(descriptors)
            for word in visual_words:
                print(word)
            hist, _ = np.histogram(visual_words, bins=np.arange(len(hkm.tree[-1]['centers']) + 1))
            histograms[key] = hist
        return histograms
    
    def compute_tfidf(self, histograms):
        """
        Compute the TF-IDF encoding for the histograms.

        :param histograms: Array of histograms (bag of visual words for each image).
        :return: TF-IDF encoded array.
        """
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(list(histograms.values()))
        tfidf_dict = {}
        for i, key in enumerate(histograms.keys()):
            tfidf_dict[key] = np.array(tfidf[i].todense()).flatten()
        return tfidf_dict

# Example usage
if __name__ == "__main__":
    descriptors = {}

    with h5py.File('outputs/sfm/features.h5', 'r') as file:
        for key, val in file.items():
            descriptors[key] = np.array(val['descriptors']).reshape(-1, 256)
    
    # 32K Vocab Size
    hkm = HierarchicalKMeans(n_clusters=2, max_depth=5)
    all_descriptors = np.vstack(list(descriptors.values()))
    print(f"Descriptors shape: {all_descriptors.shape}")
    hkm.fit(all_descriptors)
    # hkm.tree = np.load('hkm.npy', allow_pickle=True)


    # Save the hierarchical k-means model
    # np.save('hkm.npy', hkm.tree)

    # Predict the cluster assignments for the descriptors

    histograms = hkm.build_histograms(descriptors)
    tfidf_encoded_images = hkm.compute_tfidf(histograms)

    # Compute the similarity
    similarities = {}
    for key, val in tfidf_encoded_images.items():
        similarities[key] = cosine_similarity(val, tfidf_encoded_images['keyframe_0245.png'])
    
    # Sort the images by similarity
    sorted_images = sorted(similarities, key=similarities.get, reverse=True)
    print(sorted_images)

