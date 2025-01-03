"""
----------------------------------------------------------------------------------------------------------------------------------------
Description:                                                                                                                            |
    This module is used for creating graphs suitable for scParaLaG models.                                                              |
                                                                                                                                        |
Copyright:                                                                                                                              |
    Copyright © 2024. All rights reserved.                                                                                              |                                                                            
                                                                                                                                        |
License:                                                                                                                                |
    This script is licensed under the MIT License.                                                                                      |
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software                                       |
    and associated documentation files (the "Software"), to deal in the Software without restriction,                                   |
    including without limitation the rights to use, copy, modify, merge, publish, distribute,                                           |
    sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,                   |
    subject to the following conditions:                                                                                                |
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.      |
                                                                                                                                        |
Disclaimer:                                                                                                                             |
    This software is provided 'as is' and without any express or implied warranties.                                                    |
    The author or the copyright holders make no representations about the suitability of this software for any purpose.                 |
                                                                                                                                        |
Contact:                                                                                                                                |
    For any queries or issues related to this script, please contact fchumeh@gmail.com.                                                 |
----------------------------------------------------------------------------------------------------------------------------------------
"""

import dgl
import torch
import scipy.stats
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from dance.data.base import Data
import numpy as np

class GraphCreator:
    def __init__(self, preprocess_type, n_neighbors=20, n_components=1200,
             metric='euclidean', weight_type='gaussian', sigma=1.0):
        """
        Initialize the GraphCreator with weighted edge computation.

        Parameters
        ----------
        preprocess_type : str
            Type of preprocessing to apply. Options: 'None', 'SVD'
        n_neighbors : int, optional
            Number of nearest neighbors for graph construction. Default is 20.
        n_components : int, optional
            Number of components if SVD preprocessing is used. Default is 1200.
        metric : str, optional
            Distance metric for KNN computation. Default is 'euclidean'.
        weight_type : str, optional
            Type of edge weight computation. Options: 'gaussian', 'cosine'. Default is 'gaussian'.
        sigma : float, optional
            Bandwidth parameter for Gaussian kernel. Default is 1.0.
        """
        self.preprocess_type = preprocess_type
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.weight_type = weight_type
        self.sigma = sigma

    def _compute_edge_weights(self, distances):
        """
        Transform distances to edge weights using specified weighting scheme.

        Parameters
        ----------
        distances : np.ndarray
            Array of pairwise distances between nodes.

        Returns
        -------
        np.ndarray
            Transformed edge weights. For Gaussian kernel: exp(-d²/2σ²),
            For cosine: 1-d, For others: raw distances.
        """
        if self.weight_type == 'gaussian':
            return np.exp(-distances**2 / (2 * self.sigma**2))
        elif self.weight_type == 'cosine':
            return 1 - distances
        return distances

    def _build_knn_graph(self, features):
        """
        Construct a weighted k-nearest neighbors graph from feature vectors.

        Parameters
        ----------
        features : array-like
            Input feature matrix of shape (n_samples, n_features).

        Returns
        -------
        dgl.DGLGraph
            A DGL graph with:
            - Nodes representing samples
            - Edges connecting k-nearest neighbors
            - Edge weights computed using specified weighting scheme
            - Edge weights stored in graph.edata['weight']
        """
        A = kneighbors_graph(features, n_neighbors=self.n_neighbors,
                        metric=self.metric, mode='distance',
                        include_self=True)
        # Convert distances to weights
        weights = self._compute_edge_weights(A.data)
        A.data = weights

        graph = dgl.from_scipy(A)
        graph.edata['weight'] = torch.tensor(weights, dtype=torch.float32)
        return graph

    def _create_graphs(self, train_features, val_features, test_features):
        """
        Create training and testing graphs using the provided feature sets.

        Parameters
        ----------
        train_features : array-like
            The feature set for training data.
        val_features : array-like
            The feature set for validation data.
        test_features : array-like
            The feature set for testing data.

        Returns
        -------
        tuple
            A tuple containing the training, validation and testing graphs (dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph).
        """
        train_graph = self._build_knn_graph(train_features)
        val_graph = self._build_knn_graph(val_features)
        test_graph = self._build_knn_graph(test_features)
        return train_graph, val_graph, test_graph

    def __call__(self, data: Data) -> Data:
        """
        Call method to process the data and create graphs.

        Parameters
        ----------
        data : Data
            The data object containing training and testing data.

        Returns
        -------
        Data
            The updated data object with training and testing graphs added.
        train_label : torch.Tensor
            Labels corresponding to the training data.
        val_label : torch.Tensor
            Labels corresponding to the validation data.
        test_label : torch.Tensor
            Labels corresponding to the testing data.
        ftl_shape : tuple
            Feature and Label size of the dataset.
        """
        input, label = data.get_train_data(return_type="numpy")
        test_input, test_label = data.get_test_data(return_type="numpy")
        train_input, val_input, train_label, val_label = train_test_split(
            input, label, test_size=0.05, random_state=42)



        if self.preprocess_type == 'SVD':
            embedder = TruncatedSVD(n_components=self.n_components)
            train_input = embedder.fit_transform(
                scipy.sparse.csr_matrix(train_input))
            val_input = embedder.transform(scipy.sparse.csr_matrix(val_input))
            test_input = embedder.transform(scipy.sparse.csr_matrix(test_input))

        train_graph, val_graph, test_graph = self._create_graphs(
            train_input, val_input, test_input)
        train_graph.ndata['feat'] = torch.tensor(train_input, dtype=torch.float32)
        val_graph.ndata['feat'] = torch.tensor(val_input, dtype=torch.float32)
        test_graph.ndata['feat'] = torch.tensor(test_input, dtype=torch.float32)

        data.data.uns['gtrain'] = train_graph
        data.data.uns['gval'] = val_graph
        data.data.uns['gtest'] = test_graph
        ftl_shape = (train_input.shape[1], train_label.shape[1])

        return data, train_label, val_label, test_label, ftl_shape
    