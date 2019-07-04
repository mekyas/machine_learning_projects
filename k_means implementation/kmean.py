#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy



class KMeans(object):
    def __init__(self, n_clusters):
        self.k = n_clusters
    
    def _kmean(self, data, k):
        """function that implements the k-means algorithms
        inputs:
            data: 2D numpy matrix 
                  rows are points and columns are coordinates 
            k : number of clusters
        outputs:
            group: group id of each points
            centroids: coordianetes of each centroids
        """
        #np.seterr(divide='ignore', invalid='ignore')
        # 1 step: choose random points as initial centroids
        X_centroid = np.random.randint(low = np.min(data[0,:]), high=np.max(data[0,:]), size=k)
        Y_centroid = np.random.randint(low = np.min(data[:,1]), high=np.max(data[:,1]), size=k)
        centroids = np.array([X_centroid, Y_centroid]).T
        #
        while True:
            # calculate distance
            distance = np.array([np.linalg.norm(data-centroids[i,:], axis=1) for i in range(k)])
            # assign each point to closest centroid
            labels = np.argmin(distance, axis=0)
            # copy the centroids coordiantes
            old_centroids = copy.deepcopy(centroids)
            # update centroids coordiates
            centroids = np.array([np.nanmean(data[np.where(labels==i)[0],:], axis=0) 
                                if np.any(labels==i) else old_centroids[i,:] for i in range(k) ])
            # verify if centroids changed
            if np.allclose(centroids, old_centroids):
                break
    
        return labels, centroids
    
    def fit(self, data):
        self.labels_, self.cluster_centers_ = self._kmean(data, self.k)
    
    def predict(self, test_data):
        distance = np.array([np.linalg.norm(test_data-self.cluster_centers_[i,:], axis=1) for i in range(self.k)])
        return np.argmin(distance, axis=0)
    
