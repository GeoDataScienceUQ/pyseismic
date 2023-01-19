# -*- coding: utf-8 -*-

"""
maintainer: <corlayquentin@gmail.com>
"""

import numpy as np
from math import hypot, atan2
import cv2
from sklearn.neighbors import KDTree
from datetime import datetime
import pandas as pd

class PointCloudObjectRetrieval():
    """Characterise every object in a seismic point cloud"""
    def __init__( self, point_cloud, semblance, labels_point_cloud ):
        self.point_cloud = point_cloud
        self.semblance = semblance
        self.labels_point_cloud = labels_point_cloud

    def get_features(self, selected_clusters):
        """
        Extract pandas dataFrame containing the features descriptives of every selected point cloud
        Features: segmentID, n points, amplitude mean, semblance mean, Zeboudj distance, 
            contour ratio, lambda1, lambda2, "lambda3", linearity,
            slope, planarity, orientation, rZHigh, rZLow
        """
        t0 = datetime.now()
        Features = []
        i=0
        for isegment in selected_clusters:
            i+=1
            print("({}/{})".format(i, len(selected_clusters)), end = "\r")
            Features.append([isegment] + extract_features(
                    self.point_cloud[np.isin(self.labels_point_cloud, isegment)], 
                    self.semblance[np.isin(self.labels_point_cloud, isegment)]
                )
            )
        ## store it as a dataframe
        featureDF = pd.DataFrame(data=Features, columns=[
            "segmentID", "n points", "amplitude mean", "semblance mean", "Zeboudj distance", 
            "contour ratio", "lambda1", "lambda2", "lambda3", "linearity",
            "slope", "planarity", "orientation", "rZHigh", "rZLow",
        ])
        featureDF.set_index('segmentID', inplace=True)
        featureDF=(featureDF-featureDF.mean())/featureDF.std()
        print('time: {}'.format(datetime.now()-t0))
        return featureDF


#########
## functions
#########

def get_aspect_ratio(pcd):
    """
    """
    pcd_sampling = np.random.choice(len(pcd), 
        size=min(int(len(pcd)*0.5), 500), replace=False) #min([int(len(point_cloud)*0.2), 1000]), replace=False)
    eigvals, eigvecs = np.linalg.eig(np.cov(pcd[pcd_sampling].T))
    idx = eigvals.argsort()[::-1]   
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    orientation = atan2(eigvecs[0,1], eigvecs[0,0])
    return (
        [eigvals[0], eigvals[1], eigvals[2], 
        (eigvals[0]-eigvals[1])/eigvals[0], eigvals[2]/eigvals[0], (eigvals[1]-eigvals[2])/eigvals[0], orientation]
    )

def get_outline_ratio(pcd):
    """
    """
    #convert 2d projection to image
    #shift X and Y coordinate to 0 origin
    X_shift = pcd[:,0]-min(pcd[:,0]); Y_shift = pcd[:,1]-min(pcd[:,1])
    X_shift = X_shift.astype(int, copy=False); Y_shift = Y_shift.astype(int, copy=False)
    Img = np.zeros((max(X_shift)+1, max(Y_shift)+1))
    Img[X_shift, Y_shift] = 255
    Img=Img.astype('uint8')
    Imgblur = cv2.blur(Img, (10, 10))
    _, Imgthresh = cv2.threshold(Imgblur, 50, 255, cv2.THRESH_BINARY)
    imct, _ = cv2.findContours(Imgthresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = 0
    for contour in imct:
        length += sum(hypot(x1 - x2, y1 - y2) for (x1, y1), (x2, y2) in zip(contour[:,0, :], contour[1:,0, :]))
    surface = Imgthresh.sum()/255
    return (length/surface)

def get_zeboudj_distance(pcd, distance = 2.5):
    """
    """
    #build kdtree for each cluster
    pcd_sampling = np.random.choice(len(pcd), size=min(int(len(pcd)*0.5), 10000), replace=False)
    cl_sample = pcd[pcd_sampling]
    tree = KDTree(cl_sample[:,:3])
    #loop over every point within the cluster to compute color distances with its neighbourhood
    L_neighbors_idxs = tree.query_radius(cl_sample[:,:3], r=distance)
    CI = 0
    for ipoint, neighbors_idxs in enumerate (L_neighbors_idxs):
        if len(neighbors_idxs) > 0:
            Dist_neighbs = np.zeros(len(neighbors_idxs))
            for ineighb, neighb in enumerate (neighbors_idxs):
                Dist_neighbs[ineighb] = cl_sample[:,3][ipoint] - cl_sample[:,3][neighb]
            CI += Dist_neighbs.max()
    CI = CI/len(cl_sample)
    return (CI)

def get_depth_distribution(pcd):
    """
    """
    zmin = np.percentile(pcd[:,2], 2)
    zmax = np.percentile(pcd[:,2], 98)
    r_zlow = len(pcd[pcd[:,2]<zmin+(zmax-zmin)/4])/len(pcd) #proportion_under_one_fourth
    r_zhigh = len(pcd[pcd[:,2]>zmax-(zmax-zmin)/4])/len(pcd) #proportion_over_three_fourth
    return([r_zlow, r_zhigh])

def extract_features(pcd, semblance):
    """
    """
    n_points = len(pcd)
    amp_mean = pcd[:,3].mean()
    semb_mean = semblance.mean()
    zeboudj_2_5 = get_zeboudj_distance(pcd, distance=2.5)
#     zeboudj_3_5 = get_zeboudj_distance(pcd, distance=3.5)
    outline_ratio = get_outline_ratio(pcd[:,:2])
    aspect_ratios = get_aspect_ratio(pcd[:,:3])
    r_z = get_depth_distribution(pcd[:,:3])
    return ([n_points]+[amp_mean]+[semb_mean]+[zeboudj_2_5]+[outline_ratio]+aspect_ratios+r_z)

