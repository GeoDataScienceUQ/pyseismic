# -*- coding: utf-8 -*-

"""
maintainer: <corlayquentin@gmail.com>
"""

import numpy as np 
import pandas as pd
from scipy.signal import argrelextrema
from datetime import datetime
# import math
# from numba import njit, jit, vectorize
# import itertools
# import multiprocessing
# import sys
# from tqdm import tqdm
# import laspy
from sklearn.cluster import DBSCAN
import dask.array as da
import dask

from d2geo.attributes.EdgeDetection import EdgeDetection
from d2geo.attributes.util import compute_chunk_size
from utils import *


class PointCloudSeismicInterpretation():
    """
    Description
    -----------
    Create Point Cloud Seismic dataframe from a 3D seismic cube
    The point cloud is extracted by local extrema extraction in the trace direction, 
        filter on semblance and filter on amplitude

    Attributes:
        seismic_array (np.array): 3D seismic cube as a 3D numpy array with amplitude reflectivity values
        point_cloud (np.array): point cloud attribute as a 2D array with shape (n, 3) with n being the number 
            of points and the second axis being the coordinates of the points (x, y, z)
        amplitude_point_cloud (np.array): amplitude reflectivity values of the points of the point cloud
        semblance_array (np.array): 3D semblance cube computed from the 3D seismic 
        semblance_point_cloud (np.array): semblance values of the points of the point cloud 
        ampv95p (float): approximate of the 95-th percentile of the seismic amplitude

    Methods:
        load_seismic_array: Loads the 3D seismic cube
        extrema_extraction: Extracts the point cloud from the seismic with local extrema in the trace direction
        extrema_extraction_dask: Extracts the point cloud from the seismic with local extrema in the trace direction
            - multiprocessing with dask
        filter_point_cloud_with_semblance: Computes the semblance cube and filters the point cloud based on semblance cut-off 
        filter_point_cloud_with_amplitude: Filters the point cloud based on amplitude cut-off
        DBSCAN_segmentation_sklearn: segments the point cloud based on DBSCAN clustering - sklearn implementation

    """
    def __init__( self, seismic_array, ampv95p=None ):
        self.load_seismic_array(seismic_array, ampv95p=ampv95p)

    def load_seismic_array(self, seismic_array, ampv95p=None):
        """
        Description
        -----------
        Loads the 3D seismic cube, computes 95th amplitude quartile and the intialise the point cloud objects 

        Args:
            seismic_array (np.array): 3D seismic cube as a 3D numpy array with amplitude reflectivity values
            ampv95p (float or None): the 95-th percentile of the seismic amplitude 
                - if not None prevents from having to compute it
        """
        self.seismic_array = seismic_array
        if not ampv95p:
            self.ampv95p = np.percentile(self.seismic_array[self.seismic_array.shape[0]//2:self.seismic_array.shape[0]//2+100,
                                                                self.seismic_array.shape[1]//2:self.seismic_array.shape[1]//2+100,
                                                                : ], 95)  # compute the 99th percentile of amplitude to scale
        else:
            self.ampv95p = ampv95p
        self.point_cloud = np.array([])
        self.amplitude_point_cloud = np.array([])
        self.semblance_array = np.array([])
        self.semblance_point_cloud = np.array([])

    def extrema_extraction(self, extrema_type=np.greater):
        """
        Description
        -----------
        Extract extrema in the trace direction (3rd dimension) of a 3D array
        Creates a point cloud attribute (numpy.array) of shape [n_extrema, 3]
        Each row of the point cloud being [x,, y, z] of the each extrema
        Amplitude values are normalized and stored in a amplitude point cloud attribute 

        Args:
            extrema_type (callable, default np.greater): np.greater or np.less, the type of extrema to extract (maxima or minima)
        """
        t0 = datetime.now()
        (nx, ny, nz) = self.seismic_array.shape
        Lx = []
        Ly = []
        Lz = []
        for i in range (nx):
            for j in range (ny):
                maxima = argrelextrema(self.seismic_array[i, j], extrema_type)[0]
                Lx.extend([i]*len(maxima))
                Ly.extend([j]*len(maxima))
                Lz.extend(maxima)
        self.point_cloud = np.zeros((len(Lx), 3), dtype='int32')
        self.point_cloud[:, 0] = Lx
        self.point_cloud[:, 1] = Ly
        self.point_cloud[:, 2] = Lz
        self.amplitude_point_cloud = NormalizeData(self.seismic_array[Lx, Ly, Lz], thr=self.ampv95p, type='positive')
        self.semblance_point_cloud = np.array([])
        print('Point cloud created -  {} points - time to execute: {}'.format(self.point_cloud.shape[0], datetime.now()-t0))

    def extrema_extraction_dask(self, extrema_type=np.greater):
        """
        Description
        -----------
        Extract extrema in the trace direction (3rd dimension) of a 3D array
        Creates a point cloud attribute (numpy.array) of shape [n_extrema, 3]
        Each row of the point cloud being [x,, y, z] of the each extrema
        Amplitude values are normalized and stored in a amplitude point cloud attribute 
        Dask multiprocessing implementation

        Args:
            extrema_type (callable, default np.greater): np.greater or np.less, the type of extrema to extract (maxima or minima)
        """
        t0 = datetime.now()
        (xChunkSize, yChunkSize, zChunkSize) = compute_chunk_size(self.seismic_array.shape, 
                        self.seismic_array.dtype.itemsize, 
                        kernel=(1, 1, self.seismic_array.shape[2]),
                        preview=None)
        seis_ilBlock_dask_trace = da.from_array(self.seismic_array, chunks=(xChunkSize, yChunkSize, self.seismic_array.shape[2]))
        blocks = seis_ilBlock_dask_trace.to_delayed()#.ravel()
        results = []
        for ixb, xb in enumerate(blocks):
            results.extend([da.from_delayed(get_point_cloud_chunks(yb[0], ixb*xChunkSize, iyb*xChunkSize, extrema_type), shape=(3, np.nan), dtype=np.int64)  
                            for iyb, yb in enumerate(xb)])
        arr = da.concatenate(results, axis=1, allow_unknown_chunksizes=True)
        self.point_cloud = arr.compute()
        self.point_cloud = self.point_cloud.T
        print('point cloud shape {}'.format(self.point_cloud.shape))
        self.amplitude_point_cloud = NormalizeData(
            self.point_cloud[:,3], thr=self.ampv95p, type='positive'
        )
        self.point_cloud = self.point_cloud[:,:3]
        self.semblance_point_cloud = np.array([])
        print('Point cloud created -  {} points - time to execute: {}'.format(self.point_cloud.shape[0], datetime.now()-t0))

    def filter_point_cloud_with_semblance(self, kernel=(3, 3, 9), thr=0.9, in_place=True):
        """
        Description
        -----------
        Computes the semblance cube and filters the point cloud based on semblance cut-off 

        Args:
            kernel (tuple, default (3, 3, 9)): tuple of int (x, y, z) the dimension of the 3D kernel applied to compute semblance
            thr (float, default 0.9): semblance threshold, float between 0 and 1
            in_place (bool, default True): If True, perform operation in-place.
        """
        if self.point_cloud.size == 0:
            print('Point cloud not computed - extracting extrema first')
            self.extrema_extraction()
            return()
        
        if self.semblance_array.size == 0:
            t0 = datetime.now()
            daSemblance = EdgeDetection().semblance(darray=self.seismic_array, kernel=kernel) #Dask instance of the semblance
            self.semblance_array = daSemblance.compute()
            self.semblance_point_cloud = np.array([])
            print('Semblance attribute computed - time to execute: {}'.format(datetime.now()-t0))
        if self.semblance_point_cloud.size == 0:
            t0 = datetime.now()
            self.semblance_point_cloud = self.semblance_array[self.point_cloud[:, 0], self.point_cloud[:, 1], self.point_cloud[:, 2]]
            print('Semblance point cloud extracted - time to execute: {}'.format(datetime.now()-t0))
        t0 = datetime.now()
        if in_place:
            mask = [ self.semblance_point_cloud > thr ]
            self.point_cloud = self.point_cloud[mask]
            self.amplitude_point_cloud = self.amplitude_point_cloud[mask]
            self.semblance_point_cloud = self.semblance_point_cloud[mask]
            print('Applied semblance filter in place -  {} points - time to execute: {}'.format(self.point_cloud.shape[0], datetime.now()-t0))
        else:
            return (self.point_cloud[ self.semblance_point_cloud > thr])

    def filter_point_cloud_with_amplitude(self, thr=0.25, in_place=True):
        """
        Description
        -----------
        Filters the point cloud based on amplitude cut-off

        Args:
            thr (float, default 0.25): amplitude threshold, float between 0 and 1
            in_place (bool, default True): If True, perform operation in-place.
        """
        if self.point_cloud.size == 0:
            print('Point cloud not computed - extract extrema first')
        t0 = datetime.now()
        if in_place:
            mask = [ self.amplitude_point_cloud > thr]
            self.point_cloud = self.point_cloud[mask]
            self.semblance_point_cloud = self.semblance_point_cloud[mask]
            self.amplitude_point_cloud = self.amplitude_point_cloud[mask]
            print('Applied amplitude filter in place -  {} points - time to execute: {}'.format(self.point_cloud.shape[0], datetime.now()-t0))
        else:
            return (self.point_cloud[ self.amplitude_point_cloud > thr ])

    def DBSCAN_segmentation_sklearn(self, eps=2, min_samples=8, z_factor=1):
        """
        Description
        -----------
        Segments the point cloud based on DBSCAN clustering - sklearn implementation

        Args:
            eps (float, default 2): epsilon distance parameter to DBSCAN
            min_samples (int, default 8): minimum number of points parameter to DBSCAN
            z_factor (int, default 1): vertical exageration to apply to the seismic point cloud
        """
        t0=datetime.now()
        point_cloud_to_compute = self.point_cloud.copy()
        point_cloud_to_compute[:, 2] = self.point_cloud[:, 2]*z_factor
        print('vertical exageration applied')
        clustering = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', n_jobs=-1).fit(point_cloud_to_compute)
        self.labels = clustering.labels_
        print('seismic segmented - {} clusters - time {}'.format( self.labels.max()+1, datetime.now()-t0 ))

######
# Functions
######

def NormalizeData(data: np.array, thr: float, type='positive'):
    """
    Description
    -----------
    Normalize data between [0, thr] if positive and [-thr, 0] if negative

    Args: 
        data (np.array): 
        thr (float): threshold to ceil the  data 

    Returns:
        np.array: data normalized
    """
    if type == 'positive':
        data[data < 0] = 0.
        data[data > thr] = thr
        return (data / thr)
    elif type == 'negative':
        data[data > 0] = 0.
        data[data < -thr] = -thr
        return (data/ thr)
    else:
        print('Unrecognized type, expected "positive" or "negative", got {}'.format(type))
        return None

@dask.delayed
def get_point_cloud_chunks(seismic, xchunk, ychunk, extrema_type=np.greater):
    """
    Description
    -----------
    Extract extrema of a 1D signal

    Args: 
        data (np.array): 
        thr (float): threshold to ceil the  data 

    Returns:
        np.array: point cloud of extrema from data, shape (3, n): first axis being the coordinates, 
            second axis being the n extrema points
    """
    (nx, ny, nz) = seismic.shape
    Lx = []
    Ly = []
    Lz = []
    Lamp = []
    for i in range (nx):
        for j in range (ny):
            maxima = argrelextrema(seismic[i,j], extrema_type)[0]
            Lx.extend([xchunk + i]*len(maxima))
            Ly.extend([ychunk + j]*len(maxima))
            Lz.extend(maxima)
            Lamp.extend(seismic[i,j,maxima])
    point_cloud = np.zeros((4, len(Lx),), dtype='int32')
    point_cloud[0, :] = Lx
    point_cloud[1, :] = Ly
    point_cloud[2, :] = Lz
    point_cloud[3, :] = Lamp
    return(point_cloud)