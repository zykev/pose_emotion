# -*- coding: utf-8 -*-

#!/usr/bin/env python
# This file is taken from Github project 'Proxemo' and modified by Zeyu Chen, BIAI Inc.
# Github link     :https://github.com/vijay4313/proxemo
# Paper           :ProxEmo: Gait-based Emotion Learning and Multi-view Proxemic Fusion for Socially-Aware Robot Navigation
#                  2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
# Title           :transform3DPose.py
# Original Author :Venkatraman Narayanan, Bala Murali Manoghar, Vishnu Shashank Dorbala, Aniket Bera, Dinesh Manocha
# Github          :"2020, Proxemo project"
# Version         :1.0
# Email           :vnarayan@terpmail.umd.edu, bsaisudh@terpmail.umd.edu
#==============================================================================
import numpy as np


def augment3D(data_features, theta, trans = 0, scale = 1):
    """Function to transform the given gait points
    theta should be given in degrees
    Args:
        data_features (np.array): gait cycle data
        theta (int): Augmentation angle (wrt skeletal root)
        trans (np.array): transformation matrix to be applied.
        scale (int): scaling factor for 2D camera matrix
    Returns:
        [np.array]: augmented matrix
    """
    theta = theta*np.pi/180
    rotMat = np.array([[np.cos(theta), 0, -np.sin(theta)],
                       [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
    #rotMat = rotMat.T
    xyTrans = [trans*np.cos(theta), trans*np.sin(theta), 0]
    xyTrans = np.array(xyTrans)
    K = np.array([[scale, 0, 0, 0], [0, scale, 0, 0],
                  [0, 0, 1, 0], [0, 0, 0, 1]])
    H = np.ones((4, 4))
    H[0:3, 0:3] = rotMat
    H[3, 0:3] = xyTrans
    P = K @ H
    data_features = np.reshape(data_features, (len(data_features), -1, 3))
    df = np.ones((data_features.shape[0], data_features.shape[1], 4))
    df[:, :, 0:3] = data_features
    df = df @ P
    df = df[:, :, 0:3]
    df = np.reshape(df, (len(df), -1))
    return df