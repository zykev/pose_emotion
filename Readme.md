# Emotion Recognition with Walk Pose through CNN

## Introduction

This repository records a convolutional neural network architecture that can categorically learn people's emotional state through their working pose (i.e., gait). The whole process consists of two parts. First, the 16 key skeleton position points of a human figure are learned given a human body image through a U-Net structure with the residual connection. Second, these skeleton points would be transformed as pseudo-RGB images through a technique called "image embedding" as proposed by [Venkatraman Narayanan, etc.](https://gamma.umd.edu/researchdirections/socrob/proxemo). In this way, the gait feature in these skeleton points could be learned by CNN structure and they are trained to match  people's emotional states including happy, angry, neutral, and sad. A fine-tuned network PoseNet based on the idea by [Xingyi Zhou etc.](https://github.com/xingyizhou/pytorch-pose-hg-3d) is adopted to obtain the 16 key skeleton position points. Then the gait emotion network transferring from the work by [Venkatraman Narayanan etc.](https://github.com/vijay4313/proxemo) is adopted to predict people's emotional state given their gait key points. The dataset for this network partly comes from [EWalk dataset](http://gamma.cs.unc.edu/GAIT/#EWalk) and [Edinburgh Locomotion MOCAP Dataset](https://bitbucket.org/jonathan-schwarz/edinburgh_locomotion_mocap_dataset/src/master). For the convenience of transfer learning, we also collect some data ourselves for training and validation. 

## Technical overview

1. video2image.py: change video to image frames.
2. image2pose.py: change imagesto 3d pose location data. Here, we use a pre-trained PoseNet, a U-Net structure to learn the key joint location based on the input images. The 2d joint location is learned by the U-Net output feature maps (heatmaps), where each map models the probability distribution of a single joint given the input image. The learned heatmaps are further processed together with the encoded representation of input images to learn the depth joint location. The network structure is stored in lib/mara_resnet.py.
3. vsgcnn.py: the network structure to learn emotional states given 3d joint location.
   Data augmentation: The input 3d joint location data is augmented by different views through translations and rotations. (front/right/left/back, see details in transfrom3D.py)
   Image embedding: Match R G B to (x,t), (y,t), (z,t) respectively, where (x, y, z) is the spatial location of skeleton joints learned by the PoseNet.
   Network structure: The conv blocks consist of group convolution + max pooling + batch norm. Group convolution is used to reduce parameters and learn features related to different views   
   separately.

![gait emotion recognition](https://github.com/zeyuchen-kevin/pose_emotion/blob/master/readme_fig.png)

## Usage

cmd: python main.py --video_path = path/to/video

e.g. python main.py --video_path=./pose_emotion/videos/test_video.mp4

Input: A 5-second video file of a walking person. The file format is .mp4, and the video file captured by the camera should be stored in the pose_emotion/videos folder
Output: Expression prediction from 4 angles, that is, the return value of main() in main.py
