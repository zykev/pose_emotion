# Emotion Recognition with Walk Pose through CNN

## Introduction

This repository records a convolutional neural network architecture which can learn people's emotion state categorically through their working pose (i.e. gait). The whole process consists of two parts. First, the 16 key skeleton position points of a human figure is learned given a human body image through resnet. Second, these skeleton points would be transformed as pseudo RGB image through a technique called "image embedding" as proposed by . In this way, the gait feature in these skeleton points could be learned by CNN structure and they are trained to match  people's emotion state including happy, angry, neutral and sad. A fine-tuned network based on the idea by [Xingyi Zhou etc.](https://github.com/xingyizhou/pytorch-pose-hg-3d) is adopted to obtain the 16 key skeleton position points. Then the gait emotion network transferring from the work by [Venkatraman Narayanan etc.](https://github.com/vijay4313/proxemo) is adopted to predict people's emotion state given their gait key points. The dataset for this network partly comes from [EWalk dataset](http://gamma.cs.unc.edu/GAIT/#EWalk). For the convenience of transfer learning, we also collect some data by ourselves for validation. 
![gait emotion recognition process](https://github.com/zeyuchen-kevin/pose_emotion/edit/master/readme_fig.png)

## Usage

cmd: python main.py --video_path = path/to/video

e.g. python main.py --video_path=./pose_emotion/videos/test_video.mp4

Input: A 5-second video file of a walking person. The file format is .mp4, and the video file captured by the camera should be stored in the pose_emotion/videos folder
Output: Expression prediction from 4 angles, that is, the return value of main() in main.py
