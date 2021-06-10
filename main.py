# -*- coding: utf-8 -*-
# This file is created by Zeyu Chen, BIAI, Inc.
# Title           :main.py
# Version         :1.0
# Email           :k83110835@126.com
# Copyright       :BIAI, Inc.
#==============================================================================


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import os

from lib.opts import opts
from video2image import readvideo2image
#from video2image import video2image
from image2pose import image2pose
from show_emo import predict_emo
from visual_pose import animation_pose
# from show_emo import show_video

def main():

    # duration = 0 #begin program agter duration time
    # start_time = time.time()
    # while int(time.time() - start_time) < duration:
    #     time.sleep(1)
    # print('waiting for {} seconds...'.format(duration))
    
    
    #input a video which has a unique name. This video must exists in directory './videos'.
    opt = opts().parse() #load parameters for pose detection
    video_path = opt.video_path  #obtain current video path
    video = os.path.basename(opt.video_path).split('.')[0] #obtain current video name in video path
    
    readvideo2image(opt, video, video_path)   #load existing videos in video folder
    # video = 'test_video43'
    # video2image(video)  #make new videos and save it in video folder

    pose_data = image2pose(opt, video)
    emotion, emotion_syn = predict_emo(pose_data)

    _ = animation_pose(video, pose_data, 2)  # param video_name, angle(0,1,2,3)
    #print('============================================================')
    # print('output predicted emotion:', emotion)
    # print('\noutput synthetical emotion:', '\nPositive (Happy, Neutral): {}%'.format(emotion_syn['positive']), '\nNegative (Angry, Sad): {}%'.format(emotion_syn['negative']))
    print(emotion, emotion_syn)

    #show_video(video)  #show final result
    
    # #delete pose_coords csv file
    # os.remove(os.path.join('posedata', video + '_pose_coords.csv'))
    
    return emotion, emotion_syn
    
    
    
    #view default param
    #readvideo2image.__defaults__
    
if __name__ == '__main__':
    _ = main()

