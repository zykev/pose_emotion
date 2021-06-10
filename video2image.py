# -*- coding: utf-8 -*-

# This file is created by Zeyu Chen, BIAI, Inc.
# Title           :video2image.py
# Version         :1.0
# Email           :k83110835@126.com
# Copyright       :BIAI, Inc.
#==============================================================================


import os
#os.chdir('E:/pose_emotion')
import cv2
import re
import time
import shutil


def is_video(file_name):
    """
    This function will detect whether a file is a video.
    """
    video_ext = ['mp4', 'mov', 'mpg']
    ext = file_name[file_name.rfind('.') + 1:].lower()
    return ext in video_ext


def save_image(image,addr,num):
    """
    Define the images to be saved.
    Args:
        image: the name of the saving image
        addr:  the picture directory address and the first part of the picture name
        num:   int dtype, the id number in the image filename
    """
    address = os.path.join(addr, 'image' + '_' + str(num)+ '.jpg')
    cv2.imwrite(address,image)

def video2image(video_name = 'test_video'):
    """
    Define a function to change the input video to images.
    """
        
    #Capture the camera
    #video_url="http://admin:admin@192.168.2.13:8081/video"
    #cameraCapture = cv2.VideoCapture(video_url)
    
    cameraCapture = cv2.VideoCapture(0)
    if cameraCapture.isOpened() == False:
        
        print('Camera Capture failed!')
    else:
        
        fps = 30 #save video frame number settings
        i = 0
        output_image_path = os.path.join('images', video_name)
        input_video_path = os.path.join('videos', video_name + '.mp4')

        # Create a new folder after removing the old picture folder
        if os.path.exists(output_image_path):
            shutil.rmtree(output_image_path)
        
        os.makedirs(output_image_path)
                
        
        size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH))) 
        #image size to be saved(width,height) 后面进行了旋转，故此处反过来
        videoWriter = cv2.VideoWriter(input_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
        capture_duration = 10 #duration time for video (in seconds)
        start_time = time.time()
        
        while (int(time.time() - start_time) < capture_duration):
            success, frame = cameraCapture.read()
            if success:
                
                frame = cv2.transpose(frame)
                frame = cv2.flip(frame,1)
                videoWriter.write(frame) #save video
                cv2.imshow('frame', frame) # show video
                save_image(frame, output_image_path, i) #save images from video
                cv2.waitKey(int(1000/30))
                # if i % 10 == 0:
                #     print('save images from video {video_name}: image {i} etc...'.format(video_name = video_name, i = i))
                i += 1
            else:
                break
        cameraCapture.release()
        videoWriter.release() 
        cv2.destroyAllWindows()


def readvideo2image(opt, video_name = 'test_video', video_path = '.'):
    # 读取视频文件
    # if video_name != '':
    #     video_path = os.path.join('videos', video_name + '.mp4') #直接视频文件
    # else:
    #     #video_path = 'videos/' #含有视频文件的文件夹
    #     print('Video name is empty! You should give a video name which exists in the video folder to continue.')

    if is_video(video_path):
        
        output_image_path = os.path.join('images', video_name)
        if os.path.exists(output_image_path):
            shutil.rmtree(output_image_path)
        os.makedirs(output_image_path)
        videoCapture = cv2.VideoCapture(video_path)

        success, frame = videoCapture.read()
        i = 0
        while success:
            if opt.vidmode == 'vertical':
                frame = cv2.transpose(frame)
                frame = cv2.flip(frame, 1)
            save_image(frame, output_image_path, i)
            if i % 20 == 0 and opt.outstep:
                print('save images from {video_name}: image {i} etc...'.format(video_name = video_name, i = i))
            i += 1
            success, frame = videoCapture.read()


    # # 情形一：含有视频文件的文件夹
    # elif os.path.isdir(video_path):
    #     ls = os.listdir(video_path)
    #     for i, file_name in enumerate(sorted(ls)):
    #         if is_video(file_name):
    #             sub_video_info = re.compile('(\w*).mp4')
    #             sub_video_name = sub_video_info.findall(file_name)
    #
    #             output_image_path = os.path.join('images', sub_video_name)
    #             if os.path.exists(output_image_path):  # 移除旧图片文件夹后创建新的文件夹
    #                 shutil.rmtree(output_image_path)
    #
    #             os.makedirs(output_image_path)
    #
    #             file_path = video_path + file_name
    #             videoCapture = cv2.VideoCapture(file_path)
    #
    #
    #             success, frame = videoCapture.read()
    #             i = 0
    #             while success:
    #
    #                 save_image(frame, output_image_path, i)
    #                 if success and i % 20 == 0:
    #                     print('save images from {sub_video_name}: image {i} etc...'.format(sub_video_name=sub_video_name,
    #                                                                                        i=i))
    #                 i += 1
    #                 success, frame = videoCapture.read()

        
'''
if __name__ == '__main__':
    
    video2image(video_name='testa') # param video_name
    #readvideo2image()
'''


