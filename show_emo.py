# -*- coding: utf-8 -*-

# This file is created by Zeyu Chen, BIAI, Inc.
# Title           :show_emo.py
# Version         :1.0
# Email           :k83110835@126.com
# Copyright       :BIAI, Inc.
#==============================================================================


import os
import glob
import cv2
import torch
import numpy as np
#import win32gui

from vsgcnn import VSGCNN
from transform3DPose import augment3D



class TrainTestLoader(torch.utils.data.Dataset):
    """Create torch dataset object from gait cycle data."""

    def __init__(self, data, joints = 16, coords = 3, num_classes = 4):
        """Initialize the dataloader.
        Args:
            data (np.array): gait cycles
            label (np.array): emotion class 1-hot vector
            joints (int): Number of joints in gait cycles
            coords (int): Number of co-ordinates
                          representing each joint (2D/3D)
            num_classes (int): Number of emotion classes
        """
        # data: N C T J
        self.data = data
        self.joints = joints
        self.coords = coords

    def __len__(self):
        """Return dataset size."""
        return len(self.data)

    def _convert_skeletion_to_image(self, data_numpy):
        """Convert gait cycle into image sequence.
        Args:
            data_numpy (np.array): Gait sequence data
        """
        # (1, 3, 75, 16, 1) shape
        data_numpy = np.squeeze(data_numpy, axis=0)   # (3, 75, 16, 1)
        data_max = np.max(data_numpy, (1, 2, 3)) #maximun value in each coord(channel)
        data_min = np.min(data_numpy, (1, 2, 3))
        img_data = np.zeros((data_numpy.shape[1],  # time_steps * joints * coords(channels)
                             data_numpy.shape[2],
                             data_numpy.shape[0]))
        
        #projection: data_min --> 255 data_max --> 0
        img_data[:, :, 0] = (data_max[0] - data_numpy[0, :, :, 0]
                             ) * (255 / (data_max[0] - data_min[0]))
        img_data[:, :, 1] = (data_max[1] - data_numpy[1, :, :, 0]
                             ) * (255 / (data_max[1] - data_min[1]))
        img_data[:, :, 2] = (data_max[2] - data_numpy[2, :, :, 0]
                             ) * (255 / (data_max[2] - data_min[2]))   

        img_data = cv2.resize(img_data, (244, 244))
        
        img_data /= 255

        return img_data

    def __getitem__(self, index):
        """Get data & label pair for each gait cycle.
        Args:
            index (int): Sequence number to retrieve
        Returns:
            [list]: gait cycle and emotion label pair
        """

        # data: N C T J
        data_numpy = np.asarray(self.data[index])
        data_numpy = np.reshape(data_numpy,
                                (1,
                                 data_numpy.shape[0],
                                 self.joints,
                                 self.coords,
                                 1))
        data_numpy = np.moveaxis(data_numpy, [1, 2, 3], [2, 3, 1])
        self.N, self.C, self.T, self.J, self.M = data_numpy.shape #C:coords T:time J:joints
        img_data = self._convert_skeletion_to_image(data_numpy)

        return img_data

    
# def pose_multi_view(pose_coords, num_groups = 4):
#     """A function to return pose coordinates based on four different viewing angles"""
#
#     pose_coords_groups = np.zeros((num_groups * pose_coords.shape[0], pose_coords.shape[1], pose_coords.shape[2]))
#     angles = np.arange(0,360,90)
#     for i in range(num_groups):
#         if pose_coords.shape[0] == 1:
#             pose_coords_groups[i,:,:] = augment3D(pose_coords[0,:,:], angles[i], 0, 1)
#         else:
#             for j in range(pose_coords.shape[0]):
#                pose_coords_groups[j * num_groups + i,:,:] = augment3D(pose_coords[j,:,:], angles[i], 0, 1)
#
#     return pose_coords_groups


#import pretrained model        
def load_model():
    """Load pretrained weights for model."""


    vgcnn = VSGCNN(4, 3, 4, 0.2) #n_classes, in_channels, num_groups, dropout
        
    # pretrained .pth model file
    
    #emomodel_path = 'models/mdl_epoch283_acc76.15_model.pth.tar'
    
    #checkpoint = torch.load(emomodel_path, map_location = 'cpu')
    #vgcnn.load_state_dict(checkpoint['model_state_dict'], strict=False) # loading parameters

    emomodel_path = 'models/mdl_epoch283_acc76.15_model.pth'
    state_dict = torch.load(emomodel_path, map_location='cpu')
    vgcnn.load_state_dict(state_dict, strict=True)
    
        
    return vgcnn

def load_pose_data(pose_coords, num_groups=4):
    # change joints position from posenet to proxemo net
    pose_coords = pose_coords[:,
                  [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36, 37, 38, 33, 34, 35, 30, 31, 32, 39, 40, 41, 42,
                   43, 44, 45, 46, 47, 9, 10, 11, 12, 13, 14, 15, 16, 17, 6, 7, 8, 3, 4, 5, 0, 1, 2]]

    # repeat pose_coords serval times
    if pose_coords.shape[0] < 50:
        pose_coords = np.tile(pose_coords, (8, 1))

    pose_coords = pose_coords.reshape(1, pose_coords.shape[0], pose_coords.shape[1])  # shape 1*T*(J*C)

    pose_coords_groups = np.zeros((num_groups * pose_coords.shape[0], pose_coords.shape[1], pose_coords.shape[2]))
    angles = np.arange(0, 360, 90)
    for i in range(num_groups):
        if pose_coords.shape[0] == 1:
            pose_coords_groups[i, :, :] = augment3D(pose_coords[0, :, :], angles[i], 0, 1)
        else:
            for j in range(pose_coords.shape[0]):
                pose_coords_groups[j * num_groups + i, :, :] = augment3D(pose_coords[j, :, :], angles[i], 0, 1)

    return pose_coords_groups


# def load_data(multi_view = True, video_name = 'test_video'):
#     """Obtain pose coordinate data from single person."""
#
#     pose_coords = np.loadtxt(os.path.join('posedata', video_name + '_pose_coords.csv'), delimiter=',') #shape T*(J*C)
#     #change joints position from posenet to proxemo net
#     pose_coords = pose_coords[:,[18,19,20,21,22,23,24,25,26,27,28,29,36,37,38,33,34,35,30,31,32,39,40,41,42,43,44,45,46,47,9,10,11,12,13,14,15,16,17,6,7,8,3,4,5,0,1,2]]
#
#     #repeat pose_coords serval times
#     if pose_coords.shape[0] < 50:
#         pose_coords = np.tile(pose_coords, (8,1))
#
#     pose_coords = pose_coords.reshape(1, pose_coords.shape[0], pose_coords.shape[1]) #shape 1*T*(J*C)
#
#     if multi_view:
#         pose_coords = pose_multi_view(pose_coords)
#
#     return pose_coords
#
#
# def load_multi_data(multi_view = True):
#     """Obtain pose coordinate data from multiple people."""
#     pose_multi_coords = glob.glob('posedata/*_pase_coords.csv')
#     print(f'---> Number of coordinate data files = {len(pose_multi_coords)}')
#     pose_coords = []
#     for pose_coords_csv in pose_multi_coords:
#         pose_single_coords = np.loadtxt(os.path.join('posedata', pose_coords_csv), delimiter = ',')
#         pose_single_coords = pose_single_coords[:,[18,19,20,21,22,23,24,25,26,27,28,29,36,37,38,33,34,35,30,31,32,39,40,41,42,43,44,45,46,47,9,10,11,12,13,14,15,16,17,6,7,8,3,4,5,0,1,2]]
#         #repeat pose_coords serval times
#         if pose_single_coords.shape[0] < 50:
#             pose_single_coords = np.tile(pose_single_coords, (8,1))
#
#         pose_coords.append(pose_single_coords)
#
#     pose_coords_new = np.zeros(shape = (len(pose_coords), pose_coords[0].shape[0], pose_coords[0].shape[1]))
#     for i in range(len(pose_coords)):
#         pose_coords_new[i,:,:] = pose_coords[i]
#
#     return pose_coords_new
    

    
# def predict_emo(video_name = 'test_video', multi_view = True):
def predict_emo(pose_coords):

    """Predict emotion from pose coordinate data through model."""
    
    vgcnn = load_model()
    vgcnn.eval() ## put models in eval mode
    # pose_coords = load_data(multi_view, video_name) #param multi_view, video_name
    pose_coords = load_pose_data(pose_coords)
    loader = torch.utils.data.DataLoader(dataset=TrainTestLoader(pose_coords))
    
    
    result_frag = []
    for data in loader:
        data = data.float()
        with torch.no_grad():
            output = vgcnn(data)
        result_frag.append(output.data.numpy())
    
    result = np.concatenate(result_frag)
    rank = result.argsort()
    pred_label = rank[:, -1]
    pred_emotion_label = (pred_label - pred_label % 4) / 4
    pred_emotion_label = pred_emotion_label.astype(int)
    
    
    # emotion = ['Negative', 'Positive', 'Negative', 'Positive']
    # emo_result = [emotion[i] for i in pred_emotion_label]
    # emo_result = np.array(emo_result)
    # per_p = int(sum(emo_result=='Positive')/len(emo_result) * 100)
    # per_n = int(sum(emo_result=='Negative')/len(emo_result) * 100)
    #return per_p, per_n
    
    emotion = ['angry', 'happy', 'sad', 'neutral']
    emotion_tag = ['negative', 'positive', 'negative', 'positive']
    # if multi_view:
        
    #     emotion_idx = np.argmax(np.bincount(pred_emotion_label))
    #     print('This person seems to be', emotion[emotion_idx])
    # else:
        
    #     emotion_idx = pred_emotion_label[0]
    #     print('This person seems to be', emotion[emotion_idx])
        
    # return emotion[emotion_idx]
    
    emo_result = [emotion[i] for i in pred_emotion_label]
    emo_tag = [emotion_tag[i] for i in pred_emotion_label]
    emo_tag = np.array(emo_tag)
    per_p = int(sum(emo_tag == 'positive') / len(emo_tag) * 100)
    per_n = int(sum(emo_tag == 'negative') / len(emo_tag) * 100)
    emo_result_syn = {'positive': per_p, 'negative': per_n}
    return emo_result, emo_result_syn
    
    

'''
def show_video(video_name = 'test_video', multi_view = True):
    """ A function to return the captured video and predicted emotion."""
    
    video_path = os.path.join('videos', video_name + '.mp4')
    # per_p, per_n = predict_emo(video_name, multi_view)
    # emo_info1 = 'Positive (Happy, Neutral): {}%'.format(per_p)
    # emo_info2 = 'Negative (Angry, Sad): {}%'.format(per_n)
    emo_result = predict_emo(video_name, multi_view)
    emo_info = '{}'.format(emo_result)
    cv2.namedWindow("Emotion Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Emotion Detector', 550, 800) # change the video size to be shown
    videoCapture = cv2.VideoCapture(video_path)
    
    while True:  
            success, frame = videoCapture.read()
            if success:
                if win32gui.FindWindow(None,'Emotion Detector'):
                    x,y,w,h = 0,0,1000,60

                    # Draw black background rectangle
                    cv2.rectangle(frame, (x, x), (x + w, y + h), (128,128,128), -1)

                    #cv2.putText(frame,emo_info1,(x + int(w/10),y + int(h*1/3)),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.6,(255,255,255),1,cv2.LINE_AA)
                    #cv2.putText(frame,emo_info2,(x + int(w/10),y + int(h*2/3)),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.6,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(frame, emo_info, (x + int(w / 10), y + int(h * 1 / 3)), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Emotion Detector', frame)
                    
                else:  
                    videoCapture.release() 
                    cv2.destroyAllWindows()
                    break
            cv2.waitKey(int(1000/30))
            if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'): 
                videoCapture.release() 
                cv2.destroyAllWindows()
                break



if __name__ == '__main__':
    
    #_ = predict_emo('dataprox1') #param video_name, multi_view
    show_video(video_name = 'testb') #param video_name, multi_view
'''
