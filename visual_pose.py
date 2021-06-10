# -*- coding: utf-8 -*-

# This file is created by Zeyu Chen, BIAI, Inc.
# Title           :visual_pose.py
# Version         :1.0
# Email           :k83110835@126.com
# Copyright       :BIAI, Inc.
#==============================================================================


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import pandas as pd

from show_emo import load_pose_data
  

def animation_pose(video_name, pose_coords, angle_idx = 0):
    """
    This function will create an animation plot based on the coordinates position of different joints to visualize people's walking pose.
    Args:
        video_name: The name of the video.
        angle_idx: The angle to be analized. (0: back; 1: right; 2: front; 3: left)
    """
    # pose_coords = data_prox[1,:,:]
    # pose_coords = pose_coords.reshape(1, pose_coords.shape[0], pose_coords.shape[1]) #shape 1*T*(J*C)
    # pose_coords = pose_multi_view(pose_coords)
    
    # edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
    #                   [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
    #                   [6, 7], [7, 8], [8, 9]] #posenet trained joints connection

    # pose_coords = load_data(True, video_name) #param multi_view, video_name
    pose_coords = load_pose_data(pose_coords)
    pose_coords = pose_coords[:,0:31,:]
    edges = [[15, 14], [14, 13], [13, 0], [0, 10], [10, 11], [11, 12], [6, 5], [5, 4], [4, 2],
             [2, 7], [7, 8], [8, 9], [0, 1], [1, 2], [2, 3]] #posenet trained joints connection(if input training data in proxemo)

    
    def update_graph(num, edges, graph2):
        data=df_draw[df_draw['time']==num]
        data = np.array(data)
        graph1._offsets3d = (data[:,1], data[:,2], data[:,3])
        for e, line in zip(edges, graph2):
            
            line.set_data(data[e,1:3].T)
            line.set_3d_properties(data[e,3])
        title.set_text('Pose Visualization, time={}'.format(num))
        return graph1, graph2
    
    # load posedata for visualization    
    points = []
    for t in range(pose_coords.shape[1]):
        points.append(pose_coords[angle_idx,t,:].reshape(-1,3))
    points = np.concatenate(points)
    time = np.array([np.ones(16)*i for i in range(pose_coords.shape[1])]).flatten()
    df_draw = pd.DataFrame({"time": time ,"x" : points[:,0], "y" : points[:,2], "z" : points[:,1]})
        
    max_range = np.array([df_draw.x.max()-df_draw.x.min(), df_draw.y.max()-df_draw.y.min(), df_draw.z.max()-df_draw.z.min()]).max() / 2.0
    mid_x = np.mean([df_draw.x.max(), df_draw.x.min()])
    mid_y = np.mean([df_draw.y.max(), df_draw.y.min()])
    mid_z = np.mean([df_draw.z.max(), df_draw.z.min()])
    
    
    
    fig = plt.figure(figsize=(6,6))
    #fig.canvas.manager.window.move(0, 300)   # 调整窗口在屏幕上弹出的位置
    ax = fig.add_subplot((111), projection='3d')
    ax.grid(False)
    title = ax.set_title('Pose Visualization')
    ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
    ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
    ax.set_zlim3d(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('x', labelpad=10)
    ax.set_ylabel('y', labelpad=8)
    ax.set_zlabel('z', labelpad=2)
    
    
    data=df_draw[df_draw['time']==0]
    data = np.array(data)
    
    graph1 = ax.scatter(data[:,1], data[:,2], data[:,3], s = 100, c = 'dimgray', marker = 'o')
    
    graph2 = [ax.plot(data[e,1], data[e,2], data[e,3], c = 'red', linewidth=4, alpha=0.5)[0] for e in edges]
    
    
    
    ani = animation.FuncAnimation(fig, update_graph, np.arange(pose_coords.shape[1]), fargs = (edges, graph2), 
                                  interval=int(1000/5), blit=False)
    
    ani.save('./posedata/' + video_name + '.gif', writer = 'pillow')
    return ani

 

'''
def visual_single_pose(video_name, t=0):
    """
    This function will create a static plot on people's walking pose for four angles.
    Args:
        video_name: The name of the video
        t: time step
    """
    
    pose_coords = load_data(True, video_name)   #param multi_view, video_name 
    edges = [[15, 14], [14, 13], [13, 0], [0, 10], [10, 11], [11, 12], [6, 5], [5, 4], [4, 2],
             [2, 7], [7, 8], [8, 9], [0, 1], [1, 2], [2, 3]] 
    #posenet trained joints connection(if input training data in proxemo)
    plot_pos = [221, 222, 223, 224] # plot position for subplots
    fig = plt.figure(figsize=(10,10))
    for i in range(4): # four angles
    
        points = pose_coords[i,t,:].reshape(-1,3)
        x = points[:, 0]
        y = points[:, 2]
        z = points[:, 1]

        ax = fig.add_subplot(plot_pos[i], projection='3d')
        ax.grid(False)
        
        ax.scatter(x, y, z, s = 100, c = 'dimgray', marker = 'o')
        
        for e in edges:
        
            ax.plot(x[e], y[e], z[e], c = 'red', linewidth=4, alpha=0.5)
        
           
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        
        ax.set_title('Pose Visualization time:{}'.format(t))
        ax.set_xlabel('x', labelpad=10)
        ax.set_ylabel('y', labelpad=8)
        ax.set_zlabel('z', labelpad=2)
        #ax.set_xticklabels(labels=x, rotation=90)
        #ax.set_yticklabels(labels=y, rotation=90)
        #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        
    plt.show()
    




if __name__ == '__main__':
    
    video = 'test_video43'
    out_ani = animation_pose(video,2)  
    plt.show()
    # visual_single_pose(video,33)  #param video_name, t
'''