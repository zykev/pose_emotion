# This file is taken from Github project 'Pytorch-pose-hg-3d' and modified by Zeyu Chen, BIAI Inc.
# Github link     :https://github.com/xingyizhou/pytorch-pose-hg-3d
# Paper           :Towards 3D Human Pose Estimation in the Wild: A Weakly-Supervised Approach
#                  The IEEE International Conference on Computer Vision (ICCV)
# Title           :opts.py
# Original Author :Zhou, Xingyi and Huang, Qixing and Sun, Xiao and Xue, Xiangyang and Wei, Yichen
# Github          :"2020, PyTorch implementation for 3D human pose estimation"
# Version         :1.0
# Email           :zhouxy2017@gmail.com
#==============================================================================


import argparse

class opts():
  def __init__(self):
    self.parser = argparse.ArgumentParser()


    self.parser.add_argument('--gpus', default='-1', help='-1 for CPU')
    self.parser.add_argument('--demo', default='', help='path/to/image')
    self.parser.add_argument('--demo_folder', default='', help='path/to/image_directory')
    self.parser.add_argument('--video_path', help='path/to/video file', required=True)
    self.parser.add_argument('--load_model', default='')

    self.parser.add_argument('--input_h', type=int, default=256)
    self.parser.add_argument('--input_w', type=int, default=256)
    self.parser.add_argument('--output_h', type=int, default=64)
    self.parser.add_argument('--output_w', type=int, default=64)

    self.parser.add_argument('--outstep', type=bool, default=False, help='output middle steps when processing')
    self.parser.add_argument('--vidmode', type=str, choices=['vertical', 'horizontal'], default='vertical', help='the direction when taking the video')

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)
    

    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]

    return opt
