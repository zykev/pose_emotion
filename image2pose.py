# This file is created by Zeyu Chen, BIAI, Inc.
# Title           :image2pose.py
# Version         :1.0
# Email           :k83110835@126.com
# Copyright       :BIAI, Inc.
#==============================================================================


import sys
import os
import torch
import torch.utils.data
import shutil

from lib.msra_resnet import get_pose_net
from lib.util import *

# from lib.models.msra_resnet import get_pose_net
# from lib.utils.image import get_affine_transform, transform_preds
# from lib.utils.eval import get_preds, get_preds_3d




def is_image(file_name):
    """This function detect whether a file is an image file."""
    image_ext = ['jpg', 'jpeg', 'png']
    ext = file_name[file_name.rfind('.') + 1:].lower()
    return ext in image_ext


def demo_image(image, model, opt):
    """
    This function extracts pose coordinates data from a given image based on posenet model.
    Args:
        image: image array
        model: posenet model
        opt: options setting
    """
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    s = max(image.shape[0], image.shape[1]) * 1.0
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    trans_input = get_affine_transform(
            c, s, 0, [opt.input_w, opt.input_h])
    inp = cv2.warpAffine(image, trans_input, (opt.input_w, opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp / 255. - mean) / std
    inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    inp = torch.from_numpy(inp).to(opt.device)
    out = model(inp)
    # pred = get_preds(out[0, :].detach().cpu().numpy())[0] #get predicted joints pixel position in the posenet output image (64x64)
    # pred = transform_preds(pred, c, s, (opt.output_w, opt.output_h)) #get predicted joints pixel position in the original image
    pred_3d = get_preds_3d(out[0, :].detach().cpu().numpy(),
                         out[1, :].detach().cpu().numpy())[0]
  

    return pred_3d


def image_select_check(image_dir_path):
    """
        This function checks whether the given video is valid or not. It will detect a video missing human completed body as invalid.
    """
    # 采用hot-svm进行人物检测
    defaultHog = cv2.HOGDescriptor()
    defaultHog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    init_image_idx = -1
    if os.path.isdir(image_dir_path):
        ls = os.listdir(image_dir_path)
        ls.sort(key=lambda x: int(x.split('_')[1][:-4]))
        for i in range(len(ls) - len(ls) % 10 - 30, 0, -10):
            ls_sub = ls[i:i+31]
            flag = 0
            for image_name in ls_sub:
                if is_image(image_name):
                    image_path = os.path.join(image_dir_path, image_name)
                    image = cv2.imread(image_path)
                    # image = cv2.resize(image,(256, 256))
                    (rects, weights) = defaultHog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.15)
                    # rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                    if len(rects) == 0:
                        flag = 1
                        break
            if flag == 0:
                init_image_idx = i
                break

    return init_image_idx

def load_pose_model(model_path):
  num_layers = 50
  heads = {'hm': 16, 'depth': 16}
  model = get_pose_net(num_layers, heads)

  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

  state_dict = checkpoint['state_dict']
  model.load_state_dict(state_dict, strict=False)

  return model

def image2pose(opt, video_name):
    """
    This function will change an image extracted from a video to pose coordinates data.
    """
    coords_result = []
    trans = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    # opt.heads['depth'] = opt.num_output
    #video_name = os.path.basename(opt.video_path).split('.')[0]
    if opt.load_model == '':
        opt.load_model = 'models/fusion_3d_var.pth'
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
    else:
        opt.device = torch.device('cpu')
    if opt.demo == '':
        opt.demo = os.path.join('images', video_name)




    # model, _, _ = create_model(opt)
    model = load_pose_model(opt.load_model)
    model = model.to(opt.device)
    model.eval()

    # check image qualification (human figure detection and proper initial image index)
    if opt.outstep:
        print('---Examing input images and finding images containing proper human walking pose.---')
    begin_img = image_select_check(opt.demo)
    #begin_img = image_check(opt.demo)
    if begin_img == -1:
        print('Invalid video! This video may not contain human or human may not appear in image center.')
        #delect image directory
        if os.path.exists(opt.demo):
            shutil.rmtree(opt.demo)
        sys.exit(1)

    if os.path.isdir(opt.demo):
        ls = os.listdir(opt.demo)
        ls.sort(key=lambda x: int(x.split('_')[1][:-4]))
        for i, image_name in enumerate(ls):
            if is_image(image_name):
                image_path = os.path.join(opt.demo, image_name)
                image = cv2.imread(image_path)
                if i in range(begin_img, begin_img + 31, 1):
                    pred_3d = demo_image(image, model, opt)
                    pred_3d_pros = pred_3d @ trans
                    coords_result.append(pred_3d_pros.reshape(1, -1))
                    if i % 10 == 0 and opt.outstep:
                        print('Detecting pose in {} etc. from {}'.format(image_name, opt.demo))

        coords_result = np.concatenate(coords_result)
        # np.savetxt(os.path.join('posedata', video_name + '_pose_coords.csv'), coords_result, delimiter=',')

    #delete image directory
    if os.path.exists(opt.demo):
        shutil.rmtree(opt.demo)


    elif is_image(opt.demo):
        #print('Detecting pose in {} ...'.format(opt.demo))
        image = cv2.imread(opt.demo)

        pred_3d = demo_image(image, model, opt)
        pred_3d_pros = pred_3d @ trans
        coords_result = pred_3d_pros.reshape(1, -1)
        # np.savetxt(os.path.join('posedata', video_name + '_pose_coords.csv'), coords_result, delimiter=',')

    return coords_result
    
'''
def main_folder(opt):
  trans = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
#  ffNew = h5py.File(os.path.join('E:/Project/posenet/data' , 'features_gait' + '.h5'), 'w')
  opt.heads['depth'] = opt.num_output
  if opt.load_model == '':
    opt.load_model = '../models/fusion_3d_var.pth'
  if opt.gpus[0] >= 0:
    opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
  else:
    opt.device = torch.device('cpu')
  if opt.demo_folder == '':
      opt.demo_folder = 'images/'
  
  model, _, _ = create_model(opt)
  model = model.to(opt.device)
  model.eval()
  
  if os.path.isdir(opt.demo_folder):
      ls_id = os.listdir(opt.demo_folder)
      for file_id_name in sorted(ls_id):

          file_id_path = os.path.join(opt.demo_folder, file_id_name)
          if os.path.isdir(file_id_path):
              coords_result = []
              for file_name in sorted(os.listdir(file_id_path)):
                  if is_image(file_name):
                      image_name = os.path.join(file_id_path, file_name)
                      print('Running {} ...'.format(image_name))
                      image = cv2.imread(image_name)
                      pred_3d = demo_image(image, model, opt)
                      
                      #save pred_3d
                      pred_3d_pros = pred_3d @ trans
                      coords_result.append(pred_3d_pros.reshape(1, -1))
              coords_result = np.concatenate(coords_result)
              np.savetxt(os.path.join('posedata', file_id_name + '_' + 'pose_coords.csv'),
                         coords_result, delimiter=',')
#              ffNew.create_dataset(name = 'gaitID' + '_' + str(file_id_name), data = coords_result)
              
#      ffNew.close()
      
'''
      
              
                
    
'''
if __name__ == '__main__':
  opt = opts().parse()
  image2pose(opt, 'testb') #param opt, video_name
  #main_folder(opt)
'''

