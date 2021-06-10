import numpy as np
import cv2

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_preds(hm, return_conf=False):
    assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
    h = hm.shape[2]
    w = hm.shape[3]
    hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
    idx = np.argmax(hm, axis=2)

    preds = np.zeros((hm.shape[0], hm.shape[1], 2))
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            preds[i, j, 0], preds[i, j, 1] = idx[i, j] % w, idx[i, j] / w
    if return_conf:
        conf = np.amax(hm, axis=2).reshape(hm.shape[0], hm.shape[1], 1)
        return preds, conf
    else:
        return preds


def get_preds_3d(heatmap, depthmap):
  output_res = max(heatmap.shape[2], heatmap.shape[3])
  preds = get_preds(heatmap).astype(np.int32)
  preds_3d = np.zeros((preds.shape[0], preds.shape[1], 3), dtype=np.float32)
  for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
      idx = min(j, depthmap.shape[1] - 1)
      pt = preds[i, j]
      preds_3d[i, j, 2] = depthmap[i, idx, pt[1], pt[0]]
      preds_3d[i, j, :2] = 1.0 * preds[i, j] / output_res
    preds_3d[i] = preds_3d[i] - preds_3d[i, 6:7]
  return preds_3d