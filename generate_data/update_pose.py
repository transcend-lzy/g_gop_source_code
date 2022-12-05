import numpy as np
import os.path as osp
import sys
import os
from tqdm import trange
from trans_pose import abg2rt_final
from convert_pose_from_op import rotationMatrixToEulerAngles

cur_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(cur_dir)
from g_gop_code.utils import read_yaml, opt, mkdir

abg_range = read_yaml(osp.join(opt.data_path, 'abg_range.yml'))
Arange, Brange, Grange, Xrange, Yrange, Rrange = abg_range['Arange'][0], abg_range['Brange'][0], abg_range['Grange'][0], \
                                                 abg_range['Xrange'][0], abg_range['Yrange'][0], abg_range['Rrange'][0]
amin, amax = Arange[0], Arange[1]
bmin, bmax = Brange[0], Brange[1]
gmin, gmax = Grange[0], Grange[1]
xmin, xmax = Xrange[0], Xrange[1]
ymin, ymax = Yrange[0], Yrange[1]
rmin, rmax = Rrange[0], Rrange[1]


def norm_op(ori_path, dst_path):
    mkdir(dst_path)
    dirs = os.listdir(ori_path)
    if 'validation(640).npy' in dirs:
        dirs.remove('validation(640).npy')
        pose_npy = np.load(osp.join(ori_path, 'validation(640).npy'))
        for i in range(640):
            a, b, g, x, y, r = min_max(pose_npy[i])
            pose_npy[i][:6] = [a, b, g, x, y, r]
        np.save(osp.join(dst_path, 'validation(640).npy'), pose_npy)
    for i in trange(len(dirs)):
        start = i * 10000
        end = (i + 1) * 10000
        pose_npy = np.load(osp.join(ori_path, '{}-{}.npy').format(start, end))
        for i in range(10000):
            a, b, g, x, y, r = min_max(pose_npy[i])
            pose_npy[i][:6] = [a, b, g, x, y, r]
        np.save(osp.join(dst_path, '{}-{}.npy').format(start, end), pose_npy)


def min_max(pose):
    a = (pose[0] - amin) / (amax - amin)
    b = (pose[1] - bmin) / (bmax - bmin)
    g = (pose[2] - gmin) / (gmax - gmin)
    x = (pose[3] - xmin) / (xmax - xmin)
    y = (pose[4] - ymin) / (ymax - ymin)
    r = (pose[5] - rmin) / (rmax - rmin)
    return a, b, g, x, y, r


def from_abg_generate_rt(ori_path, dst_path):
    mkdir(dst_path)
    dirs = os.listdir(ori_path)
    if 'validation(640).npy' in dirs:
        dirs.remove('validation(640).npy')
        pose_npy = np.load(osp.join(ori_path, 'validation(640).npy'))
        for i in range(640):
            a, b, g, x, y, r = pose_npy[i][:6]
            m2c_r, m2c_t = abg2rt_final(a, b, g, x, y, r)
            rt_g, rt_b, rt_a = rotationMatrixToEulerAngles(m2c_r)[:3]
            rt_x, rt_y, rt_z = m2c_t[0][:3]
            pose_npy[i][:6] = [rt_a, rt_b, rt_g, rt_x, rt_y, rt_z]
        np.save(osp.join(dst_path, 'validation(640).npy'), pose_npy)
    for i in trange(len(dirs)):
        start = i * 10000
        end = (i + 1) * 10000
        pose_npy = np.load(osp.join(ori_path, '{}-{}.npy').format(start, end))
        for i in range(10000):
            a, b, g, x, y, r = pose_npy[i][:6]
            m2c_r, m2c_t = abg2rt_final(a, b, g, x, y, r)
            rt_g, rt_b, rt_a = rotationMatrixToEulerAngles(m2c_r)[:3]
            rt_x, rt_y, rt_z = m2c_t[0][:3]
            pose_npy[i][:6] = [rt_a, rt_b, rt_g, rt_x, rt_y, rt_z]
        np.save(osp.join(dst_path, '{}-{}.npy').format(start, end), pose_npy)


if __name__ == '__main__':
    pose_path = osp.join(opt.data_path, 'pose_set')
    norm_pose = osp.join(opt.data_path, 'pose_set_norm')
    rt_pose = osp.join(opt.data_path, 'pose_set_rt')
    from_abg_generate_rt(pose_path, rt_pose)
    norm_op(pose_path, norm_pose)