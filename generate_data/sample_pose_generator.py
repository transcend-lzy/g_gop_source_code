import os
import random
import argparse
import numpy as np
from utils import *
from math import pi, sqrt, cos, sin, asin
from trans_pose import abg2rt_final


class GenPose(object):

    def __init__(self, model_index, set_length, all_set):
        dir_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = osp.join(dir_name, 'data')
        self.modelIndex = str(model_index)
        self.obj_data_path = osp.join(self.data_path, self.modelIndex)
        self.pose_path = osp.join(self.obj_data_path, 'pose_set')
        mkdir(self.pose_path)
        # 中间截取图像的生成范围大小，这个要根据标签提供的bbox提前确定，测试集中目标的bbox最大值不能超过这个范围
        self.bbox_length = read_yaml(osp.join(self.obj_data_path,'obj_bbox_length.yml'))[self.modelIndex]
        self.fibPoints = np.load(osp.join(self.data_path, 'fibonacci.npy'))
        self.special_points = read_yaml(osp.join(self.data_path, 'special_points.yml'))[self.modelIndex]
        self.sample_num = 10000000  # 生成fibonacci球的点集点的数量
        self.ox = ox  # 相机内参
        self.oy = oy  # 相机内参
        self.FocalLength_x = FocalLength_x  # 相机内参
        self.FocalLength_y = FocalLength_y  # 相机内参
        self.K = np.array([[self.FocalLength_x, 0, self.ox], [0, self.FocalLength_y, self.oy], [0, 0, 1]])
        dir_name = os.path.dirname(os.path.abspath(__file__))
        self.stl_path = osp.join(dir_name, 'CADmodels','stl','{}.stl'.format(self.modelIndex))
        self.height = 2048  # 图片长
        self.width = 2448  # 图片宽
        self.img_range = read_yaml(osp.join(self.obj_data_path, 'obj_gop_range.yml'))[self.modelIndex]
        self.uRangeLeft = max(0, self.img_range[0] - 200)
        self.uRangeRight = min(self.width, self.img_range[1] + 200)
        self.vRangeTop = max(0, self.img_range[2] - 200)
        self.vRangeBottom = min(self.height, self.img_range[3] + 200)
        self.ranges = read_yaml(osp.join(self.obj_data_path, 'abg_range.yml'))
        self.Arange = self.ranges['Arange']
        self.Brange = self.ranges['Brange']
        self.Grange = self.ranges['Grange']
        self.Xrange = self.ranges['Xrange']
        self.Yrange = self.ranges['Yrange']
        self.Rrange = self.ranges['Rrange']
        self.offset = 6 * (self.bbox_length // 128)
        self.set_length = set_length
        self.all_set = all_set

    def fibonacci_sphere(self):
        points = []
        phi = pi * (3. - sqrt(5.))  # golden angle in radians
        for i in range(self.sample_num):
            y = 1 - (i / float(self.sample_num - 1)) * 2  # y goes from 1 to -1
            radius = sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = cos(theta) * radius
            z = sin(theta) * radius
            if y > 0:
                points.append([x, y, z])
        return points

    def estimate_3D_to_2D(self, a, b, g, x_trans, z_trans, r, points):
        points = np.array(points, dtype=np.float64).reshape(-1, 3)
        m2c_r, m2c_t = abg2rt_final(a, b, g, x_trans, z_trans, r)
        m2c_t = np.matmul(m2c_t, 1000 * np.identity(3))
        points_2d, _ = cv2.projectPoints(points, m2c_r, m2c_t, self.K, None)

        return points_2d

    def random_gen(self):
        range_g = self.Grange[random.randint(0, len(self.Grange) - 1)]
        range_x = self.Xrange[random.randint(0, len(self.Xrange) - 1)]
        range_y = self.Yrange[random.randint(0, len(self.Yrange) - 1)]
        range_r = self.Rrange[random.randint(0, len(self.Rrange) - 1)]
        g = np.random.uniform(range_g[0], range_g[1])
        x = np.random.uniform(range_x[0], range_x[1])
        y = np.random.uniform(range_y[0], range_y[1])
        r = np.random.uniform(range_r[0], range_r[1])
        return g, x, y, r

    def val_ab(self, a, b):
        flagA = False
        flagB = False
        for range_a in self.Arange:
            if range_a[0]-0.01 <= a <= range_a[1] + 0.01:
                flagA = True
                break
        if not flagA:
            return False
        for range_b in self.Brange:
            if range_b[0]-0.01 <= b <= range_b[1] + 0.01:
                flagB = True
        return flagB

    def dot_generator(self):
        while True:
            while True:  # 这里while true做的就是把xyz转化为了ab
                index = np.random.randint(len(self.fibPoints))  # sample_num是斐波那契球的全部数量

                x = self.fibPoints[index][0]
                y = self.fibPoints[index][1]
                z = self.fibPoints[index][2]
                # 坐标系是y轴朝上的右手坐标系、
                # 计算beta
                beta = asin(abs(y))
                if beta > pi / 2.:
                    beta = pi - beta
                L = abs(cos(beta))
                if y < 0:
                    if z < 0:
                        beta += pi / 2.
                else:
                    if z < 0:
                        beta += pi
                    else:
                        beta += pi / 2. * 3

                # 计算 alpha
                alpha = asin(abs(x) / L)
                if alpha > pi / 2.:
                    alpha = pi - alpha
                if x < 0:
                    if z < 0:
                        alpha += pi / 2.
                else:
                    if z < 0:
                        alpha += pi
                    else:
                        alpha += pi / 2. * 3

                if self.val_ab(alpha, beta):
                    a = alpha
                    b = beta
                    break

            g, x, y, r = self.random_gen()

            spe_points2D = self.estimate_3D_to_2D(a, b, g, x, y, r, self.special_points)

            if_window_in = True
            # 判定生成图像的八个最远点是不是都在图像上，要保证生成图像的完整性
            for k in range(np.size(spe_points2D, 0)):
                u = spe_points2D[k, 0, 0]
                v = spe_points2D[k, 0, 1]

                if u < self.uRangeLeft or u >= self.uRangeRight or v < self.vRangeTop or v >= self.vRangeBottom:
                    if_window_in = False
                    break

            center = self.estimate_3D_to_2D(a, b, g, x, y, r, [[0, 0, 0]])
            center = center[0][0]
            # 零件正好在矩形框中心，这时候的矩形框中心点像素坐标
            center_x, center_y = int(np.round(center[0])), int(np.round(center[1]))
            centerRangeLeft = self.uRangeLeft + self.bbox_length // 2
            centerRangeRight = self.uRangeRight - self.bbox_length // 2
            centerRangeTop = self.vRangeTop + self.bbox_length // 2
            centerRangeBottom = self.vRangeBottom - self.bbox_length // 2
            if center_x <= centerRangeLeft or center_x >= centerRangeRight or center_y <= centerRangeTop or \
                    center_y >= centerRangeBottom:
                if_window_in = False

            if if_window_in:
                trans_x = np.random.randint(-self.offset, self.offset + 1)  # 这个-self.offset到self.offset + 1是偏移量
                trans_y = np.random.randint(-self.offset, self.offset + 1)
                u = center_x - self.bbox_length // 2 + trans_x
                v = center_y - self.bbox_length // 2 + trans_y
                return [a, b, g, x, y, r, center_x, center_y, trans_x, trans_y, u, v]
            else:
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_val", type=bool, default=True)
    parser.add_argument("--obj_id", type=str, default='6')
    parser.add_argument("--set_length", type=int, default=10000)
    parser.add_argument("--all_set", type=int, default=64)
    args = parser.parse_args()
    pose_gen = GenPose(int(args.obj_id), args.set_length, args.all_set)
    start_index = 0
    sample_set = []
    while True:
        pose = pose_gen.dot_generator()
        sample_set.append(pose)
        if args.is_val:
            if len(sample_set) == 640:
                name = 'validation(640)'
                np.save(osp.join(pose_gen.pose_path, name), sample_set)
                break
        start_index = start_index + 1
        if start_index % pose_gen.set_length == 0:
            name = str(start_index - pose_gen.set_length) + '-' + str(start_index)
            print(name)
            if not osp.exists(pose_gen.pose_path):
                os.mkdir(pose_gen.pose_path)
            np.save(osp.join(pose_gen.pose_path, name), sample_set)
            sample_set = []
            sample_set_img = []
        if start_index == pose_gen.set_length * pose_gen.all_set:
            break