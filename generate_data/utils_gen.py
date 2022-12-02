import yaml
import cv2
import os
import os.path as osp
import numpy as np
import math

ox = 1239.951701787861  # 相机内参
oy = 1023.50823945964  # 相机内参
FocalLength_x = 2344.665256639324  # 相机内参
FocalLength_y = 2344.050648508343  # 相机内参
data_path = '..data'


def mkdir(path):
    if not osp.exists(path):
        os.makedirs(path)


def read_yaml(yaml_path):
    f = open(yaml_path, 'r', encoding='utf-8')
    cfg = f.read()
    dic = yaml.safe_load(cfg)
    return dic


def show_photo(photo):  # 展示照片
    if photo.shape[0] > 1500:
        photo = cv2.resize(photo, (int(2448.0 // 2), int(2048.0 // 2)))
    cv2.imshow("abc", photo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def eulerAnglesToRotationMatrix(theta):
    # 分别构建三个轴对应的旋转矩阵
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    # 将三个矩阵相乘，得到最终的旋转矩阵
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def isRotationMatrix(R):
    # 得到该矩阵的转置
    Rt = np.transpose(R)
    # 旋转矩阵的一个性质是，相乘后为单位阵
    shouldBeIdentity = np.dot(Rt, R)
    # 构建一个三维单位阵
    I = np.identity(3, dtype=R.dtype)
    # 将单位阵和旋转矩阵相乘后的值做差
    n = np.linalg.norm(I - shouldBeIdentity)
    # 如果小于一个极小值，则表示该矩阵为旋转矩阵
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    # 断言判断是否为有效的旋转矩阵
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([z, y, x])