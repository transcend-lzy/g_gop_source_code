from trans_pose import abg2rt_final
import numpy as np
import math


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


if __name__ == '__main__':
    pose_npy = np.load('../data/6/pose_set/0-10000.npy')
    for i in range(10000):
        a, b, g, x, y, r = pose_npy[i][:6]
        m2c_r, m2c_t = abg2rt_final(a, b, g, x, y, r)
        rt_g, rt_b, rt_a = rotationMatrixToEulerAngles(m2c_r)[:3]
        rt_x, rt_y, rt_z = m2c_t[0][:3]
        pose_npy[i][:6] = [rt_a, rt_b, rt_g, rt_x, rt_y, rt_z]
    np.save('../data/6/pose_set/0-10000_new.npy', pose_npy)
