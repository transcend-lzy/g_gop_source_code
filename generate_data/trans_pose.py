'''
共有四个位姿表达方式互相转换：
1、worldOrientation, worldLocation ： matlab的pnp位姿结果形式
2、rotationMatrix, translationVector： matlab定义的m2c位姿形式
（worldOrientation, worldLocation做cameraPoseToExtrinsics转换的结果）
3、m2c_r, m2c_t : opencv python  pnp得到的结果，也是我们所熟知的m2c形式
4、abgxyr: 一种新型的位姿表示形式
'''
import time

import numpy as np
from math import cos, sin, asin, atan, pi
import matlab
import matlab.engine


def trans_pyrt_to_matrt2(r, t):
    """
    将py m2c的rt 转换为matlab m2c的rt
    Args:
        r,t: py m2c的rt，就是pnp计算出来的结果

    Returns:
        r,t: matlab m2c的rt，是matlab pnp计算出来的结果再做cameraPoseToExtrinsics转换的结果
    """
    matrix = [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]
    # 旋转矩阵先转置在前两列取反
    matr2 = np.dot(np.array(r, dtype=np.float32).T, np.array(matrix, dtype=np.float32))
    # 平移矩阵前两列取反
    matt2 = np.dot(np.multiply(np.array(t, dtype=np.float32), 1000), np.array(matrix, dtype=np.float32))
    return matr2, matt2


def trans_matr2_tomatr(r, t):
    """
    转换关系参考官网公式：https://ww2.mathworks.cn/help/vision/ref/cameraposetoextrinsics.html?requestedDomain=cn
    Args:
        r,t: matlab m2c的rt，是matlab pnp计算出来的结果再做cameraPoseToExtrinsics转换的结果

    Returns:
        r,t: matlab 的rt，是matlab pnp计算出来的结果
    """

    matr = np.array(r).T
    matr_inv = np.linalg.inv(r)
    matt = np.dot(np.array(-t), matr_inv)
    return matr, matt


def trans_matrt2_to_rt(r, t):
    """
    将rotationMatrix, translationVector 转换为m2c_r, m2c_t
    """
    matrix = [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]
    # 旋转矩阵先前两列取反再转置
    m2c_r = np.dot(np.array(r, dtype=np.float32), np.array(matrix, dtype=np.float32)).T
    # 平移向量前两列取反
    m2c_t = np.dot(np.array(t, dtype=np.float32), np.array(matrix, dtype=np.float32))
    return m2c_r, m2c_t


def abg2rt(a, b, g, x_trans, z_trans, r):
    """
    retunrn:
    worldOrientation, worldLocation: matlab pnp得到的结果
    rotationMatrix, translationVector： matlab pnp结果后cameraPoseToExtrinsic的结果
    m2c_r, m2c_t : python pnp的结果 正常的m2c结果
    """
    r_x = [[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]]
    r_y = [[cos(b), 0, sin(b)], [0, 1, 0], [-sin(b), 0, cos(b)]]
    r_z = [[cos(g), -sin(g), 0], [sin(g), cos(g), 0], [0, 0, 1]]

    bag = np.matmul(np.matmul(r_y, r_x), r_z)  # np.matmul是做矩阵相乘

    rm = bag

    xx = np.array([rm[0, 0], rm[1, 0], rm[2, 0]])
    yy = np.array([rm[0, 1], rm[1, 1], rm[2, 1]])
    zz = np.array([rm[0, 2], rm[1, 2], rm[2, 2]])

    x = r * -zz[0]
    y = r * -zz[1]
    z = r * -zz[2]

    pos = np.array([x, y, z]) + x_trans * xx + z_trans * yy  # T 3×1

    worldOrientation = rm.T
    worldLocation = pos  # 平移矩阵

    rotationMatrix = worldOrientation.T
    translationVector = -np.matmul(worldLocation, worldOrientation.T)
    m2c_r, m2c_t = trans_matrt2_to_rt(rotationMatrix, translationVector)

    return worldOrientation, worldLocation, rotationMatrix, translationVector, m2c_r, m2c_t


def rt2abg(m2c_r, m2c_t, eng):
    """
    将m2c_r, m2c_t 转换为 abgxyr
    """
    rotationMatrix, translationVector = trans_pyrt_to_matrt2(m2c_r, m2c_t)
    # 先转换成worldOrientation, worldLocation
    worldOrientation, worldLocation = trans_matr2_tomatr(rotationMatrix, translationVector)
    # 做一次转换，matlab不允许直接使用np
    r = matlab.double(np.array(worldOrientation).tolist())
    t = matlab.double(worldLocation.tolist())
    # 调用matlab函数，文件夹中要有一个同名同名rt2abg.m 里面写了function为rt2abg
    # eng = matlab.engine.start_matlab()
    # 输入的rt 是worldOrientation, worldLocation
    ret = eng.rt2abg(r, t[0])
    return ret[0]


def check_abg(orientation, orientation_new):
    for i in range(3):
        for j in range(3):
            if orientation[i][j] * orientation_new[i][j] <= 0:
                return False
    return True


def rt2abg_new(m2c_r, m2c_t):
    '''
    重写matlab能得到一样的结果
    :param m2c_r:
    :param m2c_t:
    :return:
    '''
    rotationMatrix, translationVector = trans_pyrt_to_matrt2(m2c_r, m2c_t)
    worldOrientation, worldLocation = trans_matr2_tomatr(rotationMatrix, translationVector)
    R = np.transpose(worldOrientation)
    T = worldLocation
    a = asin(-R[1][2])
    g = atan(R[1][0] / R[1][1])
    b = atan(R[0][2] / R[2][2])
    abg_list = [[a, pi - a], [b, b + pi], [g, g + pi]]
    for i in range(8):
        binary = bin(i).replace("0b", "")
        if len(binary) < 3:
            binary = binary.rjust(3, '0')
        a_cur = abg_list[0][int(binary[2])]
        b_cur = abg_list[1][int(binary[1])]
        g_cur = abg_list[2][int(binary[0])]
        r_x = [[1, 0, 0], [0, cos(a_cur), -sin(a_cur)], [0, sin(a_cur), cos(a_cur)]]
        r_y = [[cos(b_cur), 0, sin(b_cur)], [0, 1, 0], [-sin(b_cur), 0, cos(b_cur)]]
        r_z = [[cos(g_cur), -sin(g_cur), 0], [sin(g_cur), cos(g_cur), 0], [0, 0, 1]]
        bag = np.matmul(np.matmul(r_y, r_x), r_z)  # np.matmul是做矩阵相乘
        # print(a_cur ,b_cur, g_cur)
        if check_abg(worldOrientation, np.transpose(bag)):
            print(a_cur, b_cur, g_cur)
            a = a_cur
            b = b_cur
            g = g_cur
            break
    zz = [bag[0][2], bag[1][2], bag[2][2]]
    xx = [bag[0][0], bag[1][0], bag[2][0]]
    yy = [bag[0][1], bag[1][1], bag[2][1]]
    b = np.transpose(T)
    r = np.linalg.solve(bag, b)
    return a, b, g, r[0], r[1], -r[2]


def rt2abg_final(m2c_r, m2c_t):
    '''
    自己重写的python版本的rt2abg
    '''
    m2c_r = np.array(m2c_r)
    m2c_t = np.array(m2c_t)
    m2c_rt = np.concatenate((np.concatenate((m2c_r, m2c_t.T), 1), np.array([[0, 0, 0, 1]])), 0)
    c2m_rt = np.linalg.inv(m2c_rt)
    c2m_r = c2m_rt[:3, :3]
    r_x_pi = [[1, 0, 0], [0, cos(pi), -sin(pi)], [0, sin(pi), cos(pi)]]
    c2m_r_new = np.matmul(c2m_r, r_x_pi)
    c2m_t = c2m_rt[:3, 3]
    b = asin(-c2m_r_new[1][2])
    g = atan(c2m_r_new[1][0] / c2m_r_new[1][1])
    a = atan(c2m_r_new[0][2] / c2m_r_new[2][2])
    abg_list = [[b, pi - b], [a, a + pi], [g, g + pi]]
    bag = 0
    for i in range(8):
        binary = bin(i).replace("0b", "")
        if len(binary) < 3:
            binary = binary.rjust(3, '0')
        a_cur = abg_list[1][int(binary[2])]
        b_cur = abg_list[0][int(binary[1])]
        g_cur = abg_list[2][int(binary[0])]
        r_x = [[1, 0, 0], [0, cos(b_cur), -sin(b_cur)], [0, sin(b_cur), cos(b_cur)]]
        r_y = [[cos(a_cur), 0, sin(a_cur)], [0, 1, 0], [-sin(a_cur), 0, cos(a_cur)]]
        r_z = [[cos(g_cur), -sin(g_cur), 0], [sin(g_cur), cos(g_cur), 0], [0, 0, 1]]
        bag = np.matmul(np.matmul(r_y, r_x), r_z)  # np.matmul是做矩阵相乘
        # print(a_cur ,b_cur, g_cur)
        if check_abg(c2m_r_new, bag):
            a = a_cur
            b = b_cur
            g = g_cur
            break
    r = np.linalg.solve(bag, np.transpose(c2m_t))
    if a < 0:
        a += 2 * pi
    if b < 0:
        b += 2 * pi
    if g <0:
        g += 2 * pi
    return float(a), float(b), float(g), float(r[0]), float(r[1]), float(r[2])


def abg2rt_final(a, b, g, x, y, r):
    '''
    自己重写的python版本abg2rt
    '''
    r_x = [[1, 0, 0], [0, cos(b), -sin(b)], [0, sin(b), cos(b)]]
    r_y = [[cos(a), 0, sin(a)], [0, 1, 0], [-sin(a), 0, cos(a)]]
    r_z = [[cos(g), -sin(g), 0], [sin(g), cos(g), 0], [0, 0, 1]]
    r_x_pi = [[1, 0, 0], [0, cos(pi), -sin(pi)], [0, sin(pi), cos(pi)]]
    # np.matmul(np.matmul(r_y, r_x), r_z) 得到的是相机坐标系z轴沿透镜向内的坐标系，实际的坐标系应该是z轴沿着透镜向外，即绕x轴旋转180度
    c2m_r = np.matmul(np.matmul(r_y, r_x), r_z)
    c2m_t = np.array([c2m_r[:3, 2] * r + c2m_r[:3, 0] * x + c2m_r[:3, 1] * y])
    c2m_r_new = np.matmul(c2m_r, r_x_pi)
    c2m_rt = np.concatenate((np.concatenate((c2m_r_new, c2m_t.T), 1), np.array([[0, 0, 0, 1]])), 0)
    m2c_rt = np.linalg.inv(c2m_rt)
    m2c_r = m2c_rt[:3, :3]
    m2c_t = m2c_rt[:3, 3].T
    return m2c_r, np.array([m2c_t])


if __name__ == '__main__':
    m2c_r_ori = [[-0.99779670358243, 0.04718226924468343, 0.04664302508369422],
             [0.04801818296067178, 0.9987023642693351, 0.016965898384567264],
             [-0.04578200984220074, 0.01916823079374373, -0.998767533765013]]
    m2c_t_ori = [[-0.0479586227422524, -0.04788719731996293, 0.7677722093310708]]
    a, b, g, x, y, r = rt2abg_final(m2c_r_ori, m2c_t_ori)
    m2c_r_ori, m2c_t_ori = abg2rt_final(a, b, g, x, y, r)
    print(m2c_r_ori, m2c_t_ori)
