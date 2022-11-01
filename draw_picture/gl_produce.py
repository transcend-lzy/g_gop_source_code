import pygame
from pygame.locals import *
import cv2
import scipy.misc
from math import pi, cos, sin
# import glm

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np

from read_stl import stl_model
# 从坐标轴方向往内看，各轴旋转角方向如下：
# 绕y轴顺时针旋转为正
# 绕z轴顺时针旋转为正
# 绕x轴逆时针旋转为正
# 原因：正常来说应该都是逆时针为正，但因为物理相机坐标系和opengl规定的相机坐标系有不同（绕x轴旋转180°，所以yz轴旋转角度都取反）
path = './binary.stl'
file = stl_model(path)
tri = file.tri


def cube():
    glBegin(GL_TRIANGLES)

    for Tri in tri:
        glColor3fv(Tri['colors'])
        glVertex3fv(
            (Tri['p0'][0], Tri['p0'][1], Tri['p0'][2]))
        glVertex3fv(
            (Tri['p1'][0], Tri['p1'][1], Tri['p1'][2]))
        glVertex3fv(
            (Tri['p2'][0], Tri['p2'][1], Tri['p2'][2]))

    glEnd()


def grid():
    glBegin(GL_LINES)
    glColor3fv((0 / 255., 0 / 255., 255 / 255.))
    glVertex3fv((0., 0., 0.))
    glVertex3fv((0., 0., 1.))

    glColor3fv((255 / 255., 0 / 255., 0 / 255.))
    glVertex3fv((0., 0., 0.))
    glVertex3fv((1., 0., 0.))

    glColor3fv((0 / 255., 255 / 255., 0 / 255.))
    glVertex3fv((0., 0., 0.))
    glVertex3fv((0., 1., 0.))

    # glColor3fv((0 / 255., 255 / 255., 0/ 255.))
    # glVertex3fv((0., 1., 0.))
    # glColor3fv((0 / 255., 0 / 255., 0 / 255.))
    # glVertex3fv((1, 0., 0.))
    # glColor3fv((201 / 255., 202 / 255., 202 / 255.))
    # Range = [-.1, .1]
    #
    # inter = 0.01
    #
    # num = int((Range[1] - Range[0]) / 0.01)
    #
    # for i in range(num):
    #     m = Range[0] + (Range[1] - Range[0]) * (i / float(num))
    #
    #     glVertex3fv((m, 0, Range[0]))
    #     glVertex3fv((m, 0, Range[1]))
    #
    # for i in range(num):
    #     m = Range[0] + (Range[1] - Range[0]) * (i / float(num))
    #
    #     glVertex3fv((Range[0], 0, m))
    #     glVertex3fv((Range[1], 0, m))

    glEnd()

    # glBegin(GL_TRIANGLES)
    #
    # glColor3fv((238 / 255., 238 / 255., 239 / 255.))
    #
    # glVertex3fv((Range[1], -0.0001, Range[1]))
    # glVertex3fv((Range[1], -0.0001, Range[0]))
    # glVertex3fv((Range[0], -0.0001, Range[0]))
    #
    # glVertex3fv((Range[1], -0.0001, Range[1]))
    # glVertex3fv((Range[0], -0.0001, Range[1]))
    # glVertex3fv((Range[0], -0.0001, Range[0]))
    #
    # glEnd()


pygame.init()

L = 640 * 2
H = 480 * 2
display = (L, H)
window = pygame.display.set_mode(display, DOUBLEBUF | OPENGLBLIT | OPENGL)

L = 640
H = 480
ox = 324
oy = 263
FocalLength_x = 594
FocalLength_y = 594


def abg2rt_final(a, b, g, x, y, r):
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


def draw_cube(alpha=0., beta=0., gama=0., x_trans=0, z_trans=0, radius=0.3):
    m2c_r, m2c_t = abg2rt_final(alpha, beta, gama, x_trans, z_trans, radius)
    m2c_rt = np.concatenate((np.concatenate((m2c_r, m2c_t.T), 1), np.array([[0, 0, 0, 1]])), 0)
    c2m_rt = np.linalg.inv(m2c_rt)
    pos = c2m_rt[0:3, 3]
    obj = pos + c2m_rt[0:3, 2]
    yy = c2m_rt[0:3, 1]
    gluLookAt(pos[0], pos[1], pos[2], obj[0],obj[1],obj[2], -yy[0], -yy[1], -yy[2])

    cube()
    grid()


def draw_cube_old(alpha=0., beta=0., gama=0., x_trans=0, z_trans=0, radius=0.3):
    r = radius

    a = alpha
    b = beta
    g = gama

    r_x = [[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]]
    r_y = [[cos(b), 0, sin(b)], [0, 1, 0], [-sin(b), 0, cos(b)]]
    r_z = [[cos(g), -sin(g), 0], [sin(g), cos(g), 0], [0, 0, 1]]

    bag = np.matmul(np.matmul(r_y, r_x), r_z)
    rm = bag

    xx = np.array([rm[0, 0], rm[1, 0], rm[2, 0]])
    yy = np.array([rm[0, 1], rm[1, 1], rm[2, 1]])
    zz = np.array([rm[0, 2], rm[1, 2], rm[2, 2]])

    x = r * -zz[0]
    y = r * -zz[1]
    z = r * -zz[2]

    pos = np.array([x, y, z]) + x_trans * xx + z_trans * yy

    obj = pos + zz
    gluLookAt(pos[0], pos[1], pos[2], obj[0], obj[1], obj[2], yy[0], yy[1], yy[2])

    cube()
    grid()


a = 0
b = 0
g = 0

r = 400. / 1000
x = 0
z = 0

pic_index = 2

# ppp = [ 5.44608792e-01,  4.81379207e+00,  7.73520324e-01, -3.14899458e-02,
#   -4.30604975e-02,  4.00000000e-01]

# a,b,g,x,z,r = ppp

move_dis = 0.01
rota_dis = pi / 8

# move_dis = 0.025
# rota_dis = pi/32
if __name__ == '__main__':
    while True:
        # opengl中要对模型进行操作，要对这个模型的状态乘上这个操作对应的矩阵，
        # GL_PROJECTION 表示投影变换，即与之相乘的是投影矩阵
        glMatrixMode(GL_PROJECTION)
        # 把矩阵设为单位矩阵
        glLoadIdentity()
        # 每次重绘之前，需要先清除屏幕及深度缓存，一般放在绘图函数开头
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # 设备画布背景色
        glClearColor(1, 1, 1, 0.0)
        scale = 0.0001
        # intrinsic parameters
        # glFrustum用来设置透视投影(6个参数，前4个是坐标，后2个是距离)   glOrtho用来设置平行投影
        # 这两个函数的参数相同，都是视景体的 left / right / bottom / top / near / far 六个面
        # 视景体的 left / right / bottom / top 四个面围成的矩形，就是视口。near 就是投影面，其值是投影面距离视点的距离，
        # far 是视景体的后截面，其值是后截面距离视点的距离。far 和 near 的差值，就是视景体的深度。
        # 视点和视景体的相对位置关系是固定的，视点移动时，视景体也随之移动。
        # 由glFrustum指定的矩阵会与单位矩阵相乘，生成透视效果
        near = 0.0001
        far = 100
        left = -ox * near / FocalLength_x
        right = (L - ox) * near / FocalLength_x
        top = oy * near / FocalLength_y
        bottom = (oy - H) * near / FocalLength_y
        # glFrustum(-ox * scale, (L - ox) * scale, -(H - oy) * scale, oy * scale,
        #           0.1, 100)
        glFrustum(left, right, bottom, top, near, far)
        glMatrixMode(GL_MODELVIEW)
        glClearDepth(1.0)

        # 开启深度测试，实现遮挡关系
        glEnable(GL_DEPTH_TEST)
        # 设置深度测试函数
        glDepthFunc(GL_LESS)
        glEnable(GL_POINT_SMOOTH)
        glPolygonMode(GL_FRONT, GL_FILL)
        glPolygonMode(GL_BACK, GL_FILL)

        glPushMatrix()

        im = draw_cube(a, b, g, x, z, r)

        glPopMatrix()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # glOrtho(-320, 320, -240, 240, -200.0, 200.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        pygame.display.flip()

        event = pygame.event.poll()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break
        elif event.type == KEYDOWN and event.key == K_w:
            z = z - move_dis
        elif event.type == KEYDOWN and event.key == K_s:
            z = z + move_dis
        elif event.type == KEYDOWN and event.key == K_a:
            x = x - move_dis
        elif event.type == KEYDOWN and event.key == K_d:
            x = x + move_dis
        elif event.type == KEYDOWN and event.key == K_q:
            r = r - move_dis
        elif event.type == KEYDOWN and event.key == K_e:
            r = r + move_dis

        # special
        elif event.type == KEYDOWN and event.key == K_i:
            r = r / 1.2
        elif event.type == KEYDOWN and event.key == K_o:
            r = r * 1.2

        elif event.type == KEYDOWN and event.key == K_t:
            a = a - rota_dis
        elif event.type == KEYDOWN and event.key == K_g:
            a = a + rota_dis
        elif event.type == KEYDOWN and event.key == K_f:
            b = b - rota_dis
        elif event.type == KEYDOWN and event.key == K_h:
            b = b + rota_dis
        elif event.type == KEYDOWN and event.key == K_r:
            g = g - rota_dis
        elif event.type == KEYDOWN and event.key == K_y:
            g = g + rota_dis
        elif event.type == KEYDOWN and event.key == K_m:
            print(a, b, g, x, z, r)

        elif event.type == KEYDOWN and event.key == K_p:

            string_image = pygame.image.tostring(window, 'RGB')
            temp_surf = pygame.image.fromstring(string_image, display, 'RGB')
            tmp_arr = pygame.surfarray.array3d(temp_surf)

            im2 = np.zeros((2 * H, 2 * L, 3))
            for m in range(2 * H):
                for n in range(2 * L):
                    im2[m, n, 0] = tmp_arr[n, m, 2]
                    im2[m, n, 1] = tmp_arr[n, m, 1]
                    im2[m, n, 2] = tmp_arr[n, m, 0]
            im2 = cv2.resize(im2, (L, H))
            cv2.imwrite('produce_chaoyue_{}.png'.format(pic_index), im2)
            pic_index += 1
