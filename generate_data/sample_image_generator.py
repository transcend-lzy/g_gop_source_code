import numpy as np
from utils import *
import argparse
import os
from trans_pose import *
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from read_stl import stl_model
from tqdm import tqdm, trange


class Gen(object):
    def __init__(self, model_index, is_abg=True):
        self.modelIndex = str(model_index)
        dir_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if 'home' in dir_name:
            self.is_linux = True
        else:
            self.is_linux = False
        self.data_path = osp.join(dir_name, 'data')
        self.bbox_length = read_yaml(osp.join(self.data_path, self.modelIndex, 'obj_bbox_length.yml'))[self.modelIndex]
        self.val_img_path = osp.join(self.data_path, self.modelIndex, 'Edge_im', 'val')
        mkdir(self.val_img_path)
        self.pose_path = osp.join(self.data_path, self.modelIndex, 'pose_set')
        mkdir(self.pose_path)
        self.img_path = osp.join(self.data_path, self.modelIndex, 'Edge_im')
        self.img_npy_path = osp.join(self.data_path, self.modelIndex, 'train_data')
        mkdir(self.img_npy_path)
        self.scale = 1  # 缩放倍数，正常生成2448*2048的，先生成小一点的再扩大
        self.ox = ox / self.scale  # 相机内参
        self.oy = oy / self.scale  # 相机内参
        self.FocalLength_x = FocalLength_x / self.scale  # 相机内参
        self.FocalLength_y = FocalLength_y / self.scale  # 相机内参
        self.aver_mm = read_yaml(osp.join(self.data_path, 'eight_points.yml'))[self.modelIndex][-1]
        self.aver = np.multiply(self.aver_mm, 0.001)
        self.stl_path = osp.join(dir_name, 'generate_data/CADmodels/stl/{}.stl'.format(self.modelIndex))
        self.height = 2048 // self.scale  # 图片长
        self.width = 2448 // self.scale  # 图片宽
        self.start = 0
        self.end = 0
        self.is_abg = is_abg

    def init_pygame(self, width, height):
        pygame.init()
        display = (width, height)
        window = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        glMatrixMode(GL_PROJECTION)
        # 把矩阵设为单位矩阵
        glLoadIdentity()
        # 每次重绘之前，需要先清除屏幕及深度缓存，一般放在绘图函数开头
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # 设备画布背景色
        glClearColor(1, 1, 1, 0.0)
        near = 0.0001
        far = 100
        left = -self.ox * near / self.FocalLength_x
        right = (width - self.ox) * near / self.FocalLength_x
        top = self.oy * near / self.FocalLength_y
        bottom = (self.oy - height) * near / self.FocalLength_y
        glFrustum(left, right, bottom, top, near, far)
        glMatrixMode(GL_MODELVIEW)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)  # 设置深度测试函数
        glEnable(GL_POINT_SMOOTH)
        glPolygonMode(GL_FRONT, GL_FILL)
        glPolygonMode(GL_BACK, GL_FILL)
        return display, window

    def cube(self, tri):  # 提取一堆点
        glBegin(GL_TRIANGLES)  # 绘制多个三角形
        aver = self.aver
        for Tri in tri:
            glColor3fv(Tri['colors'])
            glVertex3fv(
                (Tri['p0'][0] - aver[0], Tri['p0'][1] - aver[1], Tri['p0'][2] - aver[2]))
            glVertex3fv(
                (Tri['p1'][0] - aver[0], Tri['p1'][1] - aver[1], Tri['p1'][2] - aver[2]))
            glVertex3fv(
                (Tri['p2'][0] - aver[0], Tri['p2'][1] - aver[1], Tri['p2'][2] - aver[2]))

        glEnd()  # 实际上以三角面片的形式保存

    def draw_cube_abg(self, alpha, beta, gama, x_trans, z_trans, radius, tri, window, display):
        glPushMatrix()
        # 清除一下原来的生成历史，否则新生成的图像会带有原来的阴影
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        m2c_r, m2c_t = abg2rt_final(alpha, beta, gama, x_trans, z_trans, radius)
        m2c_r = np.array(m2c_r)
        m2c_t = np.array(m2c_t)
        m2c_rt = np.concatenate((np.concatenate((m2c_r, m2c_t.T), 1), np.array([[0, 0, 0, 1]])), 0)
        c2m_rt = np.linalg.inv(m2c_rt)
        pos = c2m_rt[0:3, 3]
        # 以模型坐标系为世界坐标系，则摄像机的位置为pos，摄像机的z轴指向目标物体（该目标物体是opengl的目标物体，并非我们要渲染的目标物体），
        # 目标物体的位置坐标根据向量加法即可得到，为pos + z
        # 由于opengl里面定义的相机坐标系与物理相机坐标系不同，不同之处在于opengl坐标系相对于物理坐标系绕x轴旋转180度，
        # 即yz轴取反，所以此时y轴方向取反
        obj = pos + c2m_rt[0:3, 2]
        yy = c2m_rt[0:3, 1]
        gluLookAt(pos[0], pos[1], pos[2], obj[0], obj[1], obj[2], -yy[0], -yy[1], -yy[2])
        self.cube(tri)
        glPopMatrix()
        pygame.display.flip()

        # Read the result
        string_image = pygame.image.tostring(window, 'RGB')
        temp_surf = pygame.image.fromstring(string_image, display, 'RGB')
        tmp_arr = pygame.surfarray.array3d(temp_surf)
        return tmp_arr  # 得到最后的图

    def draw_cube_rt(self, m2c_r, m2c_t, tri, window, display):
        glPushMatrix()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        m2c_r = np.array(m2c_r)
        m2c_t = np.array(m2c_t)
        m2c_rt = np.concatenate((np.concatenate((m2c_r, m2c_t.T), 1), np.array([[0, 0, 0, 1]])), 0)
        c2m_rt = np.linalg.inv(m2c_rt)
        pos = c2m_rt[0:3, 3]
        # 以模型坐标系为世界坐标系，则摄像机的位置为pos，摄像机的z轴指向目标物体（该目标物体是opengl的目标物体，并非我们要渲染的目标物体），
        # 目标物体的位置坐标根据向量加法即可得到，为pos + z
        # 由于opengl里面定义的相机坐标系与物理相机坐标系不同，不同之处在于opengl坐标系相对于物理坐标系绕x轴旋转180度，
        # 即yz轴取反，所以此时y轴方向取反
        obj = pos + c2m_rt[0:3, 2]
        yy = c2m_rt[0:3, 1]
        gluLookAt(pos[0], pos[1], pos[2], obj[0], obj[1], obj[2], -yy[0], -yy[1], -yy[2])
        self.cube(tri)
        glPopMatrix()
        pygame.display.flip()

        # Read the result
        string_image = pygame.image.tostring(window, 'RGB')
        temp_surf = pygame.image.fromstring(string_image, display, 'RGB')
        tmp_arr = pygame.surfarray.array3d(temp_surf)
        return tmp_arr  # 得到最后的图

    def create_img(self):
        display, window = self.init_pygame(self.width, self.height)
        tri = stl_model(self.stl_path).tri

        for index in range(self.start, self.end):
            print(index)
            if not osp.exists(osp.join(self.img_path, str(index))):
                os.mkdir(osp.join(self.img_path, str(index)))
            pose_set = np.load(osp.join(self.pose_path, '{}-{}.npy'.format(index * 10000, index * 10000 + 10000)))
            im_index = index * 10000
            for pose in tqdm(pose_set):
                # pose = pose_set[pose_index]
                u, v = int(pose[10]), int(pose[11])
                # im = self.draw_cube_abg(a, b, g, x / 1000, y / 1000, r / 1000, tri, window, display)
                if self.is_abg:
                    a, b, g, x, y, r = pose[:6]
                    if self.is_linux:
                        im = np.array(self.draw_cube_abg(a, b, g, x, y, r, tri, window, display))
                    else:
                        im = np.array(self.draw_cube_abg(a, b, g, x, y, r, tri, window, display))
                        im = np.array(self.draw_cube_abg(a, b, g, x, y, r, tri, window, display))
                else:
                    a, b, g, x, y, z = pose[:6]
                    m2c_r = eulerAnglesToRotationMatrix([a, b, g])
                    m2c_t = np.array([[x, y, z]])
                    if self.is_linux:
                        im = np.array(self.draw_cube_rt(m2c_r, m2c_t, tri, window, display))
                    else:
                        im = np.array(self.draw_cube_rt(m2c_r, m2c_t, tri, window, display))
                        im = np.array(self.draw_cube_rt(m2c_r, m2c_t, tri, window, display))
                im_new = np.zeros((self.height, self.width, 3))
                for i in range(3):
                    im_new[:, :, i] = im[:, :, i].T
                im_index = im_index + 1
                img_ori = cv2.resize(im_new, (2448, 2048), interpolation=cv2.INTER_LINEAR)
                im_clip = img_ori[v: v + self.bbox_length, u: u + self.bbox_length]
                canny_im = cv2.Canny(im_clip.astype(np.uint8), 100, 200)
                kernel = np.ones((3, 3), np.uint8)
                dilation = cv2.dilate(canny_im, kernel, iterations=1)
                small = cv2.resize(dilation, (128, 128), cv2.INTER_AREA).astype(np.uint8)
                small = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY)
                small = np.array(small[1])
                cv2.imwrite(osp.join(self.img_path, str(index), str(im_index) + '.png'), small)
        lens = 10000
        for j in range(self.start, self.end):
            dataset = []
            for i in range(1 + lens * j, 1 + lens * (j + 1)):
                im = cv2.imread('./Edge_im/{}.png'.format(i))
                im = im[:, :, 0]
                assert len(np.shape(im)) == 2 and np.shape(im)[0] == 128 and np.shape(im)[1] == 128
                dataset.append(im)
            np.save(osp.join(self.img_npy_path, 'dataset_uint8_{}'.format(j + 1)), dataset)

    def create_val_img(self):
        display, window = self.init_pygame(self.width, self.height)
        tri = stl_model(self.stl_path).tri
        pose_set = np.load(osp.join(self.pose_path, 'validation(640).npy'))
        im_index = 0
        for pose in tqdm(pose_set):
            # pose = pose_set[pose_index]
            a, b, g, x, y, r = pose[:6]
            u, v = int(pose[10]), int(pose[11])
            im = np.array(self.draw_cube_abg(a, b, g, x, y, r, tri, window, display))
            im = np.array(self.draw_cube_abg(a, b, g, x, y, r, tri, window, display))
            im_new = np.zeros((self.height, self.width, 3))
            for i in range(3):
                im_new[:, :, i] = im[:, :, i].T
            img_ori = cv2.resize(im_new, (2448, 2048), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(img_ori, (u, v),
                          (u + self.bbox_length, v + self.bbox_length), (0, 255, 0), 2)
            # show_photo(img_ori)
            im_clip = img_ori[v: v + self.bbox_length, u: u + self.bbox_length]
            # show_photo(im_clip)
            canny_im = cv2.Canny(im_clip.astype(np.uint8), 100, 200)
            kernel = np.ones((3, 3), np.uint8)
            dilation = cv2.dilate(canny_im, kernel, iterations=1)
            small = cv2.resize(dilation, (128, 128), cv2.INTER_AREA).astype(np.uint8)
            small = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY)
            small = np.array(small[1])
            im_index += 1
            # show_photo(small)
            cv2.imwrite(osp.join(self.val_img_path, str(im_index) + '.png'), small)
        dataset = []
        for i in trange(1, 641):
            im = cv2.imread(osp.join(gen_img.val_img_path, str(i) + '.png'))
            im = im[:, :, 0]
            assert len(np.shape(im)) == 2 and np.shape(im)[0] == 128 and np.shape(im)[1] == 128
            dataset.append(im)
        np.save(osp.join(gen_img.img_npy_path, 'dataset_uint8(validation640).npy'), dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_val", action='store_true')  # 默认false， 传了就是true
    parser.add_argument("--is_abg", action='store_true')  # 默认false， 传了就是true
    parser.add_argument("--obj_id", type=str, default='6')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1)
    args = parser.parse_args()
    gen_img = Gen(args.obj_id, args.is_abg)
    if args.is_val:
        gen_img.create_val_img()
    else:
        gen_img.start = args.start
        gen_img.end = args.end
        gen_img.create_img()
