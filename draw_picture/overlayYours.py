import os

from overlayUtils import *

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from generate_data.trans_pose import abg2rt_final


class OverLay():
    def __init__(self, is_abg=True):
        self.is_abg = is_abg
        self.img_index = 101
        self.img_path = os.path.join('./img', str(self.img_index) + '.jpg')

        if is_abg:
            self.gt = read_yaml('./img/gt_abg.yml')[str(self.img_index)]['abg']
            self.gt = (3.17347754230264, 3.2856604179203144, 6.200711931334433, 0.09218437130369633, -0.00020890785840531334, 0.7660068990992237)
        else:
            self.gt = read_yaml('./img/gt.yml')[str(self.img_index)][1]
        if self.is_abg:
            self.R, self.T = abg2rt_final(self.gt[0], self.gt[1], self.gt[2], self.gt[3], self.gt[4], self.gt[5])
        else:
            self.R = self.gt['m2c_R']
            self.T = self.gt['m2c_T']
        self.K = read_yaml('img/Intrinsic.yml')['Intrinsic']
        self.cad_path = './img/6.stl'  # stl文件
        self.save_path = './img/1_new.jpg'

    def init(self, k, width, height):
        pygame.init()
        display = (width, height)
        window = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        ox = k[0][2]  # 相机标定矩阵的值
        oy = k[1][2]
        FocalLength_x = k[0][0]
        FocalLength_y = k[1][1]
        glMatrixMode(GL_PROJECTION)
        # 把矩阵设为单位矩阵
        glLoadIdentity()
        # 每次重绘之前，需要先清除屏幕及深度缓存，一般放在绘图函数开头
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # 设备画布背景色
        glClearColor(1, 1, 1, 0.0)
        near = 0.0001
        far = 100
        left = -ox * near / FocalLength_x
        right = (width - ox) * near / FocalLength_x
        top = oy * near / FocalLength_y
        bottom = (oy - height) * near / FocalLength_y
        glFrustum(left, right, bottom, top, near, far)
        glMatrixMode(GL_MODELVIEW)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)  # 设置深度测试函数
        glEnable(GL_POINT_SMOOTH)
        glPolygonMode(GL_FRONT, GL_FILL)
        glPolygonMode(GL_BACK, GL_FILL)
        return display, window

    def create_img(self):
        img = cv2.imread(self.img_path)

        height = img.shape[0]
        width = img.shape[1]
        display, window = self.init(self.K, width, height)

        tri = stl_model(self.cad_path).tri
        im = np.array(draw_cube_test(self.R, self.T, tri, window, display))
        im = np.array(draw_cube_test(self.R, self.T, tri, window, display))
        pose_mask = np.zeros((height, width, 3))
        for i in range(3):
            pose_mask[:, :, i] = im[:, :, i].T
        canny_img = cv2.Canny(pose_mask.astype(np.uint8), 10, 200)

        return canny_img


if __name__ == '__main__':
    overlay = OverLay()
    img_target = cv2.imread(overlay.img_path, cv2.IMREAD_GRAYSCALE)
    img = overlay.create_img()
    dst = cv2.addWeighted(img, 1, img_target, 0.5, 0)
    # show_photo(dst)
    # cv2.imwrite(osp.join(overlay.save_path, str(index) + '.png'), dst)
    points_3d = np.float32([[0.03, 0.0505, 0.016]]).reshape(-1, 3)
    points_2d, _ = cv2.projectPoints(points_3d, overlay.R, overlay.T, np.array(overlay.K), None)
    points_2d = points_2d.reshape(-1, 2)[0]
    cv2.circle(dst, (int(points_2d[0]), int(points_2d[1])), 10, (0, 0, 0), 2)
    show_photo(dst)
