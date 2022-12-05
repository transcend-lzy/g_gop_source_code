from model import *
from config import opt
from tqdm import tqdm
from utils import mkdir
from overlayAbg import OverLay as overlay_abg


def overlay_img_opengl(bottom_current, obj_id, abg):
    height, width = 2048, 2448
    overlay = overlay_abg(obj_id)
    overlay.abgxyr = abg
    mtx = np.zeros((3, 3))
    mtx[0][0], mtx[1][1], mtx[0][2], mtx[1][2], mtx[2][2] = opt.fx, opt.fy, opt.ox, opt.oy, 1.0
    overlay.K = mtx
    overlay.cad_path = opt.cad_path
    img = overlay.create_img().astype(np.uint8)
    canny_im = cv2.Canny(img, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(canny_im, kernel, iterations=1)
    im4 = np.copy(dilation)
    im3 = np.zeros((height, width, 3))
    color = [0, 255, 0]
    for m in range(height):
        for n in range(width):
            if im4[m][n] != 0.0:
                im3[m][n] = color
    bottom_current = cv2.addWeighted(bottom_current, 1, im3.astype(np.uint8), 0.5, 0)
    return bottom_current


def crop_img():
    img_path_ori = './final_img'
    img_path_crop = './final_img_crop'
    if not osp.exists(img_path_crop):
        os.makedirs(img_path_crop)
    for i in tqdm(range(1, 216)):
        img = cv2.imread(osp.join(img_path_ori, str(i) + '.png'))
        img_new = img[512:1536, 612:1836]
        cv2.imwrite(osp.join(img_path_crop, str(i) + '.png'), img_new)


def overlay_by_opengl():
    pick_all = np.load('./result_pick_all.npy')
    obj_ids = [6]
    img_path_dst = opt.video_img_data
    for img_index in tqdm(range(1, 216)):
        for obj_id in obj_ids:
            abg = pick_all[img_index - 1][0]
            img_path = osp.join(img_path_dst, 'rgb', str(img_index) + '.png')
            bottom_final = cv2.imread(img_path)
            bottom_final = overlay_img_opengl(bottom_final, abg=abg, obj_id=obj_id)
            if not osp.exists('./final_img'):
                mkdir('./final_img')
            cv2.imwrite(osp.join('./final_img', str(img_index) + '.png'), bottom_final)


def img_2_video():
    file_dir = './final_img_crop'
    list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            list.append(file)  # 获取目录下文件名列表

    video = cv2.VideoWriter('./video_new.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30,
                            (1224, 1024))
    for i in tqdm(range(1, 216)):
        img = cv2.imread(osp.join(file_dir, str(i) + '.png'))
        video.write(img)

    video.release()