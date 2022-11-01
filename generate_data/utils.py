import yaml
import cv2
import os
import os.path as osp

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
