import cv2
import numpy as np
import os.path as osp
import sys
import os
import inspect

cur_dir = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
sys.path.append(cur_dir)
sys.path.append(os.path.dirname(cur_dir))
sys.path.append(os.path.dirname(os.path.dirname(cur_dir)))
from tqdm import tqdm, trange
from generate_data.sample_image_generator import Gen

if __name__ == '__main__':
    gen_img = Gen('6')
    dataset = []
    for i in range(1, 641):
        im = cv2.imread(osp.join('./data/6/Edge_im/val', str(i) + '.png'))
        im = im[:, :, 0]
        assert len(np.shape(im)) == 2 and np.shape(im)[0] == 128 and np.shape(im)[1] == 128
        dataset.append(im)
    np.save(osp.join(gen_img.img_npy_path, 'dataset_uint8(validation640).npy'), dataset)

    lens = 10000
    for j in trange(0, 64):
        dataset = []
        for i in range(1 + lens * j, 1 + lens * (j + 1)):
            im = cv2.imread('./data/6/Edge_im/{}/{}.png'.format(j, i))
            im = im[:, :, 0]
            assert len(np.shape(im)) == 2 and np.shape(im)[0] == 128 and np.shape(im)[1] == 128
            dataset.append(im)
        np.save(osp.join('./data/6/train_data', 'dataset_uint8_{}'.format(j + 1)), dataset)
