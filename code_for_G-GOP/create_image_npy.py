import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm, trange

if __name__ == '__main__':
    dataset = []
    for i in range(1, 641):
        im = cv2.imread(osp.join('../data/6/Edge_im/val', str(i) + '.png'))
        im = im[:, :, 0]
        assert len(np.shape(im)) == 2 and np.shape(im)[0] == 128 and np.shape(im)[1] == 128
        dataset.append(im)
    np.save(osp.join(gen_img.img_npy_path, 'dataset_uint8(validation640).npy'), dataset)

    lens = 10000
    for j in trange(0, 64):
        dataset = []
        for i in range(1 + lens * j, 1 + lens * (j + 1)):
            im = cv2.imread('../data/6/Edge_im/{}/{}.png'.format(j, i))
            im = im[:, :, 0]
            assert len(np.shape(im)) == 2 and np.shape(im)[0] == 128 and np.shape(im)[1] == 128
            dataset.append(im)
        np.save(osp.join('../data/6/train_data', 'dataset_uint8_{}'.format(j + 1)), dataset)

