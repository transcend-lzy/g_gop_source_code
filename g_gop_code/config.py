# coding:utf8
import warnings
import os
import os.path as osp
import yaml
import sys
import inspect
cur_dir = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
sys.path.append(os.path.dirname(cur_dir))
sys.path.append(os.path.dirname(os.path.dirname(cur_dir)))


def read_yaml(yaml_path):
    f = open(yaml_path, 'r', encoding='utf-8')
    cfg = f.read()
    dic = yaml.safe_load(cfg)
    return dic


class DefaultConfig(object):
    postfix = ''
    obj_id = 41
    main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path_root = osp.join(main_path, 'data')
    data_path = osp.join(main_path, 'data', str(obj_id))
    if not osp.exists(data_path):
        os.mkdir(data_path)
    train_data_root = osp.join(data_path, 'train_data')  # 训练集存放路径os.path.join(main_path,'train_data')
    train_pose_root = osp.join(data_path, 'pose_set_norm')
    test_data_root = osp.join(data_path_root, 'test_2448')
    vis_path = osp.join(data_path_root, 'vis', str(obj_id))
    if not osp.exists(vis_path):
        os.makedirs(vis_path)
    test_sample_path = osp.join(vis_path, 'sample_test')
    val_sample_path = osp.join(vis_path, 'sample_val')
    overlay_path = osp.join(vis_path, 'img_vis')
    save_model_path = osp.join(main_path, 'checkpoint', str(obj_id))
    if not osp.exists(save_model_path):
        os.makedirs(save_model_path)
    log_path = osp.join(main_path, 'logs', str(obj_id))
    log_train = osp.join(log_path, 'train' + postfix)
    log_test = osp.join(log_path, 'test' + postfix)
    pose_num = 32  # 全训练集是100
    batch_size = 900  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 16  # how many workers for loading data
    print_freq = 100  # print info every N batch
    adjust = 10
    save_fre = 40
    bbox_len = read_yaml(osp.join(data_path, 'obj_bbox_length.yml'))[str(obj_id)]
    width = 2448
    height = 2048
    ox = 1239.951701787861  # 相机内参
    oy = 1023.50823945964  # 相机内参
    fx = 2344.665256639324
    fy = 2344.050648508343

    max_epoch = 1000
    learning_rate = 0.003  # 学习率
    learning_rate_test = 0.01
    test_break_ite = 5
    test_opt_ite = 1
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数
    latent_dim = 8
    dim = 32 * 7  # 实验得出
    test_epoch = 15
    test_ite = 100
    # num_examples_to_generate = 64  #得到样本数
    # num_epochs = 200


def parse(self, kwargs):
    """
    根据字典kwargs 更新 config参数
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
# opt.parse = parse
