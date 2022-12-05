import yaml
from convex_hull_3d.main import *
from read_stl import stl_model
from trans_pose import *
from utils import mkdir
import matplotlib.pyplot as plt


def read_yaml(yaml_path):
    f = open(yaml_path, 'r', encoding='utf-8')
    cfg = f.read()
    dic = yaml.safe_load(cfg)

    return dic


def create_yaml(dic, path):
    """
    根据dic的内容创建yaml文件
    """
    with open(path, "w") as f:
        yaml.dump(dic, f)


def cal_bbox_range(random_bbox_path, obj_id):
    """
    计算bbox的最大边长，用来确定框的边长是128的几倍
    """
    bbox_range = 0
    bbox_offset = read_yaml(random_bbox_path)
    for i, j in bbox_offset.items():
        for k in j:
            if k['obj_id'] == int(obj_id):
                w, h = k['xywh'][2], k['xywh'][3]
                bbox_range = max(w, h, bbox_range)
    return bbox_range


def get_obj_bbox_length_yml(random_bbox_path, obj_id):
    obj_id = str(obj_id)
    bbox_range = cal_bbox_range(random_bbox_path, obj_id)
    if bbox_range % 128 == 0:
        bbox_length = bbox_range
    else:
        bbox_length = (bbox_range // 128 + 1) * 128
    dic = {obj_id: bbox_length}
    # 获取当前目录文件夹的父文件夹
    parent_dir = osp.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = osp.join(parent_dir, 'data', obj_id)
    mkdir(data_dir)
    yml_path = osp.join(data_dir, 'obj_bbox_length.yml')
    create_yaml(dic, yml_path)


def get_obj_gop_range_yml(random_bbox_path, obj_id, start=1, end=104):
    """
    用来计算物体在图像上的分布, 给物体生成图像范围进一步缩小
    """
    obj_id = str(obj_id)
    obj_gop_range = {}
    bbox = read_yaml(random_bbox_path)
    xmin, xmax = 10000000, -10000000
    ymin, ymax = 10000000, -10000000
    for i in range(start, end + 1):
        for j in bbox[str(i)]:
            if j['obj_id'] == int(obj_id):
                xywh = j['xywh']
                xmin = min(xmin, xywh[0])
                xmax = max(xmax, xywh[0] + xywh[3])
                ymin = min(ymin, xywh[1])
                ymax = max(ymax, xywh[1] + xywh[2])
    obj_gop_range[str(obj_id)] = [xmin, xmax, ymin, ymax]
    parent_dir = osp.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = osp.join(parent_dir, 'data', obj_id)
    mkdir(data_dir)
    create_yaml(obj_gop_range, osp.join(data_dir, 'obj_gop_range.yml'))


def get_abg_single(abg_pose_path, index, start, end):
    gt = read_yaml(abg_pose_path)
    res = []
    for i in range(start, end + 1):
        res.append(gt[str(i)]['abg'][index])
    return res


def val_range_hist(abg_pose_path, start, end):
    a = get_abg_single(abg_pose_path, 0, start, end)
    b = get_abg_single(abg_pose_path, 1, start, end)
    g = get_abg_single(abg_pose_path, 2, start, end)
    # bin 表示分成多少个柱状图
    # hist表示每个柱状图（一个范围）有多少个数据
    # bin_edges表示表示一个范围的两个边的数值
    hist, bin_edges = np.histogram(a, bins=1, range=None, normed=False, weights=None, density=None)
    print(hist)
    print(bin_edges)
    hist, bin_edges = np.histogram(b, bins=20, range=None, normed=False, weights=None, density=None)
    print(hist)
    print(bin_edges)
    hist, bin_edges = np.histogram(g, bins=1, range=None, normed=False, weights=None, density=None)
    print(hist)
    print(bin_edges)
    plt.hist([b], bins=20, density=True)
    plt.show()

    # hist, bin_edges = np.histogram(a, bins=20, range=None, normed=False, weights=None, density=None)
    # print('a' + hist)
    # hist, bin_edges = np.histogram(b, bins=20, range=None, normed=False, weights=None, density=None)
    # print('b' + hist)
    # hist, bin_edges = np.histogram(g, bins=20, range=None, normed=False, weights=None, density=None)
    # print('g' + hist)


def get_abg_range_yml(obj_id, abg_pose_path, start=1, end=104):
    """
    将 m2c_rt 转换成 abgxyr， 用来计算abgxyr的范围
    """
    obj_id = str(obj_id)
    abg_range = {}
    bin_edges = []
    for i in range(6):
        temp = get_abg_single(abg_pose_path, i, start, end)
        _, bin_edge_temp = np.histogram(temp, bins=1, range=None, normed=False, weights=None, density=None)
        bin_edges.append(bin_edge_temp)
    keys = ['Arange', 'Brange', 'Grange', 'Xrange', 'Yrange', 'Rrange']
    for i in range(6):
        abg_range[keys[i]] = [[float(bin_edges[i][0]), float(bin_edges[i][1])]]
    parent_dir = osp.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = osp.join(parent_dir, 'data', obj_id)
    mkdir(data_dir)
    create_yaml(abg_range, osp.join(data_dir, 'abg_range.yml'))


def get_hull_points_yml(start=40, end=41):
    """
    调用quick_hull生成凸包点
    """
    hull_points = {}
    for i in range(start, end):
        hull_points[str(i + 1)] = generate_arr(create_convex_hull(str(i + 1)), unit='mm')
    parent_dir = osp.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = osp.join(parent_dir, 'data')
    mkdir(data_dir)
    create_yaml(hull_points, osp.join(data_dir, 'hull_points_41.yml'))


def get_eight_points_yml():
    """
    创建八个点，用来生成aver,eight_points_yml每个obj后面有9个点，分别是8个点加aver点
    """
    eight_points = {}

    for index in range(40, 41):
        stl_path = osp.join(dir_name, 'data', 'CADmodels', 'stl', str(index + 1) + '.stl')
        file = stl_model(stl_path)
        x_min, y_min, z_min = 0., 0., 0.
        x_max, y_max, z_max = 0., 0., 0.
        tri = file.tri
        # tri是三角面片，p0，p1，p2分别是三角面片的三个顶点
        for i in tri:
            x_max = max(i['p0'][0], i['p1'][0], i['p2'][0], x_max)
            x_min = min(i['p0'][0], i['p1'][0], i['p2'][0], x_min)
            y_max = max(i['p0'][1], i['p1'][1], i['p2'][1], y_max)
            y_min = min(i['p0'][1], i['p1'][1], i['p2'][1], y_min)
            z_max = max(i['p0'][2], i['p1'][2], i['p2'][2], z_max)
            z_min = min(i['p0'][2], i['p1'][2], i['p2'][2], z_min)
        x_max *= 1000
        x_min *= 1000
        y_max *= 1000
        y_min *= 1000
        z_max *= 1000
        z_min *= 1000
        point = np.array([[x_min, y_min, z_min]])
        # [0，x]是分别取i为0和x，这里组合最远点和最近点可以得到所有的八个点
        for i in [x_min, x_max]:
            for j in [y_min, y_max]:
                for k in [z_min, z_max]:
                    if i == x_min and j == y_min and k == z_min:
                        continue
                    point = np.concatenate((point, np.array([[i, j, k]])), 0)
        aver = sum(point) / 8
        point = np.concatenate((point, np.array([[aver[0], aver[1], aver[2]]])), 0)
        eight_points[str(index + 1)] = point.tolist()
    parent_dir = osp.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = osp.join(parent_dir, 'data')
    mkdir(data_dir)
    create_yaml(eight_points, osp.join(data_dir, 'eight_points_41.yml'))


def generate_arr(hull_point, unit='m'):
    """
    将hull的point转换成数组
    """
    res = []
    if unit == 'm':
        scale = 1000
    elif unit == 'mm':
        scale = 1
    for i in hull_point:
        res.append([float(i.x) * scale, float(i.y) * scale, float(i.z) * scale])
    return res


def get_special_points_yml():
    """
    创建特殊点
    """
    special_points = {}
    parent_dir = osp.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = osp.join(parent_dir, 'data')
    eight_points = read_yaml(osp.join(data_dir, 'eight_points_41.yml'))
    hull_points = read_yaml(osp.join(data_dir, 'hull_points_41.yml'))
    for obj_id in eight_points.keys():
        aver = eight_points[str(obj_id)][-1]
        if obj_id in hull_points.keys():
            special_point_single = np.array(hull_points[str(obj_id)]) - np.array(aver)
            special_points[obj_id] = np.around(special_point_single, decimals=2).tolist()
    create_yaml(special_points, osp.join(data_dir, 'special_points_41.yml'))


def update_rt(m2c_r, m2c_t, aver):
    """
    在模型中心转换之后，将位姿同样转换，让位姿和模型匹配
    """
    rt = np.concatenate(
        (np.concatenate((np.array(m2c_r), np.array(m2c_t).reshape((3, 1))), 1), np.array([[0, 0, 0, 1]])), 0)
    rt_new = np.dot(rt, np.array([[1, 0, 0, aver[0]],
                                  [0, 1, 0, aver[1]],
                                  [0, 0, 1, aver[2]],
                                  [0, 0, 0, 1]]))
    return rt_new[:3, :3].tolist(), rt_new[:3, 3].reshape((1, 3)).tolist()


def create_new_rt(rt_path_root, aver, obj_id):
    gt = read_yaml(osp.join(rt_path_root, 'gt.yml'))
    for i, j in gt.items():
        for item in j:
            if item['obj_id'] == int(obj_id):
                m2c_R = item['m2c_R']
                m2c_T = item['m2c_T']
                r_new, t_new = update_rt(m2c_R, m2c_T, aver)
                item['m2c_R'] = r_new
                item['m2c_T'] = t_new
    parent_dir = osp.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = osp.join(parent_dir, 'data')
    mkdir(data_dir)
    create_yaml(gt, osp.join(data_dir, 'gt_new.yml'))


def create_abg(rt_new_path_root, obj_id, start=1, end=104):
    abg_gts = {}
    gt_new = read_yaml(osp.join(rt_new_path_root, 'gt.yml'))
    for i in range(start, end + 1):
        abg_gt = {'obj_id': obj_id}
        for item in gt_new[str(i)]:
            if item['obj_id'] == int(obj_id):
                m2c_R = item['m2c_R']
                m2c_T = item['m2c_T']
                a, b, g, x, y, r = rt2abg_final(m2c_R, m2c_T)
                abg_gt['abg'] = [a, b, g, x, y, r]
        abg_gts[str(i)] = abg_gt
    parent_dir = osp.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = osp.join(parent_dir, 'data', str(obj_id), 'test')
    mkdir(data_dir)
    create_yaml(abg_gts, osp.join(data_dir, 'gt_abg.yml'))


if __name__ == '__main__':
    id = 41
    dir_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene_path = osp.join(dir_name, 'data', str(id), 'test')
    bbox_path = osp.join(scene_path, 'bboxOffset.yml')
    gt_path = osp.join(scene_path, 'gt.yml')
    # get_obj_bbox_length_yml(bbox_path, id)
    # get_obj_gop_range_yml(bbox_path, id, 1,1)
    get_hull_points_yml()
    get_eight_points_yml()
    get_special_points_yml()
    # create_abg(scene_path, id, 1, 2)
    # abg_path = osp.join(osp.join(dir_name, 'data'), 'gt_abg.yml')
    # val_range_hist(abg_path, 1, 2)
    # get_abg_range_yml(id, abg_path, 1, 2)

