import argparse
import copy
from multiprocessing import Pool
import os.path as osp
import inspect
import os
import logging

cur_dir = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))

log_path = osp.join(cur_dir, 'generate_log.log')
logging.basicConfig(filename=log_path, level=logging.INFO)

SAMPLE_IMAGE_GENERATOR = 'sample_image_generator.py'
SAMPLE_POSE_GENERATOR = 'sample_pose_generator.py'


def start_sample_image_generator(is_val, obj_id, start, end):
    init_sample_image_generator = osp.join(cur_dir, SAMPLE_IMAGE_GENERATOR)
    if is_val:
        cmd = "python %s --is_val --obj_id %s --start %s --end %s" % \
              (init_sample_image_generator, obj_id, start, end)
    else:
        cmd = "python %s --obj_id %s --start %s --end %s" % \
              (init_sample_image_generator, obj_id, start, end)
    logging.info('start_sample_image_generator and cmd is + %s' % cmd)
    os.system(cmd)


def start_sample_pose_generator(is_val, obj_id, set_length, all_set):
    init_sample_pose_generator = osp.join(cur_dir, SAMPLE_POSE_GENERATOR)
    if is_val:
        cmd = "python '%s' --is_val --obj_id %s --set_length %s --all_set %s" % \
              (init_sample_pose_generator, obj_id, set_length, all_set)
    else:
        cmd = "python '%s' --obj_id %s --set_length %s --all_set %s" % \
              (init_sample_pose_generator, obj_id, set_length, all_set)
    logging.info('start_sample_pose_generator and cmd is + %s' % cmd)
    os.system(cmd)


def arg_parse():
    parser = argparse.ArgumentParser(description="生成训练数据")
    parser.add_argument('--only_image', action='store_true')
    parser.add_argument('-s', '--start', type=int, default=0)  # 非val有作用，控制图像生成的start
    parser.add_argument('-e', '--end', type=int, default=1) # 非val有作用，控制图像生成的end
    parser.add_argument('-b', '--obj_id', type=str, default='6')
    parser.add_argument('--is_val', action='store_true')
    parser.add_argument('-l', '--set_length', type=int, default=10000)
    args = parser.parse_args()
    return args


def get_process_list(start, end):
    all_set = end - start
    single = all_set // 4
    remainder = all_set % 4
    res = [0, 0, 0, 0]
    for i in range(4):
        res[i] = single
        if i < remainder:
            res[i] += 1
    start_and_end = []
    for i in res:
        end = start + i
        cur_start_end = [start, end]
        start = end
        start_and_end.append(copy.deepcopy(cur_start_end))
    print(start_and_end)
    return start_and_end


def main(arg):
    if not args.only_image:
        start_sample_pose_generator(arg.is_val, args.obj_id, arg.set_length, arg.end - arg.start)
    pool = Pool(processes=4)
    if (args.end - args.start) <= 4:
        for i in range(args.start, args.end):
            pool.apply_async(start_sample_image_generator, args=(arg.is_val, arg.obj_id, i, i + 1))
    else:
        start_and_end = get_process_list(args.start, args.end)
        for i in start_and_end:
            pool.apply_async(start_sample_image_generator, args=(arg.is_val, arg.obj_id, i[0], i[1]))
    pool.close()
    pool.join()


if __name__ == '__main__':
    args = arg_parse()
    main(args)
