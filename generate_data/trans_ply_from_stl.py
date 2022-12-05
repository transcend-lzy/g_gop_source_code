import open3d as o3d
import os
import os.path as osp

if __name__ == '__main__':
    dir_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mesh = o3d.io.read_triangle_mesh(osp.join(dir_name, 'data', 'CADmodels', 'stl', '41.stl'))
    o3d.io.write_triangle_mesh(osp.join(dir_name, 'data', 'CADmodels', 'ply', 'obj41.ply'), mesh, write_ascii=True) #指定保存的类型
