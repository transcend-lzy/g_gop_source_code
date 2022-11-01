import copy
import open3d as o3d
import numpy as np
import os
# 文件绝对路径
current_file_path = __file__
# 借助dirname()从绝对路径中提取目录
current_file_dir = os.path.dirname(current_file_path)
import sys
sys.path.append(current_file_dir)
from utils_hull import *
from plane import Plane
import vis_convhull
import os.path as osp


def cal_dist_between_points(points):
    length = len(points)
    matrix = np.zeros((length, length))
    for i in range(length):
        for j in range(i + 1, length):
            matrix[i][j] = cal_dist_two_points(points[i], points[j])
    return matrix


def find_mid_point(matrix, i, j):
    for k in range(j + 1, len(matrix)):
        if is_same_line(matrix[i][k], matrix[i][j], matrix[j][k]):
            if max(matrix[i][k], matrix[i][j], matrix[j][k]) == matrix[i][k]:
                print("删除了{}号点，同一批次的是{}，{}".format(j, i, k))
                return j
            elif max(matrix[i][k], matrix[i][j], matrix[j][k]) == matrix[i][j]:
                print("删除了{}号点，同一批次的是{}，{}".format(k, i, j))
                return k
            else:
                print("删除了{}号点，同一批次的是{}，{}".format(i, k, j))
                return i
    return None


def is_same_line(dist1, dist2, dist3):
    dist_sort = sorted([dist1, dist2, dist3])
    return dist_sort[2] - dist_sort[0] - dist_sort[1] < 0.5 and dist_sort[2] - dist_sort[0] - dist_sort[1] >= 0


def cal_dist_two_points(pointa, pointb):
    return math.sqrt((pointa.x - pointb.x) ** 2 + (pointa.y - pointb.y) ** 2 + (pointa.z - pointb.z) ** 2)

def read_txt(txt_path):
    with open(txt_path, encoding='ANSI') as file:
        content = file.read()
        vertixs = []
        for line in content.split('\n'):
            if len(line) > 3:
                vertixs.append([float(line.split(' ')[0]), float(line.split(' ')[1]), float(line.split(' ')[2])])
    return vertixs


def create_convex_hull(obj_id, is_vis = False):
    ply_path_root = osp.join(osp.dirname(os.path.dirname(os.path.abspath(__file__))), 'CADmodels', 'ply')
    ply_path = osp.join(ply_path_root, 'obj' + str(obj_id) + '.ply')
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()
    print('\nLoaded mesh file %s' % ply_path)
    print('#vertices:', np.asarray(mesh.vertices).shape[0])
    print('#faces:', np.asarray(mesh.triangles).shape[0])
    # 创造四面体
    if is_vis:
        vertices = read_txt(osp.join('data', str(obj_id) + '.txt'))
    else:
        vertices = mesh.vertices
    extremes, points = search_init_points(vertices)

    if len(points) < 4:
        print("Less than 4 points so 1D or 2D")
        sys.exit()
    initial_line = initial_max(extremes)
    third_point = max_dist_line_point1(initial_line[0], initial_line[1], points)
    third_point1 = max_dist_line_point2(initial_line[0], initial_line[1], extremes)
    first_plane = Plane(initial_line[0], initial_line[1],
                        third_point)
    fourth_point = max_dist_plane_point(first_plane, points)
    possible_internal_points = [initial_line[0], initial_line[1], third_point,
                                fourth_point]  # 帮助计算点的方向的列表
    second_plane = Plane(initial_line[0], initial_line[1], fourth_point)
    third_plane = Plane(initial_line[0], fourth_point, third_point)
    fourth_plane = Plane(initial_line[1], third_point, fourth_point)

    list_of_planes = []  # List containing all the planes
    list_of_planes.append(first_plane)
    list_of_planes.append(second_plane)
    list_of_planes.append(third_plane)
    list_of_planes.append(fourth_plane)
    for plane in list_of_planes:
        set_correct_normal(possible_internal_points, plane)  # 设定法线的方向正确

    first_plane.calculate_to_do(points)  # 寻找每个面的todo点
    second_plane.calculate_to_do(points)
    third_plane.calculate_to_do(points)
    fourth_plane.calculate_to_do(points)
    any_left = True

    while any_left:
        any_left = False
        for working_plane in list_of_planes:
            if len(working_plane.to_do) > 0:
                any_left = True
                eye_point = find_eye_point(working_plane)  # Calculate the eye point of the face

                edge_list = set()
                visited_planes = []

                calc_horizon(visited_planes, working_plane, eye_point, edge_list,
                             list_of_planes)  # Calculate the horizon

                for internal_plane in visited_planes:  # Remove the internal planes
                    list_of_planes.remove(internal_plane)

                for edge in edge_list:  # Make new planes
                    new_plane = Plane(edge.pointA, edge.pointB, eye_point)
                    set_correct_normal(possible_internal_points, new_plane)

                    temp_to_do = set()
                    for internal_plane in visited_planes:
                        # union返回两个集合的并集，即包含了所有集合的元素
                        temp_to_do = temp_to_do.union(internal_plane.to_do)

                    new_plane.calculate_to_do(points, temp_to_do)

                    list_of_planes.append(new_plane)

    final_vertices = set()

    for plane in list_of_planes:
        final_vertices.add(plane.pointA)
        final_vertices.add(plane.pointB)
        final_vertices.add(plane.pointC)
    final_vertices = list(final_vertices)
    length = len(final_vertices)
    dic_points = {}
    for i, point in enumerate(final_vertices):
        dic_points[str(i)] = point

    matrix = cal_dist_between_points(final_vertices)

    for i in range(length):
        if str(i) in dic_points.keys():
            for j in range(i + 1, length):
                mid_point = find_mid_point(matrix, i, j)
                if mid_point is not None:
                    try:
                        del dic_points[str(mid_point)]
                    except:
                        print("重复删除{}号点".format(mid_point))
    final_vertices = []
    for point in dic_points.values():
        final_vertices.append(point)
    list_of_planes_copy = copy.deepcopy(list_of_planes)
    for plane in list_of_planes:
        if plane.pointA not in final_vertices or plane.pointB not in final_vertices or plane.pointC not in final_vertices:
            list_of_planes_copy.remove(plane)
    
    return final_vertices




    if is_vis:
        vis_convhull.vis(mesh, list_of_planes_copy)


if __name__ == '__main__':
    for i in range(38):
        create_convex_hull(i + 1, True)
