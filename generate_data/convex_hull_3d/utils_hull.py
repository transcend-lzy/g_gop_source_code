import math
from point import Point
from edge import Edge
import sys


def search_init_points(vertices):
    """
    找到xyz分别最大和最小的6个点，并将所有的顶点放到points里面
    :param mesh: open3d 读取的网格
    :return: 6个点 和 所有的点集 points
    """
    x_min_temp, y_min_temp, z_min_temp = sys.maxsize, sys.maxsize, sys.maxsize
    x_max_temp, y_max_temp, z_max_temp = -sys.maxsize - 1, -sys.maxsize - 1, -sys.maxsize - 1
    x_max, x_min, y_max, y_min, z_max, z_min = 0, 0, 0, 0, 0, 0
    points = []
    for point_ver in vertices:
        points.append(Point(point_ver[0], point_ver[1], point_ver[2]))
        if point_ver[0] > x_max_temp:
            x_max_temp = point_ver[0]
            x_max = points[-1]
        if point_ver[0] < x_min_temp:
            x_min_temp = point_ver[0]
            x_min = points[-1]

        if point_ver[1] > y_max_temp:
            y_max_temp = point_ver[1]
            y_max = points[-1]

        if point_ver[1] < y_min_temp:
            y_min_temp = point_ver[1]
            y_min = points[-1]

        if point_ver[2] > z_max_temp:
            z_max_temp = point_ver[2]
            z_max = points[-1]

        if point_ver[2] < z_min_temp:
            z_min_temp = point_ver[2]
            z_min = points[-1]


    return [x_max, x_min, y_max, y_min, z_max, z_min], points


def cal_dis(p, q):
    """
    计算两点间距离
    :param p: 第一个点
    :param q: 第二个点
    :return: 距离
    """
    return math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2)


def initial_max(six_init):
    """
    找到六个初始点中距离最大的两个点
    :param six_init:
    :return:
    """
    maxi = -1
    max_points = [[], []]
    for i in range(6):
        for j in range(i + 1, 6):
            dist = cal_dis(six_init[i], six_init[j])
            if dist > maxi:
                max_points = [six_init[i], six_init[j]]
    return max_points


def cross(pointA, pointB):
    """
    两点向量积（叉积）
    :param pointA:
    :param pointB:
    :return: 向量积是得到结果是向量，方向垂直于ab所在平面，大小是|a||b|sin<a，b>
    """
    x = (pointA.y * pointB.z) - (pointA.z * pointB.y)
    y = (pointA.z * pointB.x) - (pointA.x * pointB.z)
    z = (pointA.x * pointB.y) - (pointA.y * pointB.x)
    return Point(x, y, z)


def dot(pointA, pointB):
    """
    点积（数量积）
    :param pointA:
    :param pointB:
    :return:|a||b|cos<a，b>
    """
    return (pointA.x * pointB.x + pointA.y * pointB.y + pointA.z * pointB.z)


def distLine(pointA, pointB, pointX):
    """
    计算点到直线的距离
    :param pointA: 直线端点1
    :param pointB: 直线端点2
    :param pointX: 直线外一点
    :return:
    """
    vec1 = pointX - pointA
    vec2 = pointX - pointB
    vec3 = pointB - pointA
    vec4 = cross(vec1, vec2)
    if vec3.length() == 0:
        return None

    else:
        return vec4.length() / vec3.length()


def max_dist_line_point1(pointA, pointB, points):
    """
    从points里面寻找离线段ab最远的点
    :param pointA:
    :param pointB:
    :return:
    """
    maxDist = 0
    for point in points:
        if (pointA != point) and (pointB != point):
            dist = abs(distLine(pointA, pointB, point))
            if dist > maxDist:
                maxDistPoint = point
                maxDist = dist

    return maxDistPoint


def max_dist_line_point2(pointA, pointB, extremes):
    """
    从points里面寻找离线段ab最远的点
    :param pointA:
    :param pointB:
    :return:
    """
    maxDist = 0
    for point in extremes:
        if (pointA != point) and (pointB != point):
            dist = abs(distLine(pointA, pointB, point))
            if dist > maxDist:
                maxDistPoint = point
                maxDist = dist

    return maxDistPoint


def max_dist_plane_point(plane, points):
    """
    计算到面的最大距离的点
    :param plane:
    :param points:
    :return:
    """
    maxDist = 0
    for point in points:
        dist = abs(plane.dist(point))
        if (dist > maxDist):
            maxDist = dist
            maxDistPoint = point

    return maxDistPoint


def set_correct_normal(possible_internal_points, plane):
    """
    设置面的法线方向正确
    :param possible_internal_points: 初始组成正四面体的四个点
    :param plane:  要设置面的法线方向的面
    :return:
    """
    for point in possible_internal_points:
        # dist为正则表示夹角为锐角   point - plane.pointA表示从面外一点指向三角形的顶点
        dist = dot(plane.normal, point - plane.pointA)
        if (dist != 0):
            # dist > 10 ** -10的意思是dist为正， 这样就要把面的法线方向置为相反
            # possible_internal_points是形成初始四面体的四个点，
            # 对于后面生成的所有面，初始四面体的点都应该属于是面的内部点，面的法向量与他们夹角都应该是钝角
            if (dist > 10 ** -10):
                plane.normal.x = -1 * plane.normal.x
                plane.normal.y = -1 * plane.normal.y
                plane.normal.z = -1 * plane.normal.z
                return


def checker_plane(a, b):  # Check if two planes are equal or not    检查两个平面是否相等

    if ((a.pointA.x == b.pointA.x) and (a.pointA.y == b.pointA.y) and (a.pointA.z == b.pointA.z)):
        if ((a.pointB.x == b.pointB.x) and (a.pointB.y == b.pointB.y) and (a.pointB.z == b.pointB.z)):
            if ((a.pointC.x == b.pointC.x) and (a.pointC.y == b.pointC.y) and (a.pointC.z == b.pointC.z)):
                return True

        elif ((a.pointB.x == b.pointC.x) and (a.pointB.y == b.pointC.y) and (a.pointB.z == b.pointC.z)):
            if ((a.pointC.x == b.pointB.x) and (a.pointC.y == b.pointB.y) and (a.pointC.z == b.pointB.z)):
                return True

    if ((a.pointA.x == b.pointB.x) and (a.pointA.y == b.pointB.y) and (a.pointA.z == b.pointB.z)):
        if ((a.pointB.x == b.pointA.x) and (a.pointB.y == b.pointA.y) and (a.pointB.z == b.pointA.z)):
            if ((a.pointC.x == b.pointC.x) and (a.pointC.y == b.pointC.y) and (a.pointC.z == b.pointC.z)):
                return True

        elif ((a.pointB.x == b.pointC.x) and (a.pointB.y == b.pointC.y) and (a.pointB.z == b.pointC.z)):
            if ((a.pointC.x == b.pointA.x) and (a.pointC.y == b.pointA.y) and (a.pointC.z == b.pointA.z)):
                return True

    if ((a.pointA.x == b.pointC.x) and (a.pointA.y == b.pointC.y) and (a.pointA.z == b.pointC.z)):
        if ((a.pointB.x == b.pointA.x) and (a.pointB.y == b.pointA.y) and (a.pointB.z == b.pointA.z)):
            if ((a.pointC.x == b.pointB.x) and (a.pointC.y == b.pointB.y) and (a.pointC.z == b.pointB.z)):
                return True

        elif ((a.pointB.x == b.pointC.x) and (a.pointB.y == b.pointC.y) and (a.pointB.z == b.pointC.z)):
            if ((a.pointC.x == b.pointB.x) and (a.pointC.y == b.pointB.y) and (a.pointC.z == b.pointB.z)):
                return True

    return False


def find_eye_point(plane):
    """
    在某一面点集中找到最远点
    :param plane: 某一面
    :return:返回最远点
    """
    maxDist = 0
    for point in plane.to_do:
        dist = plane.dist(point)
        if (dist > maxDist):
            maxDist = dist
            maxDistPoint = point

    return maxDistPoint


def adjacent_plane(main_plane, edge, list_of_planes):
    """
    找到和某个面相邻的面
    :param main_plane: 当前面
    :param edge: 当前面的边
    :param list_of_planes: 所有的面
    :return:  找到与当前面相邻的面
    """
    for plane in list_of_planes:
        edges = plane.get_edges()
        if (plane != main_plane) and (edge in edges):
            return plane


def calc_horizon(visited_planes, plane, eye_point, edge_list, list_of_planes):
    """
    深度优先
    找horizon（当前key_point能看见的所有三角形【包括当前plane和其所有相邻三角形（包括相邻三角形的相邻三角形）】的边组成凸多边形）
    一个边如果除了他所在的三角面可见（这样会搜索到他），其余他所在的面都不是可见的，那么他就是凸多边形的一个边
    :param visited_planes:  已经遍历过的面
    :param plane：当前面
    :param eye_point: 当前面的eye_point
    :param edge_list:
    :return:
    """
    if (plane.dist(eye_point) > 10 ** -10):
        visited_planes.append(plane)
        edges = plane.get_edges()
        for edge in edges:
            neighbour = adjacent_plane(plane, edge, list_of_planes)
            if (neighbour not in visited_planes):
                result = calc_horizon(visited_planes, neighbour, eye_point, edge_list, list_of_planes)
                if (result == 0):
                    edge_list.add(edge)
        return 1
    else:
        return 0
