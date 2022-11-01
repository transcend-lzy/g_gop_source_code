from utils_hull import *

class Plane:
	def __init__(self, pointA, pointB, pointC):
		self.pointA = pointA
		self.pointB = pointB
		self.pointC = pointC
		self.normal = None
		self.distance = None
		self.calcNorm()
		self.to_do = set()
		self.edge1 = Edge(pointA, pointB)
		self.edge2 = Edge(pointB, pointC)
		self.edge3 = Edge(pointC, pointA)

	def calcNorm(self):
		point1 = self.pointA - self.pointB
		point2 = self.pointB - self.pointC
		#三角形两条线叉乘可以得到法向量
		normVector = cross(point1 ,point2)
		length = normVector.length()
		#法向量是单位向量
		normVector.x = normVector. x /length
		normVector.y = normVector. y /length
		normVector.z = normVector. z /length
		self.normal = normVector  #noraml是法向量
		self.distance = dot(self.normal ,self.pointA)

	def dist(self, pointX):
		return dot ( self.normal, pointX - self.pointA)

	def get_edges(self):
		return [self.edge1, self.edge2, self.edge3]

	def calculate_to_do(self, points, temp=None):
		"""
		self.to_do包含了面对应的点集
		点集内的点都能看到该面
		看某一点点与面的某一顶点的连线和面的法线是锐角还是钝角，如果是锐角则表示点能看见面，则加入到该面的点集
		:param points:
		:param temp:
		:return:
		"""
		dist_thre = 10 ** (-5)
		if (temp != None):
			for p in temp:
				dist = self.dist(p)
				if dist > dist_thre:
					self.to_do.add(p)
		else:
			for p in points:
				dist = self.dist(p)
				if dist > dist_thre:
					self.to_do.add(p)

	def __eq__(self ,other):
		# print 'Checking Plane Equality'
		return checker_plane(self ,other)

	def __str__(self):
		string =  "Plane : "
		string += "\n\tX:  " +str(self.pointA.x ) +", " +str(self.pointA.y ) +", " +str(self.pointA.z)
		string += "\n\tY:  " +str(self.pointB.x ) +", " +str(self.pointB.y ) +", " +str(self.pointB.z)
		string += "\n\tZ:  " +str(self.pointC.x ) +", " +str(self.pointC.y ) +", " +str(self.pointC.z)
		string += "\n\tNormal:  " +str(self.normal.x ) +", " +str(self.normal.y ) +", " +str(self.normal.z)
		return string

	def __hash__(self):
		return hash((self.pointA ,self.pointB ,self.pointC))
