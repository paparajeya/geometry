from geometry.core.pointcloud import PointCloud
from geometry.core.point import Point

# Create PointClouds
pc1 = PointCloud([[1,2],[3,4],[5,6]], backend="numpy")
pc2 = PointCloud([[2,3],[4,5]], backend="numpy")

# Distance to point
dists = pc1.distance_to_point(Point(0,0))
print(dists)

# Distance matrix
dm = pc1.distance_to_pointcloud(pc2)
print(dm)

# Nearest neighbor (k=1)
dist, idx = pc1.nearest_neighbors(pc2, k=1)
print(dist, idx)


# Create PointClouds
pc1 = PointCloud([[1,2],[3,4],[5,6]], backend="torch")
pc2 = PointCloud([[2,3],[4,5]], backend="torch")

# Distance to point
dists = pc1.distance_to_point(Point(0,0))
print(dists)

# Distance matrix
dm = pc1.distance_to_pointcloud(pc2)
print(dm)

# Nearest neighbor (k=1)
dist, idx = pc1.nearest_neighbors(pc2, k=1)
print(dist, idx)


import math

theta = math.pi/4  # 45 degrees
rotation_matrix = [[math.cos(theta), -math.sin(theta)],
                   [math.sin(theta),  math.cos(theta)]]

pc_rotated = pc1.batch_transform(matrix=rotation_matrix, translate=[1,2])
print(pc_rotated.points)


pc = PointCloud([[1,2],[3,4],[5,6],[7,8],[9,10]], backend="torch")
pc.add_pointcloud(PointCloud([[11,12],[13,14]], backend="torch"))
pc.add_points([[15,16],[17,18]])
print(pc)

# Access all points
pts = pc.points
print(pts[0].x, pts[0].y)  # 1 2
print(pts[1])  # 9 10