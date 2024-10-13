import open3d as o3d

print("Open3D version:", o3d.__version__)

mesh = o3d.geometry.TriangleMesh.create_sphere()
o3d.visualization.draw_geometries([mesh])