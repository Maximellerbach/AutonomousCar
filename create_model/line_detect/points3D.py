import numpy as np
import cv2
import open3d as o3d

class point():
    def __init__(self, kp, index):
        self.kp = kp
        self.pt = (self.kp.pt[0], self.kp.pt[1], 0)
        self.index = index

    def set_point(self, point):
        self.pt = point


class cloud_points():
    def __init__(self):
        self.u = (1,1,1)
        self.mesh = o3d.geometry.PointCloud()
        self.colors = []
        self.pointcloud = []
        self.objs = [self.mesh]

    def create_mesh(self):
        x = np.linspace(-3, 3, 401)
        mesh_x, mesh_y = np.meshgrid(x, x)
        z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
        z_norm = (z - z.min()) / (z.max() - z.min())

        xyz = np.zeros((np.size(mesh_x), 3))
        xyz[:, 0] = np.reshape(mesh_x, -1)
        xyz[:, 1] = np.reshape(mesh_y, -1)
        xyz[:, 2] = np.reshape(z_norm, -1)
        
        self.mesh.points = o3d.utility.Vector3dVector(xyz)

    def add_points(self, points):
        self.pointcloud += points

    def set_points(self, points):
        self.pointcloud = points

    def load_mesh(self, path=""):
        self.mesh = o3d.io.read_point_cloud(path)

    def save_mesh(self, path=""):
        o3d.io.write_point_cloud(path, self.mesh)

    def display_mesh(self, width=640, height=360):
        self.mesh.points = o3d.utility.Vector3dVector(self.pointcloud)
        o3d.visualization.draw_geometries(self.objs, width=width, height=height)


if __name__ == "__main__":
    cp = cloud_points()
    cp.create_mesh()
    cp.display_mesh()