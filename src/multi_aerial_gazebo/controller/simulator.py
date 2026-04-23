# Generated with the assistance of generative AI

import numpy as np
import heapq

class KDNode:
    def __init__(self, point, index, left=None, right=None):
        self.point = point
        self.index = index
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, points):
        self.points = np.array(points)
        self.root = self._build(np.arange(len(points)), depth=0)

    def _build(self, indices, depth):
        if len(indices) == 0:
            return None
        k = self.points.shape[1]
        axis = depth % k
        # Sort indices based on point coordinates along the axis
        sorted_indices = indices[np.argsort(self.points[indices, axis])]
        median = len(sorted_indices) // 2
        return KDNode(
            point=self.points[sorted_indices[median]],
            index=sorted_indices[median],
            left=self._build(sorted_indices[:median], depth + 1),
            right=self._build(sorted_indices[median+1:], depth + 1)
        )

    def query(self, target, k_neighbors):
        """
        Query the k nearest neighbors for a target point.
        Returns a list of indices of the k nearest points.
        """
        neighbors = [] # Max-heap of (-dist, index) to track k smallest distances
        self._query_node(self.root, target, k_neighbors, 0, neighbors)
        # Return indices of found neighbors
        return [n[1] for n in sorted(neighbors, key=lambda x: -x[0])]

    def _query_node(self, node, target, k_neighbors, depth, neighbors):
        if node is None:
            return
        
        dist = np.linalg.norm(target - node.point)
        
        if len(neighbors) < k_neighbors:
            heapq.heappush(neighbors, (-dist, node.index))
        elif dist < -neighbors[0][0]:
            heapq.heapreplace(neighbors, (-dist, node.index))

        axis = depth % target.shape[0]
        diff = target[axis] - node.point[axis]
        
        nearer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        
        self._query_node(nearer, target, k_neighbors, depth + 1, neighbors)
        
        if abs(diff) < -neighbors[0][0] or len(neighbors) < k_neighbors:
            self._query_node(farther, target, k_neighbors, depth + 1, neighbors)

def rotation_matrix(phi, theta, psi):
    R = np.array([
        [np.cos(psi)*np.cos(theta) - np.sin(phi)*np.sin(psi)*np.sin(theta),
         -np.cos(phi)*np.sin(psi),
         np.cos(psi)*np.sin(theta) + np.cos(theta)*np.sin(phi)*np.sin(psi)],
        [np.sin(psi)*np.cos(theta) + np.cos(psi)*np.sin(phi)*np.sin(theta),
         np.cos(phi)*np.cos(psi),
         np.sin(psi)*np.sin(theta) - np.cos(psi)*np.cos(theta)*np.sin(phi)],
        [-np.cos(phi)*np.sin(theta),
         np.sin(phi),
         np.cos(phi)*np.cos(theta)]
    ])
    return R

def quad_dynamics(x, u1, u2, u3, u4, m, g, Ix, Iy, Iz):
    phi, theta, psi = x[3], x[4], x[5]
    vx, vy, vz = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]

    R = rotation_matrix(phi, theta, psi)

    dx = np.zeros(12)

    # Positions
    dx[0:3] = [vx, vy, vz]

    # Euler angles
    dx[3] = p + r * np.sin(theta)
    dx[4] = p * np.sin(theta) * np.tan(phi) + q - r * np.cos(theta) * np.tan(phi)
    dx[5] = -p * np.sin(theta) / np.cos(phi) + r * np.cos(theta) / np.cos(phi)

    # Velocities
    dx[6:9] = (1 / m) * R @ np.array([0, 0, u1]) + np.array([0, 0, -g])

    # Angular rates
    dx[9] = ((Iy - Iz) / Ix) * q * r + u2 / Ix
    dx[10] = ((Iz - Ix) / Iy) * p * r + u3 / Iy
    dx[11] = u4 / Iz

    return dx

def sd_box(p, center, b):
    """
    Signed Distance Function (SDF) for a box.
    p: current position (3D vector)
    center: box center (3D vector)
    b: box half-extents (3D vector)
    """
    q = np.abs(p - center) - b
    # distance = length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0)
    return np.linalg.norm(np.maximum(q, 0.0)) + min(np.max(q), 0.0)
