import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from simulator import rotation_matrix

def draw_quad(ax, x):
    L = 0.1
    phi, theta, psi = x[3], x[4], x[5]
    p = x[0:3]

    R = rotation_matrix(phi, theta, psi)

    # Arm positions
    arm1 = R @ np.array([[L, -L], [0, 0], [0, 0]])
    arm2 = R @ np.array([[0, 0], [L, -L], [0, 0]])

    h = []
    # Arms
    h.append(ax.plot([p[0] + arm1[0, 0], p[0] + arm1[0, 1]],
                     [p[1] + arm1[1, 0], p[1] + arm1[1, 1]],
                     [p[2] + arm1[2, 0], p[2] + arm1[2, 1]],
                     'r', linewidth=3)[0])
    h.append(ax.plot([p[0] + arm2[0, 0], p[0] + arm2[0, 1]],
                     [p[1] + arm2[1, 0], p[1] + arm2[1, 1]],
                     [p[2] + arm2[2, 0], p[2] + arm2[2, 1]],
                     'b', linewidth=3)[0])
    # Body
    h.append(ax.scatter(p[0], p[1], p[2], c='k', marker='o'))
    return h

def draw_box(ax, center, extents, color='g', alpha=0.1):
    """
    Renders a semi-transparent cuboid in 3D.
    center: box center (3D vector)
    extents: half-widths for x, y, z (3D vector)
    """
    dx, dy, dz = extents
    cx, cy, cz = center
    
    # Define the 8 vertices of the box
    v = np.array([
        [cx-dx, cy-dy, cz-dz], [cx+dx, cy-dy, cz-dz], 
        [cx+dx, cy+dy, cz-dz], [cx-dx, cy+dy, cz-dz],
        [cx-dx, cy-dy, cz+dz], [cx+dx, cy-dy, cz+dz], 
        [cx+dx, cy+dy, cz+dz], [cx-dx, cy+dy, cz+dz]
    ])
    
    # Define the 6 rectangular faces by indexing into the vertices
    faces = [
        [v[0], v[1], v[2], v[3]], # bottom
        [v[4], v[5], v[6], v[7]], # top
        [v[0], v[1], v[5], v[4]], # front
        [v[2], v[3], v[7], v[6]], # back
        [v[1], v[2], v[6], v[5]], # right
        [v[0], v[3], v[7], v[4]]  # left
    ]
    
    poly = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors=color, alpha=alpha)
    ax.add_collection3d(poly)
    return poly
