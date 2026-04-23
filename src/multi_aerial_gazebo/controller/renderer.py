import numpy as np
import matplotlib.pyplot as plt
from controller import rotation_matrix

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
