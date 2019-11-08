import numpy as np
import math

def calc_angle_from_2_vectors(vec_a, vec_b):
    x = np.inner(vec_a, vec_b)
    s = np.linalg.norm(vec_a)
    t = np.linalg.norm(vec_b)
    theta = np.arccos(x/(s*t))
    return math.degrees(theta)

a = np.array([1,1])
b = np.array([-1,1])
c = np.array([-1,-1])
d = np.array([1,-1])

def calc_gap_between_yaxis_and_vector(x, y):
    t = math.atan2(y, x)
    s = None
    if y >= 0:
        # 第一象限
        if x >= 0:
            s = math.pi/2 - t
        # 第二象限
        else:
            s = -t + math.pi/2
    else:
        # 第三象限
        if x < 0:
            s = -3/2 * math.pi - t
        # 第四象限
        else:
            s = -3/2 * math.pi - t

    return s