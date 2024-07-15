import numpy as np
from transformations import quaternion_matrix

def similarityTransform3(scale, quat_wxyz, t_vec3):
    quat_wxyz /= np.linalg.norm(quat_wxyz) # 归一化,保证定义的合法性
    sim3_mat = quaternion_matrix(quat_wxyz)
    sim3_mat[0:3,0:3] = sim3_mat[0:3,0:3] * scale
    sim3_mat[0,3] = t_vec3[0]
    sim3_mat[1,3] = t_vec3[1]
    sim3_mat[2,3] = t_vec3[2]
    return sim3_mat

