import numpy as np

from center_normalize_pts import CenterAndNormalizeImagePoints
from pose import CrossProductMatrix, CheckCheirality

def decomposeEssentialMatrix(E_mat):
    u,d,v = np.linalg.svd(E_mat) # numpy的特性,v已经是v^T了
    if np.linalg.det(u) < 0:
        u *= -1
    if np.linalg.det(v) < 0:
        v *= -1
    
    w = np.array([[0, 1, 0],\
                  [-1, 0, 0],\
                [0, 0, 1]])
    
    R1 = u @ w @ v
    R2 = u @ w.transpose() @ v
    t = u[:,2] / np.linalg.norm(u[:,2])
    return R1, R2, t

def poseFromEssentialMatrix(E_mat, pts1, pts2):
    R1, R2, t = decomposeEssentialMatrix(E_mat)
    # 有四种可能
    R_list = [R1, R2, R1, R2]
    # print('R1:', R1)
    # print('R2:', R2)
    t_list = [t, t, t*(-1), t*(-1)]
    # 尝试四种可能,只有一个合理
    pts3D_best = []
    R_best = None
    t_best = None
    for i in range(4):
        ret, pts3D = CheckCheirality(R_list[i], t_list[i], pts1, pts2)
        if len(pts3D) >= len(pts3D_best):
            pts3D_best = pts3D
            R_best = R_list[i]
            t_best = t_list[i]
    return R_best, t_best, pts3D_best

def EssentialMatrixFromPose(R, t):
    return CrossProductMatrix(t) @ R

def estimateEssentialMatrix(pta, ptb):
    '''
    输入:
        pta: 图像a未归一化的2d点集,2*n
        ptb: 图像b未归一化的2d点集,2*n
    输出:
        essential_matrix: 3*3 (6自由度, R和t各三个, 尺度歧义减去一个自由度)
    '''
    normed_pta, normed_mata = CenterAndNormalizeImagePoints(pta)
    normed_ptb, normed_matb = CenterAndNormalizeImagePoints(ptb)

    n = pta.shape[1]
    # x2' * F * x1 = 0.
    cmatrix = np.zeros((n, 9))
    cmatrix[:,0] = normed_pta[0,:].transpose() * normed_ptb[0,:].transpose()
    cmatrix[:,1] = normed_pta[1,:].transpose() * normed_ptb[0,:].transpose()
    cmatrix[:,2] = normed_ptb[0,:].transpose()
    cmatrix[:,3] = normed_pta[0,:].transpose() * normed_ptb[1,:].transpose()
    cmatrix[:,4] = normed_pta[1,:].transpose() * normed_ptb[1,:].transpose()
    cmatrix[:,5] = normed_ptb[1,:].transpose()
    cmatrix[:,6] = normed_pta[0,:].transpose()
    cmatrix[:,7] = normed_pta[1,:].transpose()
    cmatrix[:,8] = 1

    u,d,v = np.linalg.svd(cmatrix)
    v = v.transpose() # numpy的特性,v已经是v^T了, 需要把它变换回来

    ematrix_t = v[:,8]
    ematrix_t = ematrix_t.reshape((3,3))
    u,d,v = np.linalg.svd(ematrix_t)
    # detF = 0, 最后一个奇异值是0
    tmp = (d[0] + d[1])/2 # 八点法, 前面跟F一样, 只有奇异值这里不同, E矩阵的奇异值前两个相同, 最后一个是0
    d[0] = tmp
    d[1] = tmp
    d[-1] = 0
    E = u @ np.diag(d) @ v # numpy的特性,v已经是v^T了
    E = normed_matb.transpose() @ E @ normed_mata # 注意:normed_matb是转置, 不是求逆
    return E

def testEstimateEssentialMatrix():
    src = np.array([[1.839035, 1.924743], [0.543582, 0.37522], [0.473240, 0.142522], [0.964910, 0.598376],\
                    [0.102388, 0.140092], [15.994343, 9.622164], [0.285901, 0.430055,], [0.091150, 0.254594]])
    tgt = np.array([[1.002114, 1.129644], [1.521742, 1.846002], [1.084332, 0.275134], [0.293328, 0.588992],\
                    [0.839509, 0.087290], [1.779735, 1.11685], [0.878616, 0.602447], [0.642616, 1.028681] ])
    src = src.transpose()
    tgt = tgt.transpose()
    
    E_gt = np.array([[-0.0811666, 0.255449, -0.0478999],\
                     [-0.192392, -0.0531675, 0.119547],\
                    [0.177784, -0.22008, -0.015203]])

    E = estimateEssentialMatrix(src, tgt)
    print(E)
    print(np.allclose(E, E_gt))
    return

def testEssentialMatrixFromPose():
    R = np.eye(3)
    t = np.array([0,0,1])
    E = EssentialMatrixFromPose(R, t)
    E_gt = np.array([[0,-1,0],\
                     [1,0,0],\
                    [0,0,0]])
    print(E)
    print(np.allclose(E, E_gt))

    R = np.eye(3)
    t = np.array([0,0,2])
    E = EssentialMatrixFromPose(R, t)
    E_gt = np.array([[0,-2,0],\
                     [2,0,0],\
                    [0,0,0]])
    print(E)
    print(np.allclose(E, E_gt))
    return

def testPoseFromEssentialMatrix():
    R = np.eye(3)
    t = np.array([1,0,0])
    t = t / np.linalg.norm(t)
    E = EssentialMatrixFromPose(R, t)

    proj_mat1 = np.zeros((3,4))
    proj_mat1[0,0] = 1
    proj_mat1[1,1] = 1
    proj_mat1[2,2] = 1
    proj_mat2 = np.zeros((3,4))
    proj_mat2[0:3,0:3] = R
    proj_mat2[0,3] = t[0]
    proj_mat2[1,3] = t[1]
    proj_mat2[2,3] = t[2]

    pts3D = np.array([[0,0,1], [0,0.1,1], [0.5,0.3,1],\
                    [0.1,0,1], [0.1,0.1,1], [0.3,0.5,1]])
    pts3D = pts3D.transpose()

    pts3D_homogeneous = np.vstack((pts3D, [1]*pts3D.shape[1]))
    # print(pts3D_homogeneous)

    pts2d1 = []
    pts2d2 = []
    for i in range(pts3D_homogeneous.shape[1]):
        pt2d1 = proj_mat1 @ pts3D_homogeneous[:,i]
        pt2d2 = proj_mat2 @ pts3D_homogeneous[:,i]
        pt2d1 = pt2d1 / pt2d1[-1]
        pt2d2 = pt2d2 / pt2d2[-1]
        pts2d1.append(pt2d1[0:2])
        pts2d2.append(pt2d2[0:2])
    pts2d1 = np.array(pts2d1)
    pts2d1 = pts2d1.transpose()
    pts2d2 = np.array(pts2d2)
    pts2d2 = pts2d2.transpose()

    R_best, t_best, pts3D_best = poseFromEssentialMatrix(E, pts2d1, pts2d2)
    print(len(pts3D_best))
    print(np.allclose(R_best, R))
    print(np.allclose(t_best, t))
    return

def main():
    # testEstimateEssentialMatrix()
    # testEssentialMatrixFromPose()
    testPoseFromEssentialMatrix()
    pass

if __name__ == "__main__":
    main()
