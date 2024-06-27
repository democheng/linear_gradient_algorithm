import numpy as np

from center_normalize_pts import CenterAndNormalizeImagePoints

def estimateFundamentalMatrix(pta, ptb):
    '''
    输入:
        pta: 图像a未归一化的2d点集,2*n
        ptb: 图像b未归一化的2d点集,2*n
    输出:
        fundamental_matrix: 3*3 (7自由度, 9个元素, 尺度歧义减去一个自由度, detF=0再减去一个自由度)
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
    d[-1] = 0
    F = u @ np.diag(d) @ v # numpy的特性,v已经是v^T了
    F = normed_matb.transpose() @ F @ normed_mata # 注意:normed_matb是转置, 不是求逆
    return F

def testEstimateFundamentalMatrix():
    src = np.array([[1.839035, 1.924743], [0.543582, 0.37522], [0.473240, 0.142522], [0.964910, 0.598376],\
                    [0.102388, 0.140092], [15.994343, 9.622164], [0.285901, 0.430055,], [0.091150, 0.254594]])
    tgt = np.array([[1.002114, 1.129644], [1.521742, 1.846002], [1.084332, 0.275134], [0.293328, 0.588992],\
                    [0.839509, 0.087290], [1.779735, 1.11685], [0.878616, 0.602447], [0.642616, 1.028681] ])
    src = src.transpose()
    tgt = tgt.transpose()
    
    F_gt = np.array([[-0.217859, 0.419282, -0.0343075],\
                     [-0.0717941, 0.0451643, 0.0216073],\
                    [0.248062, -0.429478, 0.0221019]])

    F = estimateFundamentalMatrix(src, tgt)
    print(F)
    print(np.allclose(F, F_gt))
    return

def main():
    testEstimateFundamentalMatrix()
    pass

if __name__ == "__main__":
    main()