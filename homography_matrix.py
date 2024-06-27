import numpy as np

from center_normalize_pts import CenterAndNormalizeImagePoints

def estimateHomography(pta, ptb):
    '''
    输入:
        pta: 图像a未归一化的2d点集,2*n
        ptb: 图像b未归一化的2d点集,2*n
    输出:
        homography_matrix: 3*3 (8自由度, 9个元素, 尺度歧义减去一个自由度)
    '''
    normed_pta, normed_mata = CenterAndNormalizeImagePoints(pta)
    normed_ptb, normed_matb = CenterAndNormalizeImagePoints(ptb)
    N = pta.shape[1]
    A = np.zeros((2*N, 9))

    # for i in range(N):
    #     j = i + N
    #     A[i,0] = -normed_pta[0,i]
    #     A[i,1] = -normed_pta[1,i]
    #     A[i,2] = -1
    #     A[i,6] = normed_pta[0,i] * normed_ptb[0,i]
    #     A[i,7] = normed_pta[1,i] * normed_ptb[0,i]
    #     A[i,8] = normed_ptb[0,i]

    #     A[j,3] = -normed_pta[0,i]
    #     A[j,4] = -normed_pta[1,i]
    #     A[j,5] = -1
    #     A[j,6] = normed_pta[0,i] * normed_ptb[1,i]
    #     A[j,7] = normed_pta[1,i] * normed_ptb[1,i]
    #     A[j,8] = normed_ptb[1,i]

    A[0:N,0] = -normed_pta[0,:].transpose()
    A[0:N,1] = -normed_pta[1,:].transpose()
    A[0:N,2] = -1

    A[0:N,6] = (normed_pta[0,:]*normed_ptb[0,:]).transpose()
    A[0:N,7] = (normed_pta[1,:]*normed_ptb[0,:]).transpose()
    A[0:N,8] = normed_ptb[0,:].transpose()

    A[N:2*N,3] = -normed_pta[0,:].transpose()
    A[N:2*N,4] = -normed_pta[1,:].transpose()
    A[N:2*N,5] = -1

    A[N:2*N,6] = (normed_pta[0,:]*normed_ptb[1,:]).transpose()
    A[N:2*N,7] = (normed_pta[1,:]*normed_ptb[1,:]).transpose()
    A[N:2*N,8] = normed_ptb[1,:].transpose()

    # print(A)

    u,d,v = np.linalg.svd(A)
    v = v.transpose() # numpy的特性,v已经是v^T了, 需要把它变换回来
    H = v[:,8]
    H = H.reshape((3,3))
    print('|H| = ', np.sum(H**2))
    # 从右往左看, 先对a点集进行归一化, 然后H变换, 然后再对得到的归一化的b点集反归一化
    H = np.linalg.inv(normed_matb) @ H @ normed_mata
    # print('H = ')
    # print(H)
    return H

def testEstimateHomography():
    H0 = np.array([5, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1])
    H0 = H0.reshape((3,3))
    src = np.array([[4,0], [1,0], [2,1], [10,30]])
    src = src.transpose()
    # print(src.shape)
    src1 = np.vstack((src, [1]*src.shape[1]))
    # print(src1)
    tgt1 = H0 @ src1
    tgt1[0,:] /= tgt1[2,:]
    tgt1[1,:] /= tgt1[2,:]
    tgt1[2,:] = 1
    # print(tgt1)
    tgt = tgt1[0:2,:]
    H = estimateHomography(src, tgt)

    print('src = ')
    print(src)
    print('tgt = ')
    print(tgt)
    print('---------')
    
    aa = H @ src1
    aa[0,:] /= aa[2,:]
    aa[1,:] /= aa[2,:]
    aa[2,:] = 1
    print('H @ src1 = ')
    print(aa)
    print('tgt1 = ')
    print(tgt1)
    print(np.sum(aa - tgt1))
    return

def main():
    testEstimateHomography()

if __name__ == "__main__":
    main()