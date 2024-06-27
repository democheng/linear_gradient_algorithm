import numpy as np
import math
def CenterAndNormalizeImagePoints(points):
    '''
    输入:
        points: 2*n的2d点集
    输出:
        normed_points: 2*n的2d点集
        matrix: 归一化矩阵
    '''
    centroid = np.mean(points, axis=1)
    centroid = centroid.reshape(-1,1)
    # print(centroid)
    rms_mean_dist = np.sqrt(np.sum((points - centroid)**2) / float(points.shape[1]))
    # print(rms_mean_dist)
    # 构造归一化矩阵
    norm_factor = math.sqrt(2.0) / rms_mean_dist
    matrix = np.array([[norm_factor, 0, -norm_factor*centroid[0,0]],\
                        [0, norm_factor, -norm_factor*centroid[1,0]],\
                        [0,0,1]])
    # print(matrix)
    points1 = np.vstack((points, [1]*points.shape[1]))
    normed_pt1 = matrix @ points1
    normed_pt1[0,:] /= normed_pt1[2,:]
    normed_pt1[1,:] /= normed_pt1[2,:]
    return normed_pt1[0:2,:], matrix

def testCenterAndNormalizeImagePoints():
    points = np.zeros((2,11))
    for i in range(11):
        points[0,i] = i
        points[1,i] = i
    normed_pt1, matrix = CenterAndNormalizeImagePoints(points)
    print(matrix)
    matrix_gt = np.array([[0.31622776601683794, 0, -1.5811388300841898],\
                        [0, 0.31622776601683794, -1.5811388300841898],\
                        [0,0,1]])
    print(np.allclose(matrix_gt, matrix))
    centroid = np.mean(normed_pt1, axis=1)
    print(centroid)

def main():
    testCenterAndNormalizeImagePoints()

if __name__ == "__main__":
    main()