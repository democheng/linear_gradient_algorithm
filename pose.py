import numpy as np
from similarity_transform import similarityTransform3

def CrossProductMatrix(vec3):
    mat = np.array([[0, -vec3[2],  vec3[1]],\
                    [vec3[2], 0, -vec3[0]],\
                    [-vec3[1], vec3[0], 0]])
    return mat

def triangulatePoint(proj_mat1, proj_mat2, pt1, pt2):
    # 三角化一个点
    A = np.zeros((4,4))
    A[0,:] = pt1[0]*proj_mat1[2,:] - proj_mat1[0,:]
    A[1,:] = pt1[1]*proj_mat1[2,:] - proj_mat1[1,:]
    A[2,:] = pt2[0]*proj_mat2[2,:] - proj_mat2[0,:]
    A[3,:] = pt2[1]*proj_mat2[2,:] - proj_mat2[1,:]
    # SVD分解
    u,d,v = np.linalg.svd(A)
    v = v.transpose()
    ans = v[:,3]
    ans = ans / ans[-1]
    return ans[0:3]

def triangulatePoints(proj_mat1, proj_mat2, pts1, pts2):
    assert pts1.shape == pts2.shape
    n = pts1.shape[1]
    pts3D = []
    for i in range(n):
        pts3D.append(triangulatePoint(proj_mat1, proj_mat2, pts1[:,i], pts2[:,i]))
    return pts3D

def calculateDepth(proj_mat, pt3D):
    pt3Dhomo = np.ones((4,1))
    pt3Dhomo[0:3,0] = pt3D[:]
    proj_z = proj_mat[2,:] @ pt3Dhomo
    ans = proj_z[0] * np.linalg.norm(proj_mat[:,2])
    return ans

def hasPointPositiveDepth(proj_mat, pt3D):
    pt3Dhomo = np.ones((4,1))
    pt3Dhomo[0:3,0] = pt3D[:]
    proj_z = proj_mat[2,:] @ pt3Dhomo
    return proj_z > 0

def CheckCheirality(R, t, pts1, pts2):
    assert pts1.shape == pts2.shape
    proj_mat1 = np.zeros((3,4))
    proj_mat1[0,0] = 1
    proj_mat1[1,1] = 1
    proj_mat1[2,2] = 1

    proj_mat2 = np.zeros((3,4))
    proj_mat2[0:3,0:3] = R
    proj_mat2[0:3,3] = t

    max_depth = 1000 * np.linalg.norm(R.transpose() @ t)
    n = pts1.shape[1]
    pt3d_list = list()
    for i in range(n):
        pt3d = triangulatePoint(proj_mat1, proj_mat2, pts1[:,i], pts2[:,i])
        depth1 = calculateDepth(proj_mat1, pt3d)
        if 0 < depth1 < max_depth:
            depth2 = calculateDepth(proj_mat2, pt3d)
            if 0 < depth2 < max_depth:
                pt3d_list.append(pt3d)
    return len(pt3d_list) > 0, pt3d_list

def testTriangulatePoint():
    pts3D = np.array([[0, 0.1, 0.1], [0, 1, 3], [0 ,1, 2],\
                    [0.01, 0.2, 3], [-1, 0.1, 1], [0.1, 0.1, 0.2]])
    pts3D = pts3D.transpose()

    proj_mat1 = np.zeros((3,4))
    proj_mat1[0,0] = 1
    proj_mat1[1,1] = 1
    proj_mat1[2,2] = 1

    qz_list = np.arange(0, 1, 0.2)
    tx_list = np.arange(0, 10, 2)

    for qz in qz_list:
        for tx in tx_list:
            sim3_mat = similarityTransform3(1.0, np.array([0.2, 0.3, 0.4, qz]), np.array([tx, 2, 3]))
            proj_mat2 = sim3_mat[0:3,:]

            for i in range(pts3D.shape[1]):
                pt3Dhomo = np.ones((4,1))
                pt3Dhomo[0:3,0] = pts3D[:,i]

                pt2d1 = proj_mat1 @ pt3Dhomo
                pt2d2 = proj_mat2 @ pt3Dhomo
                pt2d1 = pt2d1 / pt2d1[-1,0]
                pt2d2 = pt2d2 / pt2d2[-1,0]

                tri_point3D = triangulatePoint(proj_mat1, proj_mat2, pt2d1[0:2], pt2d2[0:2])
                # print(tri_point3D)
                delta_pt3D = tri_point3D - pts3D[:,i]
                assert np.linalg.norm(delta_pt3D) < 1e-9
    print('test ok...')

def testCheckCheirality():
    R = np.eye(3)
    t = np.array([1,0,0])
    points1 = np.array([[0,0]])
    points2 = np.array([[0.1,0]])
    points1 = points1.transpose()
    points2 = points2.transpose()
    ret, pts3D = CheckCheirality(R, t, points1, points2)
    print(ret, pts3D)

    points1 = np.array([[0,0], [0,0]])
    points2 = np.array([[0.1,0], [-0.1,0]])
    points1 = points1.transpose()
    points2 = points2.transpose()
    ret, pts3D = CheckCheirality(R, t, points1, points2)
    print(ret, pts3D)

    points1 = np.array([[0,0], [0,0]])
    points2 = np.array([[0.1,0], [0.2,0]])
    points1 = points1.transpose()
    points2 = points2.transpose()
    ret, pts3D = CheckCheirality(R, t, points1, points2)
    print(ret, pts3D)

    points1 = np.array([[0,0], [0,0]])
    points2 = np.array([[-0.1,0], [-0.2,0]])
    points1 = points1.transpose()
    points2 = points2.transpose()
    ret, pts3D = CheckCheirality(R, t, points1, points2)
    print(ret, pts3D)
    return

def main():
    testTriangulatePoint()
    testCheckCheirality()
    pass

if __name__ == "__main__":
    main()