import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
class ICP():
    def __init__(self, src_pts, tgt_pts, dis_thrd):
        '''
        输入：src_pts, tgt_pts, N*3，N>=3
        '''
        self.m_src_pts = src_pts
        self.m_tgt_pts = tgt_pts
        assert self.m_src_pts.shape[0] >= 3
        assert self.m_tgt_pts.shape[0] >= 3
        self.m_dis_thrd = dis_thrd
        self.m_kdtree = None
        self.m_rotation = np.eye(3)
        self.m_translation = np.zeros((3,1))

        print('init:',self.m_tgt_pts.shape)
        return
    
    def buildKdtree(self):
        self.m_kdtree = KDTree(data=self.m_tgt_pts)
        return

    def findCorrs(self):
        src_corr_pts = None
        tgt_corr_pts = None
        new_src_pts = self.m_rotation @ self.m_src_pts.transpose() + self.m_translation
        dis, idx = self.m_kdtree.query(new_src_pts.transpose(), k=1)
        selected = (dis <= self.m_dis_thrd)
        src_corr_pts = self.m_src_pts[selected]
        tgt_corr_pts = self.m_tgt_pts[idx[selected]]
        return (src_corr_pts.transpose(), tgt_corr_pts.transpose())

    def linearSolveTransform(self, src_corr_pts, tgt_corr_pts):
        new_src_pts = self.m_rotation @ src_corr_pts + self.m_translation
        print('new_src_pts.shape:',new_src_pts.shape)
        print('tgt_corr_pts.shape:',tgt_corr_pts.shape)
        # delete the mean vector
        src_mean = np.mean(new_src_pts, axis=1).reshape(3,-1)
        tgt_mean = np.mean(tgt_corr_pts, axis=1).reshape(3,-1)
        src_del_mean_pts = new_src_pts - src_mean
        tgt_del_mean_pts = tgt_corr_pts - tgt_mean
        # print('src_del_mean_pts.shape:',src_del_mean_pts.shape)
        # print('tgt_del_mean_pts.shape:',tgt_del_mean_pts.shape)
        # covariance matrix
        cov_mat = np.zeros((3,3))
        for idx in range(new_src_pts.shape[1]):
            # warnning: tgt is first
            cov_mat += np.outer(tgt_del_mean_pts[:,idx],src_del_mean_pts[:,idx])
        # print('cov_mat.shape:', cov_mat.shape)
        cov_mat = cov_mat * (1.0/new_src_pts.shape[1])
        # svd cov matrix
        u,d,v = np.linalg.svd(cov_mat)
        s = np.eye(3)
        if np.linalg.det(u) * np.linalg.det(v) < 0:
            s[2,2] = -1
        rotation = u @ s @ v
        # print('rotation check:', rotation @ rotation.transpose())
        # calculate translation
        translation = tgt_mean - rotation @ src_mean
        # update transform
        self.m_rotation = rotation @ self.m_rotation
        self.m_translation = rotation @ self.m_translation + translation
        return
    
    def interativeLST(self):
        self.buildKdtree()
        for idx in range(3):
            # print('before:', idx)
            # print(self.m_rotation)
            # print(self.m_translation)
            src_corr_pts, tgt_corr_pts = self.findCorrs()
            self.linearSolveTransform(src_corr_pts, tgt_corr_pts)
            print('after:', idx)
            print(self.m_rotation)
            print(self.m_translation)
    
    def getTransform(self):
        return self.m_rotation, self.m_translation

def kdtree_test():
    rng = np.random.default_rng()
    src_pts = rng.random((10,3))
    tgt_pts = rng.random((15,3))
    print(src_pts)
    tgt_kdtree = KDTree(data=tgt_pts)

    dd, ii = tgt_kdtree.query(x=src_pts, k=1)
    print(dd)
    print(ii)

    selected = (dd <= 0.02)
    print('selected = ', selected)

    src_corr_pts = src_pts[selected]
    tgt_corr_pts = tgt_pts[ii[selected]]

    print(src_corr_pts)
    print(tgt_corr_pts)

if __name__ == '__main__':
    # kdtree_test()
    # rng = np.random.default_rng()
    # src_pts = rng.random((100,3))
    src_pts = []
    for idx in range(-10,11):
        src_pts.append([idx*1, 0, 0])
    for idx in range(-20,21):
        src_pts.append([0, idx*1, 0])
    src_pts = np.array(src_pts)
    # rot_gt = Rotation.random().as_matrix()
    # rot_gt = np.array([[0.9999238, -0.0087262,  0.0087265],\
    #             [0.0088024,  0.9999232, -0.0087262],\
    #             [-0.0086497,  0.0088024,  0.9999238]])
    rot_gt = np.array([[0.1731782,  0.3784012, -0.9092974],\
                [-0.0343220,  0.9250051,  0.3784012],\
                [0.9842923, -0.0343220,  0.1731782]])
    # translation_gt = rng.random((1,3))
    translation_gt = np.array([[0.4, 0.20, -0.20]])

    tgt_pts = rot_gt @ src_pts.transpose() + translation_gt.reshape(3,-1)
    tgt_pts = tgt_pts.transpose()

    icp = ICP(src_pts, tgt_pts, 10)
    icp.interativeLST()
    R,t = icp.getTransform()
    # print(R)
    # print(t)
