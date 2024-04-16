import numpy as np
import math
import matplotlib.pyplot as plt
# x_k: n*1, state vector
# w_k: n*1, process noise vector
# z_k: m*1, observation vector
# v_k: m*1, measurement noise vector
# f(): n*1, process nolinear vector function
# h(): m*1, observation nolinear vector function
# Q_k: n*n, process noise covariance matrix
# R_k: m*m, measurement noise covariance matrix

class ExtendedKalmanFilter(object):
    def __init__(self, x, w, z, v, f, h, Jf, Jh, P) -> None:
        assert x.shape == w.shape
        assert z.shape == v.shape
        assert P.shape[0] == x.shape[0]
        assert P.shape[0] == P.shape[1]
        self.m_x = x
        self.m_w = w
        self.m_z = z
        self.m_v = v
        self.m_f = f
        self.m_h = h
        self.m_Q = w @ w.transpose() # n*1 * 1*n
        self.m_R = v @ v.transpose()
        self.m_Jf = Jf
        self.m_Jh = Jh
        self.m_P = P
        self.m_x_I = np.identity(x.shape[0])

    def predictStep(self):
        Jfx = self.m_Jf(self.m_x)

        self.m_x = self.m_f(self.m_x, self.m_Q)
        self.m_P = Jfx @ self.m_P @ Jfx.transpose() + self.m_Q
        return
    
    def updateStep(self, z_cur):
        Jhx = self.m_Jh(self.m_x)
        S = Jhx @ self.m_P @ Jhx.transpose() + self.m_R
        K = self.m_P @ Jhx @ np.linalg.inv(S)

        self.m_x = self.m_x + K @ (z_cur  - self.m_h(self.m_x))
        self.m_P = (self.m_x_I - K @ Jhx) @ self.m_P
        return

def list_to_npn1(x):
    return np.array(x).reshape(-1,1)

def n1np_to_list(x):
        return list(x.reshape(1, x.shape[0])[0])

def test_process_func(x_pre, Q=None):
    # x = [x, y, vx, vy]
    assert x_pre.shape[0] == 4
    assert x_pre.shape[1] == 1
    x_cur = np.zeros(x_pre.shape)
    dt = 0.1
    x_cur[0] = x_pre[0] + dt * x_pre[2]
    x_cur[1] = x_pre[1] + dt * x_pre[3]
    x_cur[2] = x_pre[2] - 0.02 # velocity is being slower, and finally change direction
    x_cur[3] = x_pre[3] - 0.01 # velocity is being slower, and finally change direction
    if Q is not None:
        x_noise = np.random.multivariate_normal([0,0,0,0], Q)
        x_noise = list_to_npn1(x_noise)
        x_cur = x_cur + x_noise
    return x_cur

def test_J_process_wrt_x(x_pre):
    assert x_pre.shape[0] == 4
    assert x_pre.shape[1] == 1
    dt = 0.1
    J_x = np.array([[1,  0,  dt,  0],\
                    [0,  1,   0,  dt],\
                    [0,  0,   1,  0],\
                    [0,  0,   0,  1]])
    return J_x

def test_observation_func(x_cur, R=None):
    # [distance, yaw].T
    # the anchor is at [100.0, 100.0]
    delta_x = x_cur[0,0] - 100.0
    delta_y = x_cur[1,0] - 100.0
    distance = math.sqrt(delta_x**2 + delta_y**2)
    yaw = math.atan(delta_y / delta_x)
    obs = np.array([distance, yaw]).reshape(-1,1)
    if R is not None:
        obs_noise = np.random.multivariate_normal([0,0], R)
        obs_noise = list_to_npn1(obs_noise)
        obs = obs + obs_noise
    return obs

def test_J_observation_wrt_x(x_cur):
    delta_x = x_cur[0,0] - 100.0
    delta_y = x_cur[1,0] - 100.0
    distance = math.sqrt(delta_x**2 + delta_y**2)
    dis_inv = 1.0 / distance
    yaw = math.atan(delta_y / delta_x)
    J_x = np.array([[math.cos(yaw),             math.sin(yaw),             0,  0],\
                    [-math.sin(yaw) * dis_inv,  math.cos(yaw) * dis_inv,   0,  0]])
    return J_x



def test_example_ekf():
    x = list_to_npn1([0, 0, 1.5, 1.3])
    w = list_to_npn1([0.05, 0.05, 0.00001, 0.00001])
    z = list_to_npn1([100, math.pi*0.25])
    v = list_to_npn1([0.00001, math.pi*0.0001])
    R = v @ v.transpose()
    P = np.array([[0.0001,  0,  0,  0],\
                [0,  0.0001,   0,  0],\
                [0,  0,   0.0001,  0],\
                [0,  0,   0,  0.0001]])
    test_ekf = ExtendedKalmanFilter(x,w,z,v,test_process_func,test_observation_func,\
                                    test_J_process_wrt_x,test_J_observation_wrt_x, P)
    count = 200
    x_gt_list = [x]
    z_noise_list = [z]
    count_list = [0]
    for idx in range(1, count):
        count_list.append(idx)
        x_gt = test_process_func(x_gt_list[-1])
        x_gt_list.append(x_gt)
        z_noise = test_observation_func(x_gt, R)
        z_noise_list.append(z_noise)
    
    x_ekf_list = [x]
    for idx in range(1, count):
        test_ekf.predictStep()
        if idx % 10 == 0:
            test_ekf.predictStep()
        x_ekf_list.append(test_ekf.m_x)
    
    x_gt_plt_list = list()
    x_ekf_plt_list = list()
    for idx in range(count):
        tmp = n1np_to_list(x_gt_list[idx])
        x_gt_plt_list.append([tmp[0], tmp[1]])
        tmp = n1np_to_list(x_ekf_list[idx])
        x_ekf_plt_list.append([tmp[0], tmp[1]])
    x_gt_plt_list = np.array(x_gt_plt_list)
    x_ekf_plt_list = np.array(x_ekf_plt_list)
    plt.plot(x_gt_plt_list[:, 0], x_gt_plt_list[:, 1])
    plt.plot(x_ekf_plt_list[:, 0], x_ekf_plt_list[:, 1])
    plt.show()
    return

if __name__ == '__main__':
    test_example_ekf()