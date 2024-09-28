import math
import numpy as np

def RandNDimData():
    count = 100000
    data = np.zeros((count, 3))

    # 3dof, 95%
    chi_squre_threshold = 7.815

    mu0 = -1
    mu1 = 25
    mu2 = 8
    mu_np = np.array([mu0, mu1, mu2])
    cov_mat = np.array([[10, 0.01, 0.1], [0.4, 8, 0.3], [0.001, 0.002, 6]])
    data = np.random.multivariate_normal(mu_np, cov_mat, count)
    print(data.shape)
    mu_np = mu_np.reshape(1,3)
    info_mat = np.linalg.inv(cov_mat)

    total = 0
    for i in range(count):
        delta = data[i, :] - mu_np
        tmp = delta @ info_mat @ np.transpose(delta)
        # print(tmp)
        if tmp > chi_squre_threshold:
            total = total + 1
    percent = total / count
    print(percent)
    return percent

if __name__ == '__main__':
    percent_total = 0
    loop = 10
    for i in range(loop):
        percent_total = percent_total + RandNDimData()
    percent_total = percent_total / loop
    print("percent_mean = ", percent_total)