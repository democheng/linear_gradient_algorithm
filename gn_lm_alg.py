import numpy as np
import matplotlib.pyplot as plt
def f1(x1, x2, x3):
    return x1 + x2 + x3

def f2(x1, x2, x3):
    return 5.0 * x3 * x3 * x3

def f3(x1, x2, x3):
    return 4.0 * x2 * x2 - 2.0 * x3

def f4(x1, x2, x3):
    return 5.0 * x1 * x1 + 3.0 * x2

def costFunc(x1, x2, x3, v1, v2, v3, v4):
    error = np.array([f1(x1, x2, x3), f2(x1, x2, x3), f3(x1, x2, x3), f4(x1, x2, x3)]) - \
            np.array([v1, v2, v3, v4])
    return error.reshape(-1, 1)

def JfMatrix(x1, x2, x3):
    #        dx1     dx2     dx3
    # df1    1.0     0.0     0.0 
    # df2    0.0     0.0     5.0
    # df3    0.0     8.0*x2  2.0
    # df4    10.0*x1 3.0     0.0
    return np.array([
                    [1.0, 1.0, 1.0], \
                    [0.0, 0.0, 15.0*x3*x3], \
                    [0.0, 8.0*x2, 2.0], \
                    [10.0*x1, 3.0, 0.0]
                    ])

def GNupdateParam(x_vec0, cur_error):
    ratio = 1000.0
    J0 = JfMatrix(x_vec0[0,0], x_vec0[1,0], x_vec0[2,0])
    H0 = (J0.transpose() @ J0 + ratio * np.identity(3))
    
    dx = -np.linalg.pinv(H0) @ J0.transpose() @ cur_error
    dx_norm = np.linalg.norm(dx)
    x_vec1 = x_vec0 + dx
    print('dx_norm = ', dx_norm)
    print('H0 = ', H0)
    print('dx = ', dx)
    if dx_norm < 1e-6:
        return x_vec1.reshape(-1, 1), False
    return x_vec1.reshape(-1, 1), True

def testGlGradient():
    x1 = 0.0
    x2 = 0.0 
    x3 = 0.0
    v1 = 20.0
    v2 = 50.0
    v3 = 1000.0 
    v4 = 100.0
    count = 1
    x_vec_opted = np.array([x1, x2, x3])
    x_vec_opted = x_vec_opted.reshape(-1,1)
    print('count = ', count, ' , x_vec_init = ', x_vec_opted)
    init_error = costFunc(x_vec_opted[0,0], x_vec_opted[1,0], x_vec_opted[2,0], v1, v2, v3, v4)
    count_list = list()
    error_list = list()
    while True:
        cur_error = costFunc(x_vec_opted[0,0], x_vec_opted[1,0], x_vec_opted[2,0], v1, v2, v3, v4)
        cur_error_norm = np.linalg.norm(cur_error)
        error_list.append(cur_error_norm)
        count_list.append(count)
        if cur_error_norm < 1e-8:
            print('cur_error is too small, break ...')
            break
        x_vec_opted, is_ok = GNupdateParam(x_vec_opted, cur_error)
        if count % 1 == 0:
            print('count = ', count, ' , cur_error_norm = ', cur_error_norm)
            print('count = ', count, ' , x_vec_opted = ', x_vec_opted[0,0], x_vec_opted[1,0], x_vec_opted[2,0])
        if not is_ok:
            print('dx did not update, break ...')
            break
        count = count + 1
        if count > 10000:
            print('count reaches max, break ...')
            break
    
    final_error = costFunc(x_vec_opted[0,0], x_vec_opted[1,0], x_vec_opted[2,0], v1, v2, v3, v4)
    print('init_error = ', np.linalg.norm(init_error), ' , final_error = ', np.linalg.norm(final_error))
    plt.plot(count_list, error_list)
    plt.show()

if __name__ == '__main__':
    testGlGradient()