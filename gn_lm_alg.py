import numpy as np
import math
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

def testGnGradient(x1, x2, x3, v1, v2, v3, v4):
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
        
        J0 = JfMatrix(x_vec_opted[0,0], x_vec_opted[1,0], x_vec_opted[2,0])
        H0 = J0.transpose() @ J0
        
        dx = -np.linalg.pinv(H0) @ J0.transpose() @ cur_error
        dx_norm = np.linalg.norm(dx)
        x_vec_opted = x_vec_opted + dx
        print('dx_norm = ', dx_norm)
        # print('H0 = ', H0)
        # print('dx = ', dx)
        if dx_norm < 1e-8:
            print('dx did not update, break ...')
            break
        
        if count % 1 == 0:
            print('count = ', count, ' , cur_error_norm = ', cur_error_norm)
            print('count = ', count, ' , x_vec_opted = ', x_vec_opted[0,0], x_vec_opted[1,0], x_vec_opted[2,0])

        count = count + 1
        if count > 1000:
            print('count reaches max, break ...')
            break
    
    final_error = costFunc(x_vec_opted[0,0], x_vec_opted[1,0], x_vec_opted[2,0], v1, v2, v3, v4)
    print('init_error = ', np.linalg.norm(init_error), ' , final_error = ', np.linalg.norm(final_error))
    plt.plot(count_list, error_list)
    plt.show()

def testLmGradient(x1, x2, x3, v1, v2, v3, v4):
    count = 0
    x_vec_opted = np.array([x1, x2, x3])
    x_vec_opted = x_vec_opted.reshape(-1,1)
    print('count = ', count, ' , x_vec_init = ', x_vec_opted)
    
    count_list = list()
    error_list = list()
    found = False
    
    cur_error = costFunc(x_vec_opted[0,0], x_vec_opted[1,0], x_vec_opted[2,0], v1, v2, v3, v4)
    init_error_norm = np.linalg.norm(cur_error)
    error_list.append(init_error_norm)
    count_list.append(count)

    J0 = JfMatrix(x_vec_opted[0,0], x_vec_opted[1,0], x_vec_opted[2,0])
    A = J0.transpose() @ J0
    g = -J0.transpose() @ cur_error
    g_norm = np.linalg.norm(g)
    if g_norm < 1e-8:
        print('g is too small, break ...')
        found = True
    tau = 1e-3
    print('np.max(np.diag(A)) = ', np.max(np.diag(A)))
    mu = tau * np.max(np.diag(A))
    print('init mu = ', mu)
    v = 2.0
    while (not found) and (count < 1000):
        count = count + 1
        dx = np.linalg.inv(A + mu * np.identity(3)) @ g
        dx_norm = np.linalg.norm(dx)
        if dx_norm < 1e-8:
            print('dx is too small, break ...')
            found = True
            break
        x_vec_tmp = x_vec_opted + dx
        error_old = costFunc(x_vec_opted[0,0], x_vec_opted[1,0], x_vec_opted[2,0], v1, v2, v3, v4)
        error_new = costFunc(x_vec_tmp[0,0], x_vec_tmp[1,0], x_vec_tmp[2,0], v1, v2, v3, v4)
        p_a = np.linalg.norm(error_old) - np.linalg.norm(error_new)
        p_b = (0.5 * dx.transpose()@(mu * dx + g))[0][0]
        print('p_a = ', p_a)
        print('p_b = ', p_b)
        p = p_a / (p_b + 1e-8)
        print('p = ', p)
        if p > 0.0:
            x_vec_opted = x_vec_tmp
            cur_error = costFunc(x_vec_opted[0,0], x_vec_opted[1,0], x_vec_opted[2,0], v1, v2, v3, v4)
            cur_error_norm = np.linalg.norm(cur_error)
            error_list.append(cur_error_norm)
            count_list.append(count)
            if count % 1 == 0:
                print('count = ', count, ' , mu = ', mu, ', v = ', v)
                print('count = ', count, ' , cur_error_norm = ', cur_error_norm)
                print('count = ', count, ' , x_vec_opted = ', x_vec_opted[0,0], x_vec_opted[1,0], x_vec_opted[2,0])

            J0 = JfMatrix(x_vec_opted[0,0], x_vec_opted[1,0], x_vec_opted[2,0])
            A = J0.transpose() @ J0
            g = -J0.transpose() @ cur_error
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-8:
                print('g is too small, break ...')
                found = True
                break
            print('1 - math.pow(2*p-1, 3) = ', 1 - math.pow(2*p-1, 3))
            mu = mu * max(0.333333333, 1 - math.pow(2*p-1, 3))
            v = 2
        else:
            mu = mu * v
            v = v * 2

    final_error = costFunc(x_vec_opted[0,0], x_vec_opted[1,0], x_vec_opted[2,0], v1, v2, v3, v4)
    final_error_norm = np.linalg.norm(final_error)
    print('init_error = ', init_error_norm, ' , final_error = ', final_error_norm)
    plt.plot(count_list, error_list)
    plt.show()

if __name__ == '__main__':
    x1 = 30.0
    x2 = 3.0 
    x3 = 3.0
    v1 = 10.0
    v2 = 50.0
    v3 = 100.0 
    v4 = 300.0
    testGnGradient(x1, x2, x3, v1, v2, v3, v4)
    testLmGradient(x1, x2, x3, v1, v2, v3, v4)