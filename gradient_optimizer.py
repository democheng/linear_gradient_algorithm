import numpy as np
import math
import random
import matplotlib.pyplot as plt

class CostFunction(object):
    def __init__(self, input, output, objective_function, jacobian_function):
       self.m_input = input
       self.m_output = output
       self.m_objective_function = objective_function
       self.m_jacbian_function = jacobian_function
    
    def get_input_len(self):
        return len(self.m_input)
    
    def get_output_len(self):
        return len(self.m_output)

    def get_residual(self, params):
        return self.m_objective_function(params, self.m_input, self.m_output)
    
    def get_jacobian(self, params):
        return self.m_jacbian_function(params, self.m_input)

class LMGradientOptimizer(object):
    def __init__(self, params):
        self.m_params = params
        self.m_params_opted = params
        self.m_residual_block_list = list()
    
    def add_residual_block(self, cost_function):
        self.m_residual_block_list.append(cost_function)
    
    def n1np_to_list(self, a):
        return list(a.reshape(1, a.shape[0])[0])

    def solve_problem(self):
        count_list = list()
        error_list = list()
        if len(self.m_residual_block_list) < 1:
            print('no residual block...')
            return count_list, error_list
        count = 0
        print('count = ', count, ' , init m_params = ', self.m_params)
        found = False
        
        e_mean = np.zeros((self.m_residual_block_list[0].get_output_len(),1))
        H_mean = np.zeros((len(self.m_params),len(self.m_params)))
        g_mean = np.zeros((len(self.m_params),1))
        for cur_cost_func in self.m_residual_block_list:
            cur_e = cur_cost_func.get_residual(self.m_params)
            cur_J = cur_cost_func.get_jacobian(self.m_params)
            cur_H = cur_J.transpose() @ cur_J
            cur_g = -cur_J.transpose() @ cur_e
            e_mean = e_mean + cur_e
            H_mean = H_mean + cur_H
            g_mean = g_mean + cur_g
        number_inv = (1.0 / len(self.m_residual_block_list))
        e_mean = e_mean * number_inv
        H_mean = H_mean * number_inv
        g_mean = g_mean * number_inv

        init_e_mean_norm = np.linalg.norm(e_mean)
        print('count = ', count, ' , init error = ', init_e_mean_norm)
        error_list.append(init_e_mean_norm)
        count_list.append(count)

        g_mean_norm = np.linalg.norm(g_mean)
        if g_mean_norm < 1e-8:
            print('g is too small, break ...')
            found = True
            return count_list, error_list
        
        tau = 1e-3
        h_diag_max = np.max(np.diag(H_mean))
        print('np.max(np.diag(H)) = ', h_diag_max)
        mu = tau * h_diag_max
        print('init mu = ', mu)
        v = 2.0
        cur_params = np.array(self.m_params).reshape(-1,1)
        while (not found) and (count < 1000):
            count = count + 1
            dx = np.linalg.inv(H_mean + mu * np.identity(len(self.m_params))) @ g_mean
            dx_norm = np.linalg.norm(dx)
            if dx_norm < 1e-8:
                print('dx is too small, break ...')
                found = True
                break
            new_params = cur_params + dx

            e_old_mean = np.zeros((self.m_residual_block_list[0].get_output_len(),1))
            e_new_mean = np.zeros((self.m_residual_block_list[0].get_output_len(),1))
            for cur_cost_func in self.m_residual_block_list:
                e_old_mean = e_old_mean + cur_cost_func.get_residual(self.n1np_to_list(cur_params))
                e_new_mean = e_new_mean + cur_cost_func.get_residual(self.n1np_to_list(new_params))
            e_old_mean = e_old_mean * number_inv
            e_new_mean = e_new_mean * number_inv
            e_old_mean_norm = np.linalg.norm(e_old_mean)
            e_new_mean_norm = np.linalg.norm(e_new_mean)
            p_a = e_old_mean_norm - e_new_mean_norm
            p_b = (0.5 * dx.transpose()@(mu * dx + g_mean))[0][0]
            p = p_a / (p_b + 1e-8)
            print('p_a = ', p_a)
            print('p_b = ', p_b)
            print('p = ', p)

            if p > 0.0:
                cur_params = new_params
                error_list.append(e_new_mean_norm)
                count_list.append(count)
                print('count = ', count, ' , mu = ', mu, ', v = ', v)
                print('count = ', count, ' , e_new_mean_norm = ', e_new_mean_norm)
                print('count = ', count, ' , cur_params = ', cur_params)

                e_mean = np.zeros((self.m_residual_block_list[0].get_output_len(),1))
                H_mean = np.zeros((len(self.m_params),len(self.m_params)))
                g_mean = np.zeros((len(self.m_params),1))
                for cur_cost_func in self.m_residual_block_list:
                    cur_e = cur_cost_func.get_residual(self.n1np_to_list(cur_params))
                    cur_J = cur_cost_func.get_jacobian(self.n1np_to_list(cur_params))
                    cur_H = cur_J.transpose() @ cur_J
                    cur_g = -cur_J.transpose() @ cur_e
                    e_mean = e_mean + cur_e
                    H_mean = H_mean + cur_H
                    g_mean = g_mean + cur_g
                e_mean = e_mean * number_inv
                H_mean = H_mean * number_inv
                g_mean = g_mean * number_inv
                g_norm = np.linalg.norm(g_mean)
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

        e_mean = np.zeros((self.m_residual_block_list[0].get_output_len(),1))
        for cur_cost_func in self.m_residual_block_list:
            cur_e = cur_cost_func.get_residual(self.n1np_to_list(cur_params))
            e_mean = e_mean + cur_e
        e_mean = e_mean * number_inv
        final_e_mean_norm = np.linalg.norm(e_mean)
        print('init_error = ', init_e_mean_norm, ' , final_error = ', final_e_mean_norm)
        self.m_params_opted = self.n1np_to_list(cur_params)
        return count_list, error_list

def test_function(params, input):
    # y0 = a3 * x2^3 + a2 * x1^2 + a1 * x0 + a0
    # y1 = a3 * x2 + a2 * x1 + a1 * x0 + a0^2
    assert len(params) == 4
    assert len(input) == 3
    y0 = params[3] * math.pow(input[2],3) + params[2] * math.pow(input[1],2) + params[1] * input[0] + params[0]
    y1 = params[3] * input[2] + params[2] * input[1] + params[1] * input[0] + math.pow(params[0],2)
    return [y0,y1]

def test_objective_function(params, input, output):
    # y0 = a3 * x2^3 + a2 * x1^2 + a1 * x0 + a0
    # y1 = a3 * x2 + a2 * x1 + a1 * x0 + a0^2
    assert len(params) == 4
    assert len(input) == 3
    assert len(output) == 2
    y0 = params[3] * math.pow(input[2],3) + params[2] * math.pow(input[1],2) + params[1] * input[0] + params[0]
    y1 = params[3] * input[2] + params[2] * input[1] + params[1] * input[0] + math.pow(params[0],2)
    residual = [y0 - output[0], y1 - output[1]]
    return np.array(residual).reshape(-1,1)

def test_jacobian(params, input):
    #     a0    a1    a2    a3         
    # y0  1     x0    x1^2  x2^3
    # y1  2*a0  x0    x1    x2
    assert len(params) == 4
    assert len(input) == 3
    return np.array([
                    [1.0        , input[0], math.pow(input[1],2), math.pow(input[2],3)], \
                    [2*params[0], input[0], input[1]            , input[2]]
                    ])


def test_gradient_optimize():
    params_gt = [2.0, -2.0, 2.0, 2.0]
    rand_num = 100
    input_list = list()
    output_list = list()
    for idx in range(rand_num):
        x0 = random.uniform(1, 10)
        x1 = random.uniform(1, 10)
        x2 = random.uniform(1, 10)
        input_list.append([x0,x1,x2])
        output_list.append(test_function(params_gt, [x0,x1,x2]))

    params_init = [1.0, 1.0, 10.0, 100.0]
    lm_opter = LMGradientOptimizer(params_init)
    for idx in range(rand_num):
        cost_func = CostFunction(input_list[idx], output_list[idx], test_objective_function, test_jacobian)
        lm_opter.add_residual_block(cost_func)
    lm_opter.solve_problem()
    return

if __name__ == '__main__':
    test_gradient_optimize()