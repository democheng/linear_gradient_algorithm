import numpy as np
import random
from dual_number import DualNumber
from gradient_optimizer import CostFunction, LMGradientOptimizer

def objective_func0(params, input, output=[0,0]):
    assert len(params) == 3
    assert len(input) == 3
    assert len(output) == 2
    return params[2] * input[2]**3 + params[1] * input[1]**2 + params[0] * input[0] - output[0]

def objective_func1(params, input, output=[0,0]):
    assert len(params) == 3
    assert len(input) == 3
    assert len(output) == 2
    return params[2] * input[2] + params[1] * input[1]**2 + params[0] * input[0]**3 - output[1]

def objective_func(params, input, output, return_np=True):
    residual = [objective_func0(params, input, output), objective_func1(params, input, output)]
    if return_np:
        return np.array(residual).reshape(-1,1)
    return residual

def auto_jacobian_of_objective_func(params, input):
    J0 = DualNumber.jacobian(objective_func0, params, input)
    J1 = DualNumber.jacobian(objective_func1, params, input)
    J_mat = np.array([J0, J1])
    return J_mat

def test_auto_diff_optimize():
    params_gt = [2.0, -2.0, 2.0]
    rand_num = 100
    input_list = list()
    output_list = list()
    for idx in range(rand_num):
        x0 = random.uniform(-20, 20)
        x1 = random.uniform(-20, 20)
        x2 = random.uniform(-20, 20)
        input_list.append([x0,x1,x2])
        output = objective_func(params_gt, [x0,x1,x2], [0, 0], False)
        output[0] = output[0] + random.uniform(-1.5, 1.5)
        output[1] = output[1] + random.uniform(-1.5, 1.5)
        output_list.append(output)

    params_init = [1.0, 1.0, 10.0]
    lm_opter = LMGradientOptimizer(params_init)
    for idx in range(rand_num):
        cost_func = CostFunction(input_list[idx], output_list[idx], objective_func, auto_jacobian_of_objective_func)
        lm_opter.add_residual_block(cost_func)
    lm_opter.solve_problem()
    return

if __name__ == '__main__':
    test_auto_diff_optimize()