import numpy as np
import math

def TestFunc(x):
    return math.e ** (-(x ** 2))

def IntegrateByStripe(func, stripe, is_right = True):
    input_x = None
    if is_right:
        input_x = stripe[1:]
    else:
        input_x = stripe[0:-1]
    return np.sum(func(input_x)) * (stripe[1] - stripe[0])

def IntegrateByLadderShape(func, stripe):
    h = stripe[1] - stripe[0]
    return np.sum(func(stripe)) * h - (func(stripe[0]) + func(stripe[-1])) * h * 0.5

def IntegrateBySimpsonRule(func, stripe):
    h_inv_3 = (stripe[1] - stripe[0]) / 3
    return np.sum(func(stripe[1::2])) * 4 * h_inv_3 + np.sum(func(stripe[2:-1:2])) * 2 * h_inv_3 + (func(stripe[0]) + func(stripe[-1])) * h_inv_3

if __name__ == '__main__':
    x_start = 0.0
    x_end = 2.0
    total_count = 21
    input_x = np.linspace(x_start, x_end, num = total_count)
    # print(input_x)
    res_stripe_right = IntegrateByStripe(TestFunc, input_x, True)
    res_stripe_left = IntegrateByStripe(TestFunc, input_x, False)
    if res_stripe_right < res_stripe_left:
        print(res_stripe_right, res_stripe_left)
    else:
        print(res_stripe_left, res_stripe_right)

    res_ladder = IntegrateByLadderShape(TestFunc, input_x)
    print(res_ladder)

    res_simpson = IntegrateBySimpsonRule(TestFunc, input_x)
    print(res_simpson)