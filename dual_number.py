import numpy as np
import math
import random

class DualNumber(object):
    def __init__(self, a, v=0.0, name='x') -> None:
        self.m_a = a
        self.m_v = v
        self.m_name = name
    
    def __str__(self):
        if self.m_v > 0:
            return '{}={}+{}ε'.format(self.m_name, self.m_a, self.m_v)
        else:
            return '{}={}-{}ε'.format(self.m_name, self.m_a, math.fabs(self.m_v))
    
    def __add__(self, dual):
        if isinstance(dual, DualNumber):
            return DualNumber(self.m_a + dual.m_a, self.m_v + dual.m_v)
        elif isinstance(dual, int) or isinstance(dual, float):
            return DualNumber(self.m_a + dual, self.m_v)
        print('input type is error:', dual)
        return None
    
    def __radd__(self, dual):
        if isinstance(dual, DualNumber):
            return DualNumber(self.m_a + dual.m_a, self.m_v + dual.m_v)
        elif isinstance(dual, int) or isinstance(dual, float):
            return DualNumber(self.m_a + dual, self.m_v)
        print('input type is error:', dual)
        return None
    
    def __sub__(self, dual):
        if isinstance(dual, DualNumber):
            return DualNumber(self.m_a - dual.m_a, self.m_v - dual.m_v)
        elif isinstance(dual, int) or isinstance(dual, float):
            return DualNumber(self.m_a + dual, self.m_v)
        print('input type is error:', dual)
        return None
    
    def __rsub__(self, dual):
        if isinstance(dual, DualNumber):
            return DualNumber(dual.m_a - self.m_a, dual.m_v - self.m_v)
        elif isinstance(dual, int) or isinstance(dual, float):
            return DualNumber(dual - self.m_a, -self.m_v)
        print('input type is error:', dual)
        return None

    def __mul__(self, dual):
        if isinstance(dual, DualNumber):
            return DualNumber(self.m_a * dual.m_a, self.m_a * dual.m_v + self.m_v * dual.m_a)
        elif isinstance(dual, int) or isinstance(dual, float):
            return DualNumber(self.m_a * dual, self.m_v * dual)
        print('input type is error:', dual)
        return None
    
    def __rmul__(self, dual):
        if isinstance(dual, DualNumber):
            return DualNumber(self.m_a * dual.m_a, self.m_a * dual.m_v + self.m_v * dual.m_a)
        elif isinstance(dual, int) or isinstance(dual, float):
            return DualNumber(self.m_a * dual, self.m_v * dual)
        print('input type is error:', dual)
        return None
    
    def __truediv__(self, dual):
        if isinstance(dual, DualNumber) and dual.m_a != 0:
            return DualNumber(self.m_a / dual.m_a, (self.m_v * dual.m_a - self.m_a * dual.m_v) / dual.m_a ** 2)
        elif isinstance(dual, int) or isinstance(dual, float) and dual != 0:
            return DualNumber(self.m_a / dual, self.m_v / dual)
        print('input type is error:', dual)
        return None
    
    def __rtruediv__(self, dual):
        if isinstance(dual, DualNumber) and self.m_a != 0:
            return DualNumber(dual.m_a / self.m_a, (dual.m_v * self.m_a - dual.m_a * self.m_v) / self.m_a ** 2)
        elif isinstance(dual, int) or isinstance(dual, float) and self.m_a != 0:
            return DualNumber(dual / self.m_a, (-dual * self.m_v) / self.m_a ** 2)
        print('input type is error:', dual)
        return None
    
    def __neg__(self, dual):
        return DualNumber(-self.m_a, -self.m_v)
    
    def __pow__(self, n):
        if isinstance(n, int):
            return DualNumber(self.m_a ** n, n * self.m_a ** (n-1) * self.m_v)
        print('input type is error:', n)
        return None
    
    @staticmethod
    def sin(x):
        if isinstance(x, DualNumber):
            return DualNumber(math.sin(x.m_a), math.cos(x.m_a) * x.m_v)
        elif isinstance(x, int) or isinstance(x, float):
            return DualNumber(math.sin(x))
        print('input type is error:', x)
        return None
    
    @staticmethod
    def cos(x):
        if isinstance(x, DualNumber):
            return DualNumber(math.cos(x.m_a), -math.sin(x.m_a) * x.m_v)
        elif isinstance(x, int) or isinstance(x, float):
            return DualNumber(math.cos(x))
        print('input type is error:', x)
        return None
    
    @staticmethod
    def exp(x):
        if isinstance(x, DualNumber):
            return DualNumber(math.exp(x.m_a), math.exp(x.m_a) * x.m_v)
        elif isinstance(x, int) or isinstance(x, float):
            return DualNumber(math.exp(x))
        print('input type is error:', x)
        return None

    @staticmethod
    def log(x):
        if isinstance(x, DualNumber) and x.m_a != 0:
            return DualNumber(math.log(x.m_a), x.m_v / x.m_a)
        elif isinstance(x, int) or isinstance(x, float):
            return DualNumber(math.log(x))
        print('input type is error:', x)
        return None

    @staticmethod
    def sqrt(x):
        if isinstance(x, DualNumber):
            return DualNumber(math.sqrt(x.m_a), x.m_v / (2*math.sqrt(x.m_a)))
        elif isinstance(x, int) or isinstance(x, float):
            return DualNumber(math.sqrt(x))
        print('input type is error:', x)
        return None
    
    @staticmethod
    def jacobian(func, params, input):
        J_list = list()
        param_num = len(params)
        for i in range(param_num):
            param_i = list()
            for j in range(param_num):
                if not isinstance(params[j], DualNumber):
                    if i == j:
                        param_i.append(DualNumber(params[j], 1))
                    else:
                        param_i.append(DualNumber(params[j], 0))
                else:
                    if i == j:
                        param_i.append(DualNumber(params[j].m_a, 1))
                    else:
                        param_i.append(DualNumber(params[j].m_a, 0))
            J_list.append(func(param_i, input).m_v)
        return J_list


def test_dual_number_add():
    x0 = DualNumber(10, 20, 'x0')
    x1 = DualNumber(1,2, 'x1')
    print(x0)
    print(x1)
    x2 = x0 + x1
    x2.m_name = 'x0+x1'
    x3 = x1 + x0
    x3.m_name = 'x1+x0'
    print(x2)
    print(x3)
    x4 = 100 + x0
    x4.m_name = '100+x0'
    print(x4)

def test_func0(params, input):
    assert len(params) == 3
    assert len(input) == 3
    return params[2] * input[2]**3 + params[1] * input[1]**2 + params[0] * input[0] - 10.0

def test_func1(params, input):
    assert len(params) == 3
    assert len(input) == 3
    return params[2] * input[2] + params[1] * input[1]**2 + params[0] * input[0]**3 - 20.0

def test_func0_jacobian(params, input):
    return [input[0], input[1]**2, input[2]**3]

def test_func1_jacobian(params, input):
    return [input[0]**3, input[1]**2, input[2]]

def test_analytical_jacobian(params, input):
    J0 = test_func0_jacobian(params, input)
    J1 = test_func1_jacobian(params, input)
    print('test_analytical_jacobian:')
    print(J0)
    print(J1)

def test_dual_jacobian(params, input):
    J0 = DualNumber.jacobian(test_func0, params, input)
    J1 = DualNumber.jacobian(test_func1, params, input)
    print('test_dual_jacobian:')
    print(J0)
    print(J1)

if __name__ == '__main__':
    # test_dual_number_add()
    params_gt = [2.0, -2.0, 4.0]
    rand_num = 10
    input_list = list()
    output_list = list()
    for idx in range(rand_num):
        x0 = random.uniform(1, 10)
        x1 = random.uniform(1, 10)
        x2 = random.uniform(1, 10)
        input = [x0, x1, x2]
        test_analytical_jacobian(params_gt,input)
        test_dual_jacobian(params_gt,input)
        print('--------')
    