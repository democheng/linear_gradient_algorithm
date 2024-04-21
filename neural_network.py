import numpy as np
import random

class NeuralNetwork(object):
    def __init__(self, node_list, \
                activate_func, jacobian_of_activate_func, \
                cost_func, jacobian_of_cost_func) -> None:
        self.m_node_list = node_list
        # self.m_weight_list = [np.random.randn(num_from,num_to) for num_from,num_to in zip(node_list[:-1], node_list[1:])]
        # self.m_bias_list = [np.random.randn(num_to,1) for num_to in node_list[1:]]
        self.m_weight_list = [np.random.uniform(-1,0,(num_from,num_to))+1 for num_from,num_to in zip(node_list[:-1], node_list[1:])]
        self.m_bias_list = [np.random.uniform(-1,0,(num_to,1))+1 for num_to in node_list[1:]]
        # print('self.m_weight_list:', self.m_weight_list)
        # print('self.m_bias_list:', self.m_bias_list)
        self.m_activate_func = activate_func
        self.m_jacobian_of_activate_func = jacobian_of_activate_func
        self.m_cost_func = cost_func
        self.m_jacobian_of_cost_func = jacobian_of_cost_func
    
    def resetParameters(self):
        # for idx in range(len(self.m_weight_list)):
        #     self.m_weight_list[idx] = np.random.randn(self.m_weight_list[idx].shape[0],self.m_weight_list[idx].shape[1])
        # for idx in range(len(self.m_bias_list)):
        #     self.m_bias_list[idx] = np.random.randn(self.m_bias_list[idx].shape[0],self.m_bias_list[idx].shape[1])
        for idx in range(len(self.m_weight_list)):
            self.m_weight_list[idx] = 1+np.random.uniform(-1,0,(self.m_weight_list[idx].shape[0],self.m_weight_list[idx].shape[1]))
        for idx in range(len(self.m_bias_list)):
            self.m_bias_list[idx] = 1+np.random.uniform(-1,0,(self.m_bias_list[idx].shape[0],self.m_bias_list[idx].shape[1]))
    def feedforward(self,a):
        for w,b in zip(self.m_weight_list, self.m_bias_list):
            a = self.m_activate_func(w.transpose() @ a + b)
        return a
    
    def backpropagation(self,a,output):
        nabla_w = [np.zeros(w.shape) for w in self.m_weight_list]
        nabla_b = [np.zeros(b.shape) for b in self.m_bias_list]
        # feedforward
        activation = a
        activation_list = [a]
        z_list = []
        for w, b in zip(self.m_weight_list, self.m_bias_list):
            z = w.transpose() @ activation + b
            z_list.append(z)
            activation = self.m_activate_func(z)
            activation_list.append(activation)
        # backpropagation: implement 4 equations
        # eq.1
        delta = self.m_jacobian_of_cost_func(activation_list[-1], output) * self.m_jacobian_of_activate_func(z_list[-1])
        # print('eq1 activation_list[-2].shape = ', activation_list[-2].shape)
        # print('eq1 delta.transpose().shape = ', delta.transpose().shape)
        # eq.3 and eq.4
        nabla_w[-1] = activation_list[-2] @ delta.transpose()
        nabla_b[-1] = delta
        # print('eq1 nabla_w[-1].shape = ', nabla_w[-1].shape)
        # print('eq1 nabla_b[-1].shape = ', nabla_b[-1].shape)
        # eq.2, Recursive
        for l in range(2, len(self.m_node_list)):
            a_cur = self.m_activate_func(z_list[-l])
            # print('self.m_weight_list[-l+1].shape = ', self.m_weight_list[-l+1].shape)
            # print('delta.shape = ', delta.shape)
            # print('a_cur.shape = ', a_cur.shape)
            delta = self.m_weight_list[-l+1] @ delta * a_cur
            # print('activation_list[-l-1].shape = ', activation_list[-l-1].shape)
            # print('delta.transpose().shape = ', delta.transpose().shape)
            nabla_w[-l] = activation_list[-l-1] @ delta.transpose()
            nabla_b[-l] = delta
        
        return (nabla_w, nabla_b)
    
    def evaluate(self, test_input):
        res = 0.0
        for data in test_input:
            x = data[0]
            delta = self.feedforward(x) - data[1]
            delta2 = delta ** 2
            res = res + delta2.sum()
        res = res / len(test_input)
        return res

    def updateParameters(self, mini_bach, learning_rate):
        nabla_w = [np.zeros(w.shape) for w in self.m_weight_list]
        nabla_b = [np.zeros(b.shape) for b in self.m_bias_list]
        for x, y in mini_bach:
            delta_nabla_w, delta_nabla_b = self.backpropagation(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.m_weight_list = [w - nw / len(mini_bach) * learning_rate for w, nw in zip(self.m_weight_list, nabla_w)]
        self.m_bias_list = [b - nb / len(mini_bach) * learning_rate  for b, nb in zip(self.m_bias_list, nabla_b)]
    
    def SGD(self, training_data, epochs, mini_bach_size, learning_rate, test_data=None):
        for idx in range(epochs):
            random.shuffle(training_data)
            mini_baches = [training_data[k:k+mini_bach_size] for k in range(0,len(training_data),mini_bach_size)]
            for mini_bach in mini_baches:
                self.updateParameters(mini_bach,learning_rate)
            if idx % 100 == 0:
                if test_data:
                    print("epoch {0}: {1}, size:{2}".format(idx, self.evaluate(test_data), len(test_data)))
                else:
                    print("epoch ", idx, " complete...")

def sigmoidFunc(z):
    return 1.0 / (1.0 + np.exp(-z))

def jacobianOfSigmoidFunc(z):
    return sigmoidFunc(z) * (1.0 - sigmoidFunc(z))

def ReLU(z):
    return (np.abs(z) + z) * 0.5

def jacobianOfReLU(z):
    return np.where(z > 0, 1, 0)

def jacobianOfQuadraticCostFunc(y_nn, y):
    return y_nn - y


if __name__ == '__main__':
    # nn = NeuralNetwork([10,20,40,80,40,20,10], sigmoidFunc, jacobianOfSigmoidFunc, None, jacobianOfQuadraticCostFunc)
    nn = NeuralNetwork([10,4,2], ReLU, jacobianOfReLU, None, jacobianOfQuadraticCostFunc)
    count = 1000
    all_data_list = []
    for idx in range(count):
        input = np.random.randn(10,1)
        output = nn.feedforward(input)
        all_data_list.append((input, output))
    training_list = all_data_list[0:75]
    test_list = all_data_list[75:100]
    # gt evalute
    gt_output = nn.evaluate(training_list)
    # init evalute
    nn.resetParameters()
    before_training_output = nn.evaluate(training_list)
    # training
    nn.SGD(training_list, 10000, int(count*0.1), 0.01, test_list)
    output = nn.evaluate(training_list)
    print('gt_output: ', gt_output)
    print('before_training_output: ', before_training_output, ', size:', len(training_list))
    print('after training output: ', output, ', size:', len(training_list))