# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import numpy
import random
import math
from hmmlearn.hmm import MultinomialHMM

N = 3
M = 2
T = 3
A = numpy.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]], dtype='float32')
B = numpy.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]], dtype='float32')
pi = numpy.array([0.2, 0.4, 0.4], dtype='float32')
# pi = numpy.array([0, 0, 1], dtype='float32')
# A = numpy.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype='float32')
# B = numpy.array([[1, 0], [0, 1], [1, 0]], dtype='float32')

def probability_forward(observes):
    # 初始化alpha
    alpha = pi * B[:,(observes[0]-1)]
    for t in range(1,T):
        alpha = numpy.transpose(numpy.dot(alpha, A)) * B[:,(observes[t]-1)]
    res = numpy.sum(alpha)
    return res

def probability_backward(observes):
    # 初始化beta
    beta = numpy.ones((N, ), dtype='float32')
    for t in range(T-1,0,-1):
        beta = numpy.dot(A, B[:,(observes[t]-1)] * beta)
    res = numpy.dot(pi, B[:,(observes[0]-1)] * beta)
    return res

def viterbi_inference(observes):
    # 初始化delta
    states, deltas = [], []
    delta = pi * B[:,(observes[0]-1)]
    print(delta)
    deltas.append(delta)
    for t in range(1,T):
        print(numpy.reshape(delta, (N,1)) * A)
        delta = numpy.max(numpy.reshape(delta, (N,1)) * A, axis=0) * B[:,(observes[t]-1)]
        print(delta)
        deltas.append(delta)
    res = numpy.max(delta)
    state = numpy.argmax(delta)
    states.append(state+1)
    for t in range(T-2,-1,-1):
        temp = deltas[t] * A[:,state]
        state = numpy.argmax(temp)
        states.append(state+1)
    states = states[::-1]
    return (res, states)

def get_observes(n, T):
    # 根据现有参数得到多组观测
    states, observes = [], []
    for i in range(n):
        state, observe = [], []
        y = numpy.random.choice(N, p=pi)
        state.append(y)
        x = numpy.random.choice(M, p=B[y,:])
        observe.append(x)
        for j in range(1, T):
            y = numpy.random.choice(N, p=A[y,:])
            state.append(y)
            x = numpy.random.choice(M, p=B[y,:])
            observe.append(x)
        states.append(state)
        observes.append(observe)
    states = numpy.array(states)
    observes = numpy.array(observes)
    return states, observes

def train_by_statistic(states, observes):
    # 根据观测值和隐变量，通过统计的方法训练参数
    first_state_vector = numpy.zeros((N,), dtype='int32')
    trans_matrix = numpy.zeros((N, N), dtype='int32')
    obs_matrix = numpy.zeros((N, M), dtype='int32')
    for i in range(states.shape[0]):
        first_state_vector[states[i,0]] += 1
        obs_matrix[states[i,0], observes[i,0]] += 1
        for j in range(1, states.shape[1]):
            trans_matrix[states[i,j-1], states[i,j]] += 1
            obs_matrix[states[i,j], observes[i,j]] += 1
    param_pi = numpy.array(1.0 * first_state_vector / numpy.sum(first_state_vector), dtype='float32')
    param_A = numpy.array(1.0 * trans_matrix / numpy.stack([numpy.sum(trans_matrix, axis=1)] * N, axis=1), dtype='float32')
    param_B = numpy.array(1.0 * obs_matrix / numpy.stack([numpy.sum(obs_matrix, axis=1)] * M, axis=1), dtype='float32')
    return param_pi, param_A, param_B

def train_by_model_iter(observes, n_steps=1000):
    # 只根据观测值，通过模型的方法训练参数
    # param_pi = numpy.array([0.2, 0.4, 0.4], dtype='float32')
    # param_A = numpy.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]], dtype='float32')
    # param_B = numpy.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]], dtype='float32')
    param_pi = numpy.array([0.8, 0.1, 0.1], dtype='float32')
    param_A = numpy.array([[0.3, 0.2, 0.5], [0.4, 0.4, 0.2], [0.3, 0.6, 0.1]], dtype='float32')
    param_B = numpy.array([[0.2, 0.8], [0.7, 0.3], [0.9, 0.1]], dtype='float32')
    # param_pi = numpy.array([0.33, 0.33, 0.33], dtype='float32')
    # param_A = numpy.array([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]], dtype='float32')
    # param_B = numpy.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype='float32')
    E = observes.shape[0]
    T = observes.shape[1]
    for n in range(n_steps):
        # 计算alpha，迭代方法 
        alpha = numpy.zeros((E, T, N), dtype='float32')
        for i in range(E):
            for a in range(N):
                alpha[i,0,a] = param_pi[a] * param_B[a, observes[i,0]]
            for t in range(1, T):
                for a in range(N):
                    temp = 0
                    for b in range(N):
                        temp += alpha[i,t-1,b] * param_A[b,a]
                    alpha[i,t,a] = temp * param_B[a,observes[i,t]]

        # 计算beta，迭代方法
        beta = numpy.zeros((E, T, N), dtype='float32')
        for i in range(E):
            beta[i,T-1,:] = numpy.ones((N, ), dtype='float32')
            for t in range(T-2,-1,-1):
                for a in range(N):
                    temp = 0
                    for b in range(N):
                        temp += param_A[a,b] * param_B[b,observes[i,t+1]] * beta[i,t+1,b]
                    beta[i,t,a] = temp
        
        # 计算损失函数
        L = 0
        for i in range(E):
            L1 = numpy.sum(alpha[i,T-1,:])
            L2 = numpy.sum(param_pi * param_B[:,observes[i,0]] * beta[i,0,:])
            if abs(L1 - L2) >= 1e-3:
                print(L1, L2)
            L += numpy.log((L1 + L2) / 2.0 + 1e-12)
        L = L / E
        print('[%d] loss: %.8f' % (n+1, L))
        
        # 计算new_param_pi，迭代方法
        new_param_pi = numpy.zeros((E, N), dtype='float32')
        for i in range(E):
            for a in range(N):
                nume = alpha[i,0,a] * beta[i,0,a]
                deno = 0
                for aa in range(N):
                    deno += alpha[i,0,aa] * beta[i,0,aa]
                new_param_pi[i,a] = 1.0 * nume / (deno + 1e-12)
        new_param_pi = numpy.mean(new_param_pi, axis=0)

        # 计算new_param_A，迭代方法
        new_param_A = numpy.zeros((E, N, N), dtype='float32')
        for i in range(E):
            for a in range(N):
                for b in range(N):
                    nume, deno = 0, 0
                    for t in range(T-1):
                        nume += alpha[i,t,a] * param_A[a,b] * param_B[b,observes[i,t+1]] * beta[i,t+1,b]
                        deno += alpha[i,t,a] * beta[i,t,a]
                    new_param_A[i,a,b] = 1.0 * nume / (deno + 1e-12)
        new_param_A = numpy.mean(new_param_A, axis=0)
            
        # 计算new_param_B，迭代方法
        new_param_B = numpy.zeros((E, N, M), dtype='float32')
        for i in range(E):
            for a in range(N):
                for b in range(M):
                    nume, deno = 0, 0
                    for t in range(T):
                        nume += alpha[i,t,a] * beta[i,t,a] * 1.0 if observes[i,t] == b else 0.0
                        deno += alpha[i,t,a] * beta[i,t,a]
                    new_param_B[i,a,b] = 1.0 * nume / (deno + 1e-12)
        new_param_B = numpy.mean(new_param_B, axis=0)
            
        param_pi = new_param_pi
        param_A = new_param_A
        param_B = new_param_B
    
    return param_pi, param_A, param_B

def train_by_model_matrix(observes, n_steps=1000):
    # 只根据观测值，通过模型的方法训练参数
    param_pi = numpy.array([0.8, 0.1, 0.1], dtype='float32')
    param_A = numpy.array([[0.3, 0.2, 0.5], [0.4, 0.4, 0.2], [0.3, 0.6, 0.1]], dtype='float32')
    param_B = numpy.array([[0.2, 0.8], [0.7, 0.3], [0.9, 0.1]], dtype='float32')
    """
    param_pi = numpy.array([0, 0, 1.0], dtype='float32')
    param_A = numpy.array([[0.36, 0, 0.64], [0.64, 0.36, 0], [0, 0.8, 0.2]], dtype='float32')
    param_B = numpy.array([[0.55, 0.42], [0.44, 0.56], [0.57, 0.43]], dtype='float32')
    """
    
    E = observes.shape[0]
    T = observes.shape[1]
    for n in range(n_steps):
        # 计算alpha，矩阵方法 
        alpha = numpy.zeros((E, T, N), dtype='float32')
        for i in range(E):
            alpha[i,0,:] = param_pi * param_B[:,observes[i,0]]
            for t in range(1, T):
                alpha[i,t,:] = numpy.transpose(numpy.dot(alpha[i,t-1,:], param_A)) * param_B[:,observes[i,t]]
        
        # 计算beta，矩阵方法 
        beta = numpy.zeros((E, T, N), dtype='float32')
        for i in range(E):
            beta[i,T-1,:] = numpy.ones((N, ), dtype='float32')
            for t in range(T-2,-1,-1):
                beta[i,t,:] = numpy.dot(param_A, param_B[:,observes[i,t+1]] * beta[i,t+1,:])
        
        L = calculate_loss(param_pi, param_A, param_B, observes)
        print('[%d] loss: %.8f' % (n+1, L))
        
        # 计算new_param_pi，矩阵方法
        new_param_pi = numpy.zeros((E, N), dtype='float32')
        for i in range(E):
            nume = alpha[i,0,:] * beta[i,0,:]
            deno = numpy.sum(alpha[i,0,:] * beta[i,0,:])
            new_param_pi[i,:] = 1.0 * nume / (deno + 1e-12)
        new_param_pi = numpy.mean(new_param_pi, axis=0)
        
        # 计算new_param_A，矩阵方法
        new_param_A = numpy.zeros((E, N, N), dtype='float32')
        for i in range(E):
            nume = param_A * numpy.dot(
                numpy.transpose(alpha[i,:,:])[:,:-1], beta[i,1:,:] * numpy.transpose(param_B[:,observes[i,:]])[1:,:])
            deno = numpy.stack([numpy.sum(alpha[i,:-1,:] * beta[i,:-1,:], axis=0)] * N, axis=1)
            new_param_A[i,:,:] += nume / (deno + 1e-12)
        new_param_A = numpy.mean(new_param_A, axis=0)
        
        # 计算new_param_B，矩阵方法
        new_param_B = numpy.zeros((E, N, M), dtype='float32')
        for i in range(E):
            nume = numpy.dot(numpy.transpose(alpha[i] * beta[i]), numpy.transpose(numpy.eye(M)[:,observes[i,:]]))
            deno = numpy.stack([numpy.sum(alpha[i] * beta[i], axis=0)] * M, axis=1)
            new_param_B[i,:,:] += nume / (deno + 1e-12)
        new_param_B = numpy.mean(new_param_B, axis=0)
            
        param_pi = new_param_pi
        param_A = new_param_A
        param_B = new_param_B
    
    return param_pi, param_A, param_B

def train_by_model_hmmlearn(observes, n_steps=1000):
    # 只根据观测值，通过hmmlearn的方法训练参数
    param_pi = numpy.array([0.8, 0.1, 0.1], dtype='float64')
    param_A = numpy.array([[0.3, 0.2, 0.5], [0.4, 0.4, 0.2], [0.3, 0.6, 0.1]], dtype='float64')
    param_B = numpy.array([[0.2, 0.8], [0.7, 0.3], [0.9, 0.1]], dtype='float64')
    E = observes.shape[0]
    T = observes.shape[1]
    
    model = MultinomialHMM(n_components=N, n_iter=0, init_params="")
    model.startprob_ = pi
    model.transmat_ = A
    model.emissionprob_ = B
    print(model.predict(numpy.array([0,1,0], dtype='int64')))
    # model.fit(numpy.reshape(observes, (E*T,1)), lengths=[T]*E)
    
    param_pi = model.startprob_
    param_A = model.transmat_
    param_B = model.emissionprob_
    L = calculate_loss(param_pi, param_A, param_B, observes)
    print('[%d] loss: %.8f' % (1, L))

    return model.startprob_, model.transmat_, model.emissionprob_

def calculate_loss(param_pi, param_A, param_B, observes):
    E = observes.shape[0]
    T = observes.shape[1]
    
    # 计算alpha，矩阵方法 
    alpha = numpy.zeros((E, T, N), dtype='float32')
    for i in range(E):
        alpha[i,0,:] = param_pi * param_B[:,observes[i,0]]
        for t in range(1, T):
            alpha[i,t,:] = numpy.transpose(numpy.dot(alpha[i,t-1,:], param_A)) * param_B[:,observes[i,t]]
    
    # 计算损失函数
    L = 0
    for i in range(E):
        L += numpy.log(numpy.sum(alpha[i,T-1,:]) + 1e-12)
    L = 1.0 * L / E
    
    return L


if __name__ == '__main__':
    x = 2
    if x == 1:
        observes = numpy.array([1, 2, 1], dtype='int32')
        res = viterbi_inference(observes)
        print(res)
    elif x == 2:
        random.seed(1)
        numpy.random.seed(1)
        states, observes = get_observes(1, 20)
        # param_pi, param_A, param_B = train_by_model_matrix(observes, n_steps=100)
        param_pi1, param_A1, param_B1 = train_by_model_hmmlearn(observes, n_steps=100)
        exit()
        print(observes)
        print(param_pi)
        print(param_pi1)
        print(pi)
        print()
        print(param_A)
        print(param_A1)
        print(A)
        print()
        print(param_B)
        print(param_B1)
        print(B)
