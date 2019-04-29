# -*- coding: utf-8 -*-
# author: ronniecao
# date: 2019/04/29
# description: hidden markov model for python
from __future__ import print_function
import time
import numpy
import random
import math


class HMM:

    def __init__(self, n_states, n_features, 
        start_prob=None, trans_matrix=None, emission_matrix=None):
        # 初始化
        self.n_states = n_states
        self.n_features = n_features
        self.start_prob_ = start_prob if type(start_prob) != type(None) else self._init_start_prob()
        self.trans_matrix_ = trans_matrix if type(trans_matrix) != type(None) else self._init_trans_matrix()
        self.emission_matrix_ = emission_matrix if type(emission_matrix) != type(None) else self._init_emission_matrix()

    def _init_start_prob(self):
        # 用均匀分布初始化start_prob
        start_prob_ = numpy.zeros((self.n_states, ), dtype='float32') + 1.0 / self.n_states
        
        return start_prob_

    def _init_trans_matrix(self):
        # 用均匀分布初始化trans_matrix
        trans_matrix_ = numpy.zeros((self.n_states, self.n_states), dtype='float32') + 1.0 / self.n_states
        
        return trans_matrix_

    def _init_emission_matrix(self):
        # 用均匀分布初始化emission_matrix
        emission_matrix_ = numpy.zeros((self.n_states, self.n_features), dtype='float32') + 1.0 / self.n_features
        
        return emission_matrix_

    def _forward(self, observe):
        # 前向算法计算alpha
        n_sequence = len(observe)
        alpha = numpy.zeros((n_sequence, self.n_states), dtype='float32')
        alpha[0,:] = self.start_prob_ * self.emission_matrix_[:,0]
        for t in range(1, n_sequence):
            alpha[t,:] = numpy.transpose(numpy.dot(alpha[t-1,:], self.trans_matrix_)) * self.emission_matrix_[:,t]
        
        return alpha

    def _backward(self, observe):
        # 后向算法计算beta
        n_sequence = len(observe)
        beta = numpy.zeros((n_sequence, self.n_states), dtype='float32')
        beta[n_sequence-1,:] = numpy.ones((self.n_states, ), dtype='float32')
        for t in range(n_sequence-2, -1, -1):
            beta[t,:] = numpy.dot(self.trans_matrix_, self.emission_matrix_[:,t+1] * beta[t+1,:])
        
        return beta

    def viterbi_inference(self, observe):
        # 使用viterbi算法进行预测
        n_sequences = len(observe)
        states = numpy.zeros((n_sequences, ), dtype='int32')
        deltas = numpy.zeros((n_sequences, self.n_states), dtype='float32')
        deltas[0,:] = self.start_prob_ * self.emission_matrix_[:,0]
        for t in range(1, n_sequences):
            deltas[t,:] = numpy.max(
                numpy.reshape(deltas[t-1,:], (self.n_states, 1)) * self.trans_matrix_, axis=0) * self.emission_matrix_[:,t]
        res = numpy.max(delta[n_sequences-1,:])
        states[n_sequences-1] = numpy.argmax(delta[n_sequences-1,:])
        for t in range(n_sequences-2, -1, -1):
            states[t] = numpy.argmax(deltas[t] * self.trans_matrix_[:,states[t+1]])
        
        return deltas, states

    def calculate_probability(self, observes):
        # 计算损失函数
        n_samples = len(observes)
        probs = numpy.zeros((n_samples, ), dtype='float32')
        logprob = 0
        for i in range(E):
            alpha = self._forward(observes[i])
            probs[i] = numpy.sum(alpha[len(observes[i])-1,:])
            logprob += numpy.log(prob + 1e-12)
        logprob = 1.0 * logprob / n_samples
        
        return logprob, probs

    def samples(self, n_samples, lengths=None):
        # 根据现有参数得到多组观测
        states, observes = [], []
        max_length = max(lengths) if type(lengths) != type(None) else 10
        states = numpy.zeros((n_samples, max_length), dtype='int32')
        observes = numpy.zeros((n_samples, max_length), dtype='int32')
        for i in range(n_samples):
            states[i,0] = y = numpy.random.choice(self.n_states, p=self.start_prob_)
            observes[i,0] = x = numpy.random.choice(self.n_features, p=self.emission_matrix_[y,:])
            n_sequences = lengths[i] if type(lengths) != type(None) else random.randint(1,10)
            for t in range(1, n_sequences):
                states[i,t] = y = numpy.random.choice(self.n_states, p=self.trans_matrix_[y,:])
                observes[i,t] = x = numpy.random.choice(self.n_features, p=self.emission_matrix_[y,:])
        
        return states, observes

    def statistic(self, states, lengths, observes=None):
        # 根据观测值和隐变量，通过统计的方法训练参数
        start_prob_ = numpy.zeros((self.n_states, ), dtype='int32')
        trans_matrix_ = numpy.zeros((self.n_states, self.n_states), dtype='int32')
        emission_matrix_ = numpy.zeros((self.n_states, self.n_features), dtype='int32')
        n_samples = observes.shape[0]
        for i in range(n_samples):
            n_sequences = lengths[i]
            start_prob_[states[i,0]] += 1
            for t in range(1, lengths[i]):
                trans_matrix_[states[i,t-1], states[i,t]] += 1
        start_prob_ = numpy.array(1.0 * start_prob_ / numpy.sum(start_prob_), dtype='float32')
        trans_matrix_ = numpy.array(1.0 * trans_matrix_ / \
            numpy.stack([numpy.sum(trans_matrix_, axis=1)] * self.n_states, axis=1), dtype='float32')
        # 如果有observes则统计emission_matrix_
        if type(observes) != type(None):
            for i in range(n_samples):
                n_sequences = lengths[i]
                emission_matrix_[states[i,0], observes[i,0]] += 1
                for t in range(1, lengths[i]):
                    emission_matrix_[states[i,t], observes[i,t]] += 1
            emission_matrix_ = numpy.array(1.0 * emission_matrix_ / \
                numpy.stack([numpy.sum(emission_matrix_, axis=1)] * self.n_features, axis=1), dtype='float32')
        
        return start_prob_, trans_matrix_, emission_matrix_

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


if __name__ == '__main__':
    method = 'sample'
    if method == 'sample':
        pi = numpy.array([0.2, 0.4, 0.4], dtype='float32')
        A = numpy.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]], dtype='float32')
        B = numpy.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]], dtype='float32')
        model = HMM(n_states=3, n_features=2, start_prob=pi, trans_matrix=A, emission_matrix=B)
        n_samples = 1000
        lengths = numpy.array([10] * n_samples, dtype='int32')
        states, observes = model.samples(n_samples, lengths)
        param_pi, param_A, param_B = model.statistic(states, lengths, observes)
        print(param_pi)
        print(param_A)
        print(param_B)
