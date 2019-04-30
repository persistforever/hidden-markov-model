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

    def _get_emission_map(self, observe):
        # 通过emission_matrix_和observe，获得emission_map_
        n_sequence = len(observe)
        emission_map_ = self.emission_matrix_[:,observe]

        return emission_map_

    def _forward(self, observe, emission_map_):
        # 前向算法计算alpha
        n_sequence = len(observe)
        alpha = numpy.zeros((n_sequence, self.n_states), dtype='float32')
        alpha[0,:] = self.start_prob_ * emission_map_[:,0]
        for t in range(1, n_sequence):
            alpha[t,:] = numpy.transpose(numpy.dot(alpha[t-1,:], self.trans_matrix_)) * emission_map_[:,t]
        
        return alpha

    def _backward(self, observe, emission_map_):
        # 后向算法计算beta
        n_sequence = len(observe)
        beta = numpy.zeros((n_sequence, self.n_states), dtype='float32')
        beta[n_sequence-1,:] = numpy.ones((self.n_states, ), dtype='float32')
        for t in range(n_sequence-2, -1, -1):
            beta[t,:] = numpy.dot(self.trans_matrix_, emission_map_[:,t+1] * beta[t+1,:])
        
        return beta

    def calculate_probability(self, observes, mode='forward'):
        # 计算损失函数
        n_samples = len(observes)
        probs = numpy.zeros((n_samples, ), dtype='float32')
        logprob = 0
        for i in range(n_samples):
            emission_map_ = self._get_emission_map(observes[i])
            if mode == 'forward':
                alpha = self._forward(observes[i], emission_map_)
                probs[i] = numpy.sum(alpha[len(observes[i])-1,:])
            elif mode == 'backward':
                beta = self._backward(observes[i], emission_map_)
                probs[i] = numpy.dot(self.start_prob_, emission_map_[:,0] * beta[0,:])
            logprob += numpy.log(probs[i] + 1e-12)
        logprob = 1.0 * logprob / n_samples
        
        return logprob, probs

    def viterbi_inference(self, observe, emission_map=None):
        # 使用viterbi算法进行预测
        n_sequences = len(observe)
        states = numpy.zeros((n_sequences, ), dtype='int32')
        deltas = numpy.zeros((n_sequences, self.n_states), dtype='float32')
        emission_map_ = self._get_emission_map(observe) if type(emission_map) == type(None) else emission_map
        deltas[0,:] = self.start_prob_ * emission_map_[:,0]
        for t in range(1, n_sequences):
            deltas[t,:] = numpy.max(
                numpy.reshape(deltas[t-1,:], (self.n_states, 1)) * self.trans_matrix_, axis=0) * emission_map_[:,t]
        states[n_sequences-1] = numpy.argmax(deltas[n_sequences-1,:])
        for t in range(n_sequences-2, -1, -1):
            states[t] = numpy.argmax(deltas[t] * self.trans_matrix_[:,states[t+1]])
        
        return deltas, states

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
        temp_start_prob_ = numpy.zeros((self.n_states, ), dtype='int32')
        temp_trans_matrix_ = numpy.zeros((self.n_states, self.n_states), dtype='int32')
        temp_emission_matrix_ = numpy.zeros((self.n_states, self.n_features), dtype='int32')
        n_samples = observes.shape[0]
        for i in range(n_samples):
            n_sequences = lengths[i]
            temp_start_prob_[states[i,0]] += 1
            for t in range(1, lengths[i]):
                temp_trans_matrix_[states[i,t-1], states[i,t]] += 1
        start_prob_ = numpy.array(1.0 * temp_start_prob_ / numpy.sum(temp_start_prob_), dtype='float32')
        trans_matrix_ = numpy.array(1.0 * temp_trans_matrix_ / \
            numpy.stack([numpy.sum(temp_trans_matrix_, axis=1)] * self.n_states, axis=1), dtype='float32')
        
        # 如果有observes则统计emission_matrix_
        if type(observes) != type(None):
            for i in range(n_samples):
                n_sequences = lengths[i]
                temp_emission_matrix_[states[i,0], observes[i,0]] += 1
                for t in range(1, lengths[i]):
                    temp_emission_matrix_[states[i,t], observes[i,t]] += 1
            emission_matrix_ = numpy.array(1.0 * temp_emission_matrix_ / \
                numpy.stack([numpy.sum(temp_emission_matrix_, axis=1)] * self.n_features, axis=1), dtype='float32')
        
        return start_prob_, trans_matrix_, emission_matrix_, \
            (temp_start_prob_, temp_trans_matrix_, temp_emission_matrix_)

    def fit(self, observes, lengths, n_steps=1000):
        # 只根据观测值，通过模型的方法训练参数
        n_samples = observes.shape[0]
        for n in range(n_steps):
            new_start_prob_ = numpy.zeros((n_samples, self.n_states), dtype='float32')
            new_trans_matrix_ = numpy.zeros((n_samples, self.n_states, self.n_states), dtype='float32')
            new_emission_matrix_ = numpy.zeros((n_samples, self.n_states, self.n_features), dtype='float32')
            logprob, probs = model.calculate_probability(observes, mode='forward')
            print('log of probability: %.8f' % (logprob))
            
            for i in range(n_samples):
                n_sequences = lengths[i]
                observe = observes[i][0:n_sequences]
                # 计算emission_map_
                emission_map_ = self._get_emission_map(observe)
                # 计算alpha和beta
                alpha = self._forward(observe, emission_map_)
                beta = self._backward(observe, emission_map_)
                
                # 计算new_start_prob_
                nume = alpha[0,:] * beta[0,:]
                deno = numpy.sum(alpha[0,:] * beta[0,:])
                new_start_prob_[i,:] = 1.0 * nume / (deno + 1e-12)
                
                # 计算new_trans_matrix_
                nume = self.trans_matrix_ * numpy.dot(
                    numpy.transpose(alpha[:,:])[:,:-1], beta[1:,:] * numpy.transpose(emission_map_)[1:,:])
                deno = numpy.stack([numpy.sum(alpha[:-1,:] * beta[:-1,:], axis=0)] * self.n_states, axis=1)
                new_trans_matrix_[i,:,:] = nume / (deno + 1e-12)
            
                # 计算new_emission_matrix_
                nume = numpy.dot(numpy.transpose(alpha * beta), numpy.transpose(numpy.eye(self.n_features)[:,observe]))
                deno = numpy.stack([numpy.sum(alpha * beta, axis=0)] * self.n_features, axis=1)
                new_emission_matrix_[i,:,:] = nume / (deno + 1e-12)
                
            # 计算batch_logprob
            self.start_prob_ = numpy.mean(new_start_prob_, axis=0)
            self.trans_matrix_ = numpy.mean(new_trans_matrix_, axis=0)
            self.emission_matrix_ = numpy.mean(new_emission_matrix_, axis=0)
        

if __name__ == '__main__':
    method = 'train'
    
    if method == 'sample':
        pi = numpy.array([0.2, 0.4, 0.4], dtype='float32')
        A = numpy.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]], dtype='float32')
        B = numpy.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]], dtype='float32')
        model = HMM(n_states=3, n_features=2, start_prob=pi, trans_matrix=A, emission_matrix=B)
        n_samples = 1000
        lengths = numpy.array([10] * n_samples, dtype='int32')
        states, observes = model.samples(n_samples, lengths)
        param_pi, param_A, param_B, _ = model.statistic(states, lengths, observes)
        print(param_pi)
        print(param_A)
        print(param_B)
    
    elif method == 'prob':
        pi = numpy.array([0.2, 0.4, 0.4], dtype='float32')
        A = numpy.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]], dtype='float32')
        B = numpy.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]], dtype='float32')
        model = HMM(n_states=3, n_features=2, start_prob=pi, trans_matrix=A, emission_matrix=B)
        observe = numpy.array([[0,1,0]], dtype='int32')
        logprob, probs = model.calculate_probability(observe, mode='forward')
        print(probs)
    
    elif method == 'inference':
        pi = numpy.array([0.2, 0.4, 0.4], dtype='float32')
        A = numpy.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]], dtype='float32')
        B = numpy.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]], dtype='float32')
        model = HMM(n_states=3, n_features=2, start_prob=pi, trans_matrix=A, emission_matrix=B)
        observe = numpy.array([0,1,0], dtype='int32')
        deltas, states = model.viterbi_inference(observe)
        print(deltas)
        print(states)
    
    elif method == 'train':
        pi = numpy.array([0.2, 0.4, 0.4], dtype='float32')
        A = numpy.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]], dtype='float32')
        B = numpy.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]], dtype='float32')
        model_sampler = HMM(n_states=3, n_features=2, start_prob=pi, trans_matrix=A, emission_matrix=B)
        n_samples = 100
        lengths = numpy.array([10] * n_samples, dtype='int32')
        _, observes = model_sampler.samples(n_samples, lengths)
        model = HMM(n_states=3, n_features=2, start_prob=pi, trans_matrix=A, emission_matrix=B)
        model.fit(observes, lengths, n_steps=1000)
        print(model.start_prob_)
        print(model.trans_matrix_)
        print(model.emission_matrix_)
