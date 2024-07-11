# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


import numpy as np
import math
import random
from python_choice_models.utils import addtwodimdict
from python_choice_models.data_generate.ECDataGenerate import EC_bench
from python_choice_models.data_generate.MCDataGenerate import MC_bench


class MIX_EC_MKV(EC_bench, MC_bench):
    def __init__(self, n, num_assor, Single_Assor_Data, lb, ub, n_samples):
        # super(MIX_EC_MKV, self).__init__(n, num_assor, Single_Assor_Data, lb, ub)
        EC_bench.__init__(self,n, num_assor, Single_Assor_Data, lb, ub, n_samples)
        MC_bench.__init__(self,n, num_assor, Single_Assor_Data, lb, ub, n_samples)
        # self.utility = np.concatenate(([0], self.utility), axis=0)  # randomly generated utility for no purchase option and all n products

    def GenerateData(self):
        # self.utility = np.concatenate(([0], self.utility),
        #                               axis=0)  # randomly generated utility for no purchase option and all n products
        train_choice, prob_train, test_choice, prob_test_exp, train_offered_set = EC_bench.GenerateData(self)
        train_choice, prob_train, test_choice, prob_test_mc, train_offered_set = MC_bench.GenerateData(self)
        prob_dict = dict()

        #mix the test prob
        _lambda = random.uniform(0.2,0.8)
        for i in range(self.T_test):
            self.prob_test[i] = prob_test_exp[i] * _lambda + prob_test_mc[i] * (1 - _lambda)
            self.test_choices[i] = np.random.choice(range(self.n + 1), 1, p=self.prob_test[i])
            for j in range(self.n+1):
                if self.test_choice_set[i][j] == 0 and self.prob_test[i][j] > 0:
                    print(self.test_choice_set[i], self.prob_test[i])
            addtwodimdict(prob_dict, tuple(tuple(self.test_choice_set[i][1:].tolist())), self.prob_test[i])

        for t in range(self.n_samples):
            for i in range(self.MAXIMUM_T):
                self.prob_train[t][i]= prob_dict[tuple(self.train_offered_set[t][i][1:].tolist())]
                self.train_choices[t][i] = np.random.choice(range(self.n+1), 1, p=self.prob_train[t][i])

        return self.train_choices, self.prob_train, self.test_choices, self.prob_test, self.train_offered_set

    def ProbPerturb(self, delta):
        prob_dict = dict()
        delta_pro = np.zeros((self.T_test,self.n+1))
        for j in range(self.T_test):
            delta_pro[j] = np.random.uniform(1-delta,1+delta,self.n+1)
        train_choice, prob_train, test_choice, prob_test_exp, train_offered_set = EC_bench.GenerateData()
        train_choice, prob_train, test_choice, prob_test_mc, train_offered_set = MC_bench.GenerateData()
        prob_dict = dict()

        # mix the test prob
        _lambda = random.uniform(0.2, 0.8)
        for i in range(self.T_test):
            self.prob_test[i] = prob_test_exp[i] * _lambda + prob_test_mc[i] * (1 - _lambda)

        for i in range(self.T_test):
            for j in range(self.n + 1):
                self.prob_test[i][j] = self.prob_test[i][j] * delta_pro[i][j]
            self.prob_test[i] = self.prob_test[i] / sum(self.prob_test[i])
            self.test_choices[i] = np.random.choice(range(self.n + 1), 1, p=self.prob_test[i])
            addtwodimdict(prob_dict, tuple(self.test_choice_set[i][1:].tolist()), self.prob_test[i])

        for t in range(self.n_samples):
            for i in range(self.MAXIMUM_T):
                self.prob_train[t][i] = prob_dict[tuple(self.train_offered_set[t][i][1:].tolist())]
                self.train_choices[t][i] = np.random.choice(range(self.n + 1), 1, p=self.prob_train[t][i])

        return self.train_choices, self.prob_train, self.test_choices, self.prob_test, self.train_offered_set

    def GetIniParam(self):
        return self.P_arrival, self.P_transition, self.utility

