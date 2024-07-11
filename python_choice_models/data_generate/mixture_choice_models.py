# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


import numpy as np
import math
import random

from python_choice_models.data_generate.MNLDataGenerate import MNL_bench
from python_choice_models.data_generate.ECDataGenerate import EC_bench
from python_choice_models.data_generate.LCMNLDataGenerate import LCMNL_bench
from python_choice_models.data_generate.MCDataGenerate import MC_bench
from python_choice_models.data_generate.NestedDataGenerate import Nested_bench
from python_choice_models.data_generate.RankDataGenerate import rank_bench
from python_choice_models.data_generate.Half_IIA import Half_IIA_bench
from python_choice_models.utils import addtwodimdict

class Mixture_Choice_Models(MNL_bench, EC_bench, LCMNL_bench, MC_bench, Nested_bench, rank_bench, Half_IIA_bench):
    def __init__(self, n, num_assor, Single_Assor_Data, lb, ub, n_samples, model1, model2):
        # super(MIX_EC_MKV, self).__init__(n, num_assor, Single_Assor_Data, lb, ub)
        if 'mnl' in [model1, model2]:
            MNL_bench.__init__(self, n, num_assor, Single_Assor_Data, lb, ub, n_samples)
        if 'exp' in [model1, model2]:
            EC_bench.__init__(self, n, num_assor, Single_Assor_Data, lb, ub,n_samples)
        if 'lc' in [model1, model2]:
            LCMNL_bench.__init__(self, n, num_assor, Single_Assor_Data, lb, ub,n_samples)
        if 'mkv' in [model1, model2]:
            MC_bench.__init__(self, n, num_assor, Single_Assor_Data, lb, ub,n_samples)
        if 'nl' in [model1, model2]:
            Nested_bench.__init__(self, n, num_assor, Single_Assor_Data, lb, ub,n_samples)
        if 'rl' in [model1, model2]:
            rank_bench.__init__(self, n, num_assor, Single_Assor_Data, lb, ub,n_samples)
        if 'emnl' in [model1, model2]:
            Half_IIA_bench.__init__(self, n, num_assor, Single_Assor_Data, lb, ub,n_samples)
        self.model_lst = [model1, model2]


    def GenerateData(self):
        prob_test_lst = []
        if 'mnl' in self.model_lst:
            train_choice, prob_train, test_choice, prob_test_mnl, train_offered_set = MNL_bench.GenerateData(self)
            prob_test_lst.append(prob_test_mnl)
        if 'exp' in self.model_lst:
            train_choice, prob_train, test_choice, prob_test_exp, train_offered_set = EC_bench.GenerateData(self)
            prob_test_lst.append(prob_test_exp)
        if 'lc' in self.model_lst:
            train_choice, prob_train, test_choice, prob_test_lc, train_offered_set = LCMNL_bench.GenerateData(self)
            prob_test_lst.append(prob_test_lc)
        if 'mkv' in self.model_lst:
            train_choice, prob_train, test_choice, prob_test_mkv, train_offered_set = MC_bench.GenerateData(self)
            prob_test_lst.append(prob_test_mkv)
        if 'nl' in self.model_lst:
            train_choice, prob_train, test_choice, prob_test_nl, train_offered_set = Nested_bench.GenerateData(self)
            prob_test_lst.append(prob_test_nl)
        if 'rl' in self.model_lst:
            train_choice, prob_train, test_choice, prob_test_rl, train_offered_set = rank_bench.GenerateData(self)
            prob_test_lst.append(prob_test_rl)
        if 'emnl' in self.model_lst:
            train_choice, prob_train, test_choice, prob_test_emnl, train_offered_set = Half_IIA_bench.GenerateData(self)
            prob_test_lst.append(prob_test_emnl)

        if len(prob_test_lst) != 2:
            print('something wrong with the data input!!')
        prob_test0 = prob_test_lst[0]
        prob_test1 = prob_test_lst[1]
        prob_dict = dict()
        # mix the test prob
        _lambda = random.uniform(0, 1)
        for i in range(self.T_test):
            self.prob_test[i] = prob_test0[i] * _lambda + prob_test1[i] * (1 - _lambda)
            self.test_choices[i] = np.random.choice(range(self.n + 1), 1, p=self.prob_test[i])
            # for j in range(self.n + 1):
            #     if self.test_choice_set[i][j] == 0 and self.prob_test[i][j] > 0:
            #         print(self.test_choice_set[i], self.prob_test[i])
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

        prob_test_lst = []
        if 'mnl' in self.model_lst:
            train_choice, prob_train, test_choice, prob_test_mnl, train_offered_set = MNL_bench.GenerateData(self)
            prob_test_lst.append(prob_test_mnl)
        if 'exp' in self.model_lst:
            train_choice, prob_train, test_choice, prob_test_exp, train_offered_set = EC_bench.GenerateData(self)
            prob_test_lst.append(prob_test_exp)
        if 'lc' in self.model_lst:
            train_choice, prob_train, test_choice, prob_test_lc, train_offered_set = LCMNL_bench.GenerateData(self)
            prob_test_lst.append(prob_test_lc)
        if 'mkv' in self.model_lst:
            train_choice, prob_train, test_choice, prob_test_mkv, train_offered_set = MC_bench.GenerateData(self)
            prob_test_lst.append(prob_test_mkv)
        if 'nl' in self.model_lst:
            train_choice, prob_train, test_choice, prob_test_nl, train_offered_set = Nested_bench.GenerateData(self)
            prob_test_lst.append(prob_test_nl)
        if 'rl' in self.model_lst:
            train_choice, prob_train, test_choice, prob_test_rl, train_offered_set = rank_bench.GenerateData(self)
            prob_test_lst.append(prob_test_rl)
        if 'emnl' in self.model_lst:
            train_choice, prob_train, test_choice, prob_test_emnl, train_offered_set = Half_IIA_bench.GenerateData(self)
            prob_test_lst.append(prob_test_emnl)

        if len(prob_test_lst) != 2:
            print('something wrong with the data input!!')
        prob_test0 = prob_test_lst[0]
        prob_test1 = prob_test_lst[1]
        # print(prob_test0)
        # print(prob_test1)
        # mix the test prob
        _lambda = random.uniform(0, 1)
        for i in range(self.T_test):
            self.prob_test[i] = prob_test0[i] * _lambda + prob_test1[i] * (1 - _lambda)

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
        return self.utility
