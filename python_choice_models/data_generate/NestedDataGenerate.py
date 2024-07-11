# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


import numpy as np
import math
from python_choice_models.data_generate.ConsumerChoiceModel import ConsumerChoice
from python_choice_models.utils import addtwodimdict
class Nested_bench(ConsumerChoice):
    def __init__(self, n, num_assor, Single_Assor_Data, lb, ub, n_samples):
            super(Nested_bench, self).__init__(n, num_assor, Single_Assor_Data, lb, ub, n_samples)
            self.N_nest = 3
            self._lambda = np.random.uniform(1e-1,1-1e-1,self.N_nest)

    def GenerateData(self):
        prob_dict = dict()

        for i in range(self.T_test):
            #randomly pick a customer type given the probability     
            numerator = 0
            sum_g1 = 0 #sum of exp of utility/lambda in nest 1
            sum_g2 = 0 # sum of exp of utility/lambda in nest 2
                
            numerator += math.pow(np.exp(0),self._lambda[0])
            for j in range(self.n):
                if ((j+1) % 2 == 1):
                    sum_g1 += np.exp(self.utility[j]/self._lambda[1])*self.test_choice_set[i][j+1]
                elif ((j+1) % 2 == 0):
                    sum_g2 += np.exp(self.utility[j]/self._lambda[2])*self.test_choice_set[i][j+1]

            numerator += math.pow(sum_g1, self._lambda[1])
            numerator += math.pow(sum_g2, self._lambda[2])

            self.prob_test[i][0] = 1/numerator
            for j in range(self.n):
                if (j+1) % 2 == 1:
                    if self.test_choice_set[i][j+1] > 0:
                        self.prob_test[i][j+1] = np.exp(self.utility[j]/self._lambda[1])*pow(sum_g1, self._lambda[1]-1)/numerator
                elif (j+1) % 2 == 0:
                    if self.test_choice_set[i][j+1] > 0:
                        self.prob_test[i][j+1] = np.exp(self.utility[j]/self._lambda[2])*pow(sum_g2, self._lambda[2]-1)/numerator

            self.test_choices[i] = np.random.choice(range(self.n+1), 1, p=self.prob_test[i])
            addtwodimdict(prob_dict, tuple(self.test_choice_set[i][1:].tolist()),self.prob_test[i])

        for t in range(self.n_samples):
            for i in range(self.MAXIMUM_T):
                self.prob_train[t][i]= prob_dict[tuple(self.train_offered_set[t][i][1:].tolist())]
                self.train_choices[t][i] = np.random.choice(range(self.n+1), 1, p=self.prob_train[t][i])

        return self.train_choices, self.prob_train,  self.test_choices, self.prob_test, self.train_offered_set


    def ProbPerturb(self, delta):
        prob_dict = dict()
        delta_pro = np.zeros((self.T_test,self.n+1))

        for j in range(self.T_test):
            delta_pro[j] = np.random.uniform(1-delta,1+delta,self.n+1)

        for i in range(self.T_test):
            #randomly pick a customer type given the probability     
            numerator = 0
            sum_g1 = 0 #sum of exp of utility/lambda in nest 1
            sum_g2 = 0 # sum of exp of utility/lambda in nest 2
                
            numerator += math.pow(np.exp(0),self._lambda[0])
            for j in range(self.n):
                if ((j+1) % 2 == 1):
                    sum_g1 += np.exp(self.utility[j]/self._lambda[1])*self.test_choice_set[i][j+1]
                elif ((j+1) % 2 == 0):
                    sum_g2 += np.exp(self.utility[j]/self._lambda[2])*self.test_choice_set[i][j+1]

            numerator += math.pow(sum_g1, self._lambda[1])
            numerator += math.pow(sum_g2, self._lambda[2])

            self.prob_test[i][0] = 1/numerator
            for j in range(self.n):
                if (j+1) % 2 == 1:
                    if self.test_choice_set[i][j+1] > 0:
                        self.prob_test[i][j+1] = np.exp(self.utility[j]/self._lambda[1])*pow(sum_g1, self._lambda[1]-1)/numerator
                elif (j+1) % 2 == 0:
                    if self.test_choice_set[i][j+1] > 0:
                        self.prob_test[i][j+1] = np.exp(self.utility[j]/self._lambda[2])*pow(sum_g2, self._lambda[2]-1)/numerator

        for i in range(self.T_test):
            for j in range(self.n + 1):
                self.prob_test[i][j] = self.prob_test[i][j]*delta_pro[i][j]
            self.prob_test[i] = self.prob_test[i]/sum(self.prob_test[i])
            self.test_choices[i] = np.random.choice(range(self.n+1), 1, p=self.prob_test[i])
            addtwodimdict(prob_dict,tuple(self.test_choice_set[i][1:].tolist()),self.prob_test[i])
        for t in range(self.n_samples):
            for i in range(self.MAXIMUM_T):
                self.prob_train[t][i] = prob_dict[tuple(self.train_offered_set[t][i][1:].tolist())]
                self.train_choices[t][i] =  np.random.choice(range(self.n+1), 1, p=self.prob_train[t][i])
        

        return self.train_choices, self.prob_train,  self.test_choices, self.prob_test, self.train_offered_set

    def GetIniParam(self):
        return self.utility, self._lambda