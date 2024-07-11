
# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.

import numpy as np
import math
from python_choice_models.data_generate.ConsumerChoiceModel import ConsumerChoice
from python_choice_models.utils import addtwodimdict



class MNL_bench(ConsumerChoice):
    def __init__(self, n, num_assor, Single_Assor_Data, lb, ub,n_samples):
        super(MNL_bench, self).__init__( n, num_assor, Single_Assor_Data, lb, ub,n_samples)

    def GenerateData(self):
        prob_dict = dict()

        for i in range(self.T_test):
            numerator = 1
            for j in range(self.n):
                numerator += math.exp(self.utility[j])*self.test_choice_set[i][j+1]

            self.prob_test[i][0] = 1/numerator
            for j in range(self.n):
                self.prob_test[i][j+1] = math.exp(self.utility[j])*self.test_choice_set[i][j+1]/numerator

            self.test_choices[i] = np.random.choice(range(self.n+1), 1, p=self.prob_test[i])

            addtwodimdict(prob_dict, tuple(self.test_choice_set[i][1:].tolist()),self.prob_test[i])

        for t in range(self.n_samples):
            for i in range(self.MAXIMUM_T):
                self.prob_train[t][i] = prob_dict[tuple(self.train_offered_set[t][i][1:].tolist())]
                self.train_choices[t][i] = np.random.choice(range(self.n+1), 1, p=self.prob_train[t][i])

        return self.train_choices, self.prob_train,  self.test_choices, self.prob_test, self.train_offered_set

    def GetIniParam(self):
        return self.utility

    def ProbPerturb(self, delta):

        #Generate all the test data (all assortment)
        delta_pro = np.zeros((self.T_test,self.n+1))

        for j in range(self.T_test):
            delta_pro[j] = np.random.uniform(1-delta,1+delta,self.n+1)

        prob_dict = dict()
   
        for i in range(self.T_test):
            numerator = 1
            for j in range(self.n):
                numerator += math.exp(self.utility[j])*self.test_choice_set[i][j+1]

            self.prob_test[i][0] = 1/numerator
            for j in range(self.n):
                self.prob_test[i][j+1] = math.exp(self.utility[j])*self.test_choice_set[i][j+1]/numerator

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
 
