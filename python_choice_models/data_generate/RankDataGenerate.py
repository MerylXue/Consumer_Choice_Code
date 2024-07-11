# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


import numpy as np
import math
from python_choice_models.data_generate.ConsumerChoiceModel import ConsumerChoice
from python_choice_models.utils import addtwodimdict
import time
from numba import jit

class rank_bench(ConsumerChoice):
    def __init__(self, n, num_assor, Single_Assor_Data, lb, ub, n_samples):
        super(rank_bench, self).__init__(n, num_assor, Single_Assor_Data, lb, ub, n_samples)
        self.K_type = 50
        self.perm = np.zeros((self.K_type, self.n + 1), int)
        for i in range(self.K_type):
            self.perm[i] = np.random.permutation(self.n + 1)  # 0 no purchase
         #fractions of customers
        self.u = np.random.rand(self.K_type)
        sum_u = sum(self.u)
        for i in range(self.K_type):
            self.u[i] = self.u[i]/sum_u


    def GenerateData(self):
        prob_dict = dict()

        t1= time.time()
        for i in range(self.T_test):
            #randomly pick a customer type given the probability
            # print(self.test_choice_set[i])
            for k in range(self.K_type):
                perm_tmp = -1
                for j in range(self.n + 1):
                    if self.test_choice_set[i][self.perm[k][j]] > 0:
                        # perm_tmp.append(self.perm[k][j])
                        perm_tmp = self.perm[k][j]
                        break


                self.prob_test[i][perm_tmp] += self.u[k]

            self.prob_test[i] = self.prob_test[i]/sum(self.prob_test[i])
            self.test_choices[i] = np.random.choice(range(self.n+1), 1, p=self.prob_test[i])
            if self.test_choice_set[i][self.test_choices[i]] == 0:
                print(self.prob_test[i], self.test_choice_set[i])
            addtwodimdict(prob_dict,tuple(self.test_choice_set[i][1:].tolist()),self.prob_test[i])
        t2 = time.time()
        for t in range(self.n_samples):
            for i in range(self.MAXIMUM_T):
                self.prob_train[t][i] = prob_dict[tuple(self.train_offered_set[t][i][1:].tolist())]
                self.train_choices[t][i] = np.random.choice(range(self.n+1), 1, p=self.prob_train[t][i])
        print(self.prob_test)
        print("data generation time of rank list is %f"% (t2-t1))
        return self.train_choices, self.prob_train,  self.test_choices, self.prob_test, self.train_offered_set

    def ProbPerturb(self, delta):
        #Generate all the test data (all assortment)
        delta_pro = np.zeros((self.T_test,self.n+1))
        for j in range(self.T_test):
            delta_pro[j] = np.random.uniform(1-delta,1+delta,self.n+1)

        prob_dict = dict()

        for i in range(self.T_test):
            #randomly pick a customer type given the probability     
            for k in range(self.K_type):
                # perm_tmp = self.perm[k]*self.test_choice_set[i]
                perm_tmp = []
                for j in range(self.n + 1):
                    if self.test_choice_set[i][self.perm[k][j]] > 0:
                        perm_tmp.append(self.perm[k][j])
                first_item = perm_tmp[0]

                self.prob_test[i][first_item] += self.u[k]

            self.prob_test[i] = self.prob_test[i]/sum(self.prob_test[i])
        
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
        return self.perm, self.u
