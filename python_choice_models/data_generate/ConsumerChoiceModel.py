# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.

import numpy as np
import itertools
import math
class ConsumerChoice:
    """
        the consumer choice models for generating synthetic transactions data.
    """
    def __init__(self, n, num_assor, Single_Assor_Data,  lb, ub,  n_samples):
        #number of training
        self.num_assor = num_assor     #number of assortments
        self.Single_Assor_Data = Single_Assor_Data #sample size for each assortment
        self.MAXIMUM_T = sum(self.Single_Assor_Data) #number of data pieces
        self.lb = lb
        self.ub = ub
        self.n_samples = n_samples
        self.n = n
        self.utility = np.random.uniform(lb, ub ,n)

        # Generate training set that can cover all products
        offered_set_binary_limit =  np.random.randint(2,size = (self.n_samples,self.num_assor,self.n)) # training assortments

        ## deal with the case that one product never appears in the offered set
        for t in range(self.n_samples):
            sum_offer = np.sum(offered_set_binary_limit[t], axis = 0)
            for j in range(n):
                if sum_offer[j] == 0:
                    #randomly choose a assort and insert into it
                    idx_assor = np.random.randint(0, self.num_assor)
                    offered_set_binary_limit[t][idx_assor][j] = 1

        for t in range(self.n_samples):
            for i in range(self.num_assor):
                sum_offer = np.sum(offered_set_binary_limit[t][i])
                if sum_offer == 0:
                    idx_product = np.random.randint(0, self.n)
                    offered_set_binary_limit[t][i][idx_product] = 1
         #

        self.train_offered_set = np.zeros((self.n_samples,self.MAXIMUM_T,n+1),int)

        for t in range(self.n_samples):
            for i in range(self.num_assor):
                for k in range(Single_Assor_Data[i]):
                    index = sum(Single_Assor_Data[0:i])+k
                    self.train_offered_set[t][index][0] = 1
                    self.train_offered_set[t][index][1:] = offered_set_binary_limit[t][i][0:]

        self.prob_train = np.zeros((self.n_samples,self.MAXIMUM_T,n+1))
        self.train_choices = np.zeros((self.n_samples,self.MAXIMUM_T),int) #randomly generated choices
        print(self.train_offered_set)

        #test sets
        #Generate all the test data (all assortment)
        lst = list(itertools.product([0, 1], repeat=self.n))
        # lst.pop(0)
        self.T_test = int(math.pow(2,n))
        # self.T_test = len(lst)
        new_set_binary_limit = np.asarray(lst) #testing assortment
        self.test_choice_set = np.zeros((self.T_test, n+1), int)

        self.prob_test = np.zeros((self.T_test, self.n+1))
        self.test_choices = np.zeros(self.T_test, int)

        for j in range(self.T_test):
            self.test_choice_set[j][0] = 1
            self.test_choice_set[j][1:] = new_set_binary_limit[j][0:]

