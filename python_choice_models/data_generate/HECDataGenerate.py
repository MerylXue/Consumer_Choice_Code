# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


import numpy as np
import math
from python_choice_models.data_generate.ConsumerChoiceModel import ConsumerChoice
from python_choice_models.utils import addtwodimdict

class HEC_bench(ConsumerChoice):
    def __init__(self, n, num_assor, Single_Assor_Data, lb, ub, n_samples):
        super(HEC_bench, self).__init__(n, num_assor, Single_Assor_Data, lb, ub, n_samples)
        self.lambda_ = np.random.uniform(0, 1, size=self.n+1)


    def GenerateData(self):


        self.utility = np.concatenate(([0], self.utility),
                                      axis=0)  # randomly generated utility for no purchase option and all n products
        G_test = np.zeros((self.T_test,self.n+1))
        rank_test = np.zeros((self.T_test, self.n+1),int)
        #sort the products in utility increasing
        #The first num_choice are in included in the assortments, the last several are not provided
        for i in range(self.T_test):
            temp_utility_test = np.zeros(self.n+1)
            for j in range(self.n+1):
                if(self.test_choice_set[i][j] == 0):
                    temp_utility_test[j] = 10000
                else:
                    temp_utility_test[j] = self.utility[j]

            rank_test[i] = np.argsort(temp_utility_test)

        prob_dict = dict()

        for i in range(self.T_test):
            num_choice = sum(self.test_choice_set[i])
               
                #the utility are ranked in increasing sequence
                #we have to get back to u1<=u2<=u3...
            for j in range(num_choice):
                sum_u = 0
                sum_lambda = 0
                for m in np.arange(j, num_choice):
                    sum_u += self.lambda_[rank_test[i][m]] * (self.utility[rank_test[i][m]]-self.utility[rank_test[i][j]])
                    sum_lambda += self.lambda_[rank_test[i][m]]
                G_test[i][rank_test[i][j]] = self.lambda_[rank_test[i][j]]/sum_lambda * math.exp(- sum_u)

            for j in range(num_choice):
                sum_G = 0
                sum_lambda = 0
                for m in range(0,j):
                    # sum_G += 1/(num_choice-(m+1)) * G_test[i][rank_test[i][m]]
                    sum_lambda = sum([self.lambda_[rank_test[i][l]] for l in range(m+1,num_choice)])
                    sum_G += self.lambda_[rank_test[i][j]]/sum_lambda * G_test[i][rank_test[i][m]]

                self.prob_test[i][rank_test[i][j]] = G_test[i][rank_test[i][j]]-sum_G

            self.test_choices[i] = np.random.choice(range(self.n+1), 1, p=self.prob_test[i])

            addtwodimdict(prob_dict, tuple(self.test_choice_set[i][1:].tolist()),self.prob_test[i])

        for t in range(self.n_samples):
            for i in range(self.MAXIMUM_T):
                self.prob_train[t][i]= prob_dict[tuple(self.train_offered_set[t][i][1:].tolist())]
                self.train_choices[t][i] =  np.random.choice(range(self.n+1), 1, p=self.prob_train[t][i])

        return self.train_choices, self.prob_train, self.test_choices, self.prob_test, self.train_offered_set
    

    def ProbPerturb(self, delta):
        delta_pro = np.zeros((self.T_test,self.n+1))

        self.utility = np.concatenate(([0], self.utility),
                                      axis=0)  # randomly generated utility for no purchase option and all n products

        for j in range(self.T_test):
            delta_pro[j] = np.random.uniform(1-delta,1+delta,self.n+1)


        G_test = np.zeros((self.T_test,self.n+1))
        rank_test = np.zeros((self.T_test, self.n+1),int)
        #sort the products in utility increasing
        #The first num_choice are in included in the assortments, the last several are not provided
        for i in range(self.T_test):
            temp_utility_test = np.zeros(self.n+1)
            for j in range(self.n+1):
                if(self.test_choice_set[i][j] == 0):
                    temp_utility_test[j] = 10000
                else:
                    temp_utility_test[j] = self.utility[j]

            rank_test[i] = np.argsort(temp_utility_test)

        prob_dict = dict()

        for i in range(self.T_test):
            num_choice = sum(self.test_choice_set[i])

            # the utility are ranked in increasing sequence
            # we have to get back to u1<=u2<=u3...
            for j in range(num_choice):
                sum_u = 0
                sum_lambda = 0
                for m in np.arange(j, num_choice):
                    sum_u += self.lambda_[rank_test[i][m]] * (
                                self.utility[rank_test[i][m]] - self.utility[rank_test[i][j]])
                    sum_lambda += self.lambda_[rank_test[i][m]]
                G_test[i][rank_test[i][j]] = self.lambda_[rank_test[i][j]] / sum_lambda * math.exp(- sum_u)

            for j in range(num_choice):
                sum_G = 0
                sum_lambda = 0
                for m in range(0, j):
                    # sum_G += 1/(num_choice-(m+1)) * G_test[i][rank_test[i][m]]
                    sum_lambda = sum([self.lambda_[rank_test[i][l]] for l in range(m + 1, num_choice)])
                    sum_G += self.lambda_[rank_test[i][j]] / sum_lambda * G_test[i][rank_test[i][m]]

                self.prob_test[i][rank_test[i][j]] = G_test[i][rank_test[i][j]]-sum_G

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
        return self.utility
