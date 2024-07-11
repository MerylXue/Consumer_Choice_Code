# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


import numpy as np
import math
from python_choice_models.data_generate.ConsumerChoiceModel import ConsumerChoice
from python_choice_models.utils import addtwodimdict
from gurobipy import *

class MC_bench(ConsumerChoice):
    def __init__(self, n, num_assor, Single_Assor_Data, lb, ub,n_samples):
        super(MC_bench, self).__init__(n, num_assor, Single_Assor_Data, lb, ub,n_samples)

        self.P_arrival = np.random.rand(self.n+1)
        self.P_arrival[0] = 0
        self.P_arrival = self.P_arrival/sum(self.P_arrival)
        self.P_transition = np.random.rand(self.n+1,self.n+1) #i=1..n, j=0,..n
        ## increase the probability of choosing no option
        # for j in range(1,n+1):
        #     self.P_transition[j][0] = 100

        self.P_transition[0][0] = 1
        for j in range(n):
            self.P_transition[0][j+1] = 0

        for i in range(n):
            self.P_transition[i+1][i+1] = 0 #the probability of transitioning to the same 
            self.P_transition[i+1] = self.P_transition[i+1]/sum(self.P_transition[i+1])
        print(self.P_transition)

    def GenerateData(self):
        prob_dict = dict()


        for i in range(self.T_test):
            num_assor = int(sum(self.test_choice_set[i]))-1
            #(n-p,p+1)
            B = np.zeros((self.n-num_assor,num_assor+1)) #size num_assor*(N+1)    
            index_B = 0   
            for j in range(self.n):
                index_C = 0
                if(self.test_choice_set[i][j+1] == 0):
                    for k  in range(self.n+1):
                        if(self.test_choice_set[i][k] == 1):
                            B[index_B][index_C] =  self.P_transition[j+1][k]
                            index_C += 1
                    index_B += 1
                    #transition sub matrix
                # print(B)
                #n-p,n-p
            C = np.zeros((self.n-num_assor,self.n-num_assor))
            index_r = 0
            for k in range(self.n):
                if self.test_choice_set[i][k+1] == 0:
                    index_c = 0
                    for m in range(self.n):
                        if self.test_choice_set[i][m+1] == 0:
                            # print(index_r, index_c)
                            C[index_r][index_c] = self.P_transition[k+1][m+1]
                            index_c += 1 
                    index_r += 1
                # print(C)
            L1 = np.zeros((1,self.n-num_assor))
            index_l = 0
            for j in range(self.n+1):
                if self.test_choice_set[i][j] == 0:
                    L1[0][index_l] = self.P_arrival[j]
                    index_l += 1

            
            index_j = 0
            for j in range(self.n+1):
                e = np.zeros(num_assor+1)
                if self.test_choice_set[i][j] == 1:
                    e[index_j] = 1
                    # print(e)
                    self.prob_test[i][j] = self.P_arrival[j] + np.dot(np.dot(L1,np.dot(np.linalg.inv(np.identity(self.n-num_assor)-C),B)),e)
                    index_j += 1
                    # print(j, prob_test[i][j])
            self.prob_test[i] = self.prob_test[i]/np.sum(self.prob_test[i])
            self.test_choices[i] = np.random.choice(range(self.n+1), 1, p=self.prob_test[i])

            addtwodimdict(prob_dict, tuple(self.test_choice_set[i][1:].tolist()),self.prob_test[i])

        for t in range(self.n_samples):
            for i in range(self.MAXIMUM_T):
                self.prob_train[t][i]= prob_dict[tuple(self.train_offered_set[t][i][1:].tolist())]
                self.train_choices[t][i] =  np.random.choice(range(self.n+1), 1, p=self.prob_train[t][i])

        return self.train_choices, self.prob_train,  self.test_choices, self.prob_test, self.train_offered_set
    
    def ProbPerturb(self, delta):
        prob_dict = dict()
        delta_pro = np.zeros((self.T_test,self.n+1))

        for j in range(self.T_test):
            delta_pro[j] = np.random.uniform(1-delta,1+delta,self.n+1)


        for i in range(self.T_test):
            num_assor = int(sum(self.test_choice_set[i]))-1
            #(n-p,p+1)
            B = np.zeros((self.n-num_assor,num_assor+1)) #size num_assor*(N+1)    
            index_B = 0   
            for j in range(self.n):
                index_C = 0
                if(self.test_choice_set[i][j+1] == 0):
                    for k  in range(self.n+1):
                        if(self.test_choice_set[i][k] == 1):
                            B[index_B][index_C] =  self.P_transition[j+1][k]
                            index_C += 1
                    index_B += 1
                    #transition sub matrix
                # print(B)
                #n-p,n-p
            C = np.zeros((self.n-num_assor,self.n-num_assor))
            index_r = 0
            for k in range(self.n):
                if self.test_choice_set[i][k+1] == 0:
                    index_c = 0
                    for m in range(self.n):
                        if self.test_choice_set[i][m+1] == 0:
                            # print(index_r, index_c)
                            C[index_r][index_c] = self.P_transition[k+1][m+1]
                            index_c += 1 
                    index_r += 1
                # print(C)
            L1 = np.zeros((1,self.n-num_assor))
            index_l = 0
            for j in range(self.n+1):
                if self.test_choice_set[i][j] == 0:
                    L1[0][index_l] = self.P_arrival[j]
                    index_l += 1
            index_j = 0
            for j in range(self.n+1):
                e = np.zeros(num_assor+1)
                if self.test_choice_set[i][j] == 1:
                    e[index_j] = 1
                    # print(e)
                    self.prob_test[i][j] = self.P_arrival[j] + np.dot(np.dot(L1,np.dot(np.linalg.inv(np.identity(self.n-num_assor)-C),B)),e)
                    index_j += 1
                    # print(j, prob_test[i][j])
        for i in range(self.T_test):
            for j in range(self.n + 1):
                self.prob_test[i][j] = self.prob_test[i][j]*delta_pro[i][j]
            self.prob_test[i] = self.prob_test[i]/sum(self.prob_test[i])
            self.test_choices[i] = np.random.choice(range(self.n+1), 1, p=self.prob_test[i])
            addtwodimdict(prob_dict,tuple(self.test_choice_set[i][1:].tolist()),self.prob_test[i])
        for t in range(self.n_samples):
            for i in range(self.MAXIMUM_T):
                self.prob_train[t][i] = prob_dict[tuple(self.train_offered_set[t][i][1:].tolist())]
                self.train_choices[t][i] = np.random.choice(range(self.n+1), 1, p=self.prob_train[t][i])

        return self.train_choices, self.prob_train,  self.test_choices, self.prob_test, self.train_offered_set

    def Prob_dict_out(self):
        prob_dict = dict()
        prob_index_dict = dict()


        for i in range(self.T_test):
            num_assor = int(sum(self.test_choice_set[i])) - 1
            # (n-p,p+1)
            B = np.zeros((self.n - num_assor, num_assor + 1))  # size num_assor*(N+1)
            index_B = 0
            for j in range(self.n):
                index_C = 0
                if (self.test_choice_set[i][j + 1] == 0):
                    for k in range(self.n + 1):
                        if (self.test_choice_set[i][k] == 1):
                            B[index_B][index_C] = self.P_transition[j + 1][k]
                            index_C += 1
                    index_B += 1
                    # transition sub matrix
                # print(B)
                # n-p,n-p
            C = np.zeros((self.n - num_assor, self.n - num_assor))
            index_r = 0
            for k in range(self.n):
                if self.test_choice_set[i][k + 1] == 0:
                    index_c = 0
                    for m in range(self.n):
                        if self.test_choice_set[i][m + 1] == 0:
                            # print(index_r, index_c)
                            C[index_r][index_c] = self.P_transition[k + 1][m + 1]
                            index_c += 1
                    index_r += 1
                # print(C)
            L1 = np.zeros((1, self.n - num_assor))
            index_l = 0
            for j in range(self.n + 1):
                if self.test_choice_set[i][j] == 0:
                    L1[0][index_l] = self.P_arrival[j]
                    index_l += 1

            index_j = 0
            for j in range(self.n + 1):
                e = np.zeros(num_assor + 1)
                if self.test_choice_set[i][j] == 1:
                    e[index_j] = 1
                    # print(e)
                    self.prob_test[i][j] = self.P_arrival[j] + np.dot(
                        np.dot(L1, np.dot(np.linalg.inv(np.identity(self.n - num_assor) - C), B)), e)
                    index_j += 1
                    # print(j, prob_test[i][j])
            self.prob_test[i] = self.prob_test[i] / np.sum(self.prob_test[i])
            self.test_choices[i] = np.random.choice(range(self.n + 1), 1, p=self.prob_test[i])

            addtwodimdict(prob_index_dict, tuple(self.test_choice_set[i][1:].tolist()), i)
            addtwodimdict(prob_dict, tuple(self.test_choice_set[i][1:].tolist()), self.prob_test[i])
        return prob_dict,prob_index_dict

    def Check_MC_Data(self):
        prob_dict, prob_index_dict = self.Prob_dict_out()
        B_lst = list(itertools.product([0, 1], repeat=self.n))
        # print("Check supermodularity constraint")
        for B in B_lst:
            num_1 = sum(B)
            num_0 = self.n - num_1
            A_lst = list(itertools.product([0, 1], repeat=num_1))
            zero_index = [i  for i in range(len(B)) if B[i] == 0]
            one_index = [i for i in range(len(B)) if B[i] == 1]

            idx_B = prob_index_dict[tuple(B)]
            Prob_B = prob_dict[tuple(B)]

            for k in zero_index:
                B_add = list(B).copy()
                B_add[k] = 1
                Prob_B_add = prob_dict[tuple(B_add)]
                # if Prob_B[0] < Prob_B_add[0]:
                #     print(Prob_B, Prob_B_add)
                for i in one_index + [-1]:
                    if Prob_B[i + 1] < Prob_B_add[i + 1]:
                        print(Prob_B, Prob_B_add, i + 1)

                #generate A
                for A0 in A_lst:
                    A = list(B).copy()
                    idx = 0
                    for i in one_index:
                        A[i] = A0[idx]
                        idx += 1

                    A_add = A.copy()
                    A_add[k] = 1
                    # #supermodular contraint

                    Prob_A = prob_dict[tuple(A)]
                    Prob_A_add = prob_dict[tuple(A_add)]
                    one_index_A = [i for i in range(len(A)) if A[i] == 1]
                    for i in one_index_A + [-1]:
                        if Prob_A[i + 1] - Prob_A_add[i + 1] < Prob_B[i + 1] - Prob_B_add[i + 1]:
                            print(Prob_A, Prob_A_add, Prob_B, Prob_B_add)

    def Prob_perturb_new(self, delta):

        _gamma = 2*delta/(1-delta)
        # print(_gamma)
        prob_dict2 = {}
        print('start solving SUP------------------------')
        model = Model('Delta')
        model.modelSense = GRB.MINIMIZE

        prob_dict, prob_index_dict = self.Prob_dict_out()

        #add supermodular constraints
        # Generate a set B
        B_lst = list(itertools.product([0, 1], repeat=self.n))

        x = model.addVars(self.T_test, (self.n + 1),
                          lb=0.0, ub=1.0)
        # B_lst.pop(0)

        quad_expr = QuadExpr()
        for B in B_lst:
            num_1 = sum(B)
            num_0 = self.n - num_1
            A_lst = list(itertools.product([0, 1], repeat=num_1))
            zero_index = [i for i in range(len(B)) if B[i] == 0]
            one_index = [i for i in range(len(B)) if B[i] == 1]

            idx_B = prob_index_dict[tuple(B)]
            Prob_B = prob_dict[tuple(B)]
            for i in one_index + [-1]:
                if sum(list(B)) > 0:
                    quad_expr.add((x[idx_B, i + 1] - Prob_B[ i + 1]) * (x[idx_B, i + 1] - Prob_B[ i + 1]) * (sum(list(B)) + 1)/sum(list(B)))
                else:
                    quad_expr.add(
                        (x[idx_B,  i + 1] - Prob_B[ i + 1]) * (x[idx_B,  i + 1] - Prob_B[i + 1]))

            model.addConstrs((x[idx_B, i] <= B[i - 1] for i in range(1, self.n+1)))
            model.addConstr(quicksum([x[idx_B, i] for i in range(self.n+1)]) == 1.0)
            for k in zero_index:
                B_add = list(B).copy()
                B_add[k] = 1
                idx_B_add = prob_index_dict[tuple(B_add)]
                model.addConstrs((x[idx_B, i + 1] >= x[idx_B_add, i + 1] for i in one_index + [-1]))
                # model.addConstr(x[idx_B, 0] >= x[idx_B_add, 0])

                #generate A
                for A0 in A_lst:
                    A = list(B).copy()
                    idx = 0
                    for i in one_index:
                        A[i] = A0[idx]
                        idx += 1

                    A_add = A.copy()
                    A_add[k] = 1
                    # #supermodular contraint

                    idx_A = prob_index_dict[tuple(A)]

                    idx_A_add = prob_index_dict[tuple(A_add)]

                    one_index_A = [i for i in range(len(A)) if A[i] == 1]
                    model.addConstrs((x[idx_A, i + 1] - x[idx_A_add, i + 1] >= x[idx_B,i + 1] - x[idx_B_add,i + 1] for i in one_index_A  + [-1]))


        model.addConstr(quad_expr <= _gamma * len(B_lst), name='gap2')
        model.setObjective(quad_expr, GRB.MINIMIZE)

        # model.Params.MIPGap = 1e-3
        model.Params.TimeLimit = 900
        model.optimize()
        prob_new = np.zeros((self.T_test,self.n+1))

        for k in range(self.T_test):
            prob_new[k] = np.array([x[k,j].x for j in range(self.n + 1)])
        print(quad_expr.getValue() )
        # print("----------solution---------------")
        # print(prob_new)
        for k in range(self.T_test):
            for i in range(self.n + 1):
                if prob_new[k][i] < 0:
                    prob_new[k][i] = 0
                elif prob_new[k][i] > 1:
                    prob_new[k][i] = 1
            if np.sum(prob_new[k]) > 0:
                prob_new[k] = prob_new[k]/np.sum(prob_new[k])
            else:
                prob_new[k][0] = 1

        for i in range(self.T_test):
            # for j in range(self.n + 1):
            #     self.prob_test[i][j] = self.prob_test[i][j] * delta_pro[j]
            prob_new[i] = prob_new[i] / sum(prob_new[i])
            self.test_choices[i] = np.random.choice(range(self.n + 1), 1, p=prob_new[i])
            addtwodimdict(prob_dict2, tuple(self.test_choice_set[i][1:].tolist()), prob_new[i])

        for t in range(self.n_samples):
            for i in range(self.MAXIMUM_T):
                self.prob_train[t][i] = prob_dict[tuple(self.train_offered_set[t][i][1:].tolist())]
                self.train_choices[t][i] =  np.random.choice(range(self.n+1), 1, p=self.prob_train[t][i])

        return self.train_choices, self.prob_train,  self.test_choices, prob_new, self.train_offered_set

    def GetIniParam(self):
        return self.P_arrival, self.P_transition