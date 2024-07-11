# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.

import random
import numpy as np
import os
import sys
import json
from python_choice_models.data_generate.MNLDataGenerate import MNL_bench
from python_choice_models.data_generate.ECDataGenerate import EC_bench
from python_choice_models.data_generate.HECDataGenerate import HEC_bench
from python_choice_models.data_generate.LCMNLDataGenerate import LCMNL_bench
from python_choice_models.data_generate.MCDataGenerate import MC_bench
from python_choice_models.data_generate.NestedDataGenerate import Nested_bench
from python_choice_models.data_generate.RankDataGenerate import rank_bench
from python_choice_models.data_generate.Half_IIA import Half_IIA_bench
from python_choice_models.data_generate.Mixture_EC_MKV import MIX_EC_MKV
from python_choice_models.data_generate.mixture_choice_models import Mixture_Choice_Models
import itertools
from python_choice_models.utils import UserEncoder
from python_choice_models.data_generate.select_data_from_full_assortment_data import Select_Partial_Data

## generate data when assigning different number of data samples to each choice sets
def DataProbTransferJsonMultiAssortments(Type, N_prod, T_assor_list, N_sample_each_assort, lb, ub, DataDistribute, perturb, index_model, n_samples):
    Num_assort = 2 ** N_prod
    Data_set_size = np.ones(Num_assort, int)
    Data_set_size = Data_set_size * int(N_sample_each_assort)

    #first generate all the data for all assortments
    lst = list(itertools.product([0, 1], repeat=N_prod))

    T_test = len(lst)

    # print(lst)
    new_set_binary_limit = np.asarray(lst)  # testing assortment
    # new_set_binary_limit = np.random.randint(2,size = (T_test, N_prod))
    new_set_binary_expand = np.zeros((T_test, N_prod + 1), int)

    for j in range(T_test):
        new_set_binary_expand[j][0] = 1
        new_set_binary_expand[j][1:] = new_set_binary_limit[j][0:]

    if Type == 'mnl':
        MNL_model = MNL_bench(N_prod, Num_assort, Data_set_size, lb, ub, n_samples)
        if perturb == 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = MNL_model.GenerateData()
            # print(train_offered_set[240])
        elif perturb > 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = MNL_model.ProbPerturb(perturb)
        ini_utility = MNL_model.GetIniParam()


    elif Type == 'exp':
        EXP_model = EC_bench(N_prod, Num_assort, Data_set_size, lb, ub,n_samples)
        if perturb == 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = EXP_model.GenerateData()
        elif perturb > 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = EXP_model.ProbPerturb(perturb)
        ini_utility = EXP_model.GetIniParam()

    elif Type == 'hec':
        HEC_model = HEC_bench(N_prod, Num_assort, Data_set_size, lb, ub, n_samples)
        if perturb == 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = HEC_model.GenerateData()
        elif perturb > 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = HEC_model.ProbPerturb(perturb)
        ini_utility = HEC_model.GetIniParam()
    elif Type == 'lc':
        # K_MNL = 5
        LC_model = LCMNL_bench(N_prod, Num_assort, Data_set_size, lb, ub, n_samples)
        if perturb == 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = LC_model.GenerateData()
        elif perturb > 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = LC_model.ProbPerturb(perturb)
        ini_utility, ini_lambda = LC_model.GetIniParam()

    elif Type == 'mkv':

        MC_model = MC_bench(N_prod, Num_assort, Data_set_size, lb, ub, n_samples)
        MC_model.Check_MC_Data()
        if perturb == 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set= MC_model.GenerateData()
        elif perturb > 0:
            # train_choice, prob_train, test_choice, prob_test, train_offered_set = MC_model.ProbPerturb(perturb)
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = MC_model.Prob_perturb_new(perturb)

        ini_lambda, ini_rho = MC_model.GetIniParam()

    elif Type == 'rl':


        Rank_model = rank_bench(N_prod, Num_assort, Data_set_size, lb, ub, n_samples)
        if perturb == 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = Rank_model.GenerateData()
        elif perturb > 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = Rank_model.ProbPerturb(perturb)

        ini_perm, ini_beta = Rank_model.GetIniParam()

    elif Type == 'nl':
        # K_Nest = 3
        Nest_Model = Nested_bench(N_prod, Num_assort, Data_set_size, lb, ub, n_samples)
        if perturb == 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = Nest_Model.GenerateData()
        elif perturb > 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = Nest_Model.ProbPerturb(perturb)
        ini_utility, ini_lambda = Nest_Model.GetIniParam()

    elif Type == 'emnl':
        Half_IIA_model = Half_IIA_bench(N_prod, Num_assort, Data_set_size, lb, ub, n_samples)
        if perturb == 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = Half_IIA_model.GenerateData()
            # print(train_offered_set[240])
        elif perturb > 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = Half_IIA_model.ProbPerturb(perturb)
        ini_utility, ini_u0  = Half_IIA_model.GetIniParam()

    # elif Type == 'gc':
    #     GC_model = GC_bench(N_prod, T_assor, Data_set_size, lb, ub, n_samples)
    #     ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = GC_model.GenerateDataPW()

    # elif Type == 'random':
    #     ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = MNL_bench(
    #         K_train, Data_set_size, N_prod).ProbPerturb(1)

    elif Type.find('mix') >= 0:
        string_list = Type.split('_')
        model1 = string_list[1]
        model2 = string_list[2]
        Mix_choice_models = Mixture_Choice_Models(N_prod, Num_assort, Data_set_size, lb, ub, n_samples, model1, model2)
        if perturb == 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = Mix_choice_models.GenerateData()
            # print(train_offered_set[240])
        elif perturb > 0:
            ini_train_choice, ini_prob_train, test_choice, prob_test, ini_train_offered_set = Mix_choice_models.ProbPerturb(perturb)
        ini_utility = Mix_choice_models.GetIniParam()
        # Mix_exp_mkv_model = MIX_EC_MKV(N_prod, T_assor, Data_set_size, lb, ub)
        # if perturb == 0:
        #     train_choice, prob_train, test_choice, prob_test, train_offered_set = Mix_exp_mkv_model.GenerateData()
        #     # print(train_offered_set[240])
        # elif perturb > 0:
        #     train_choice, prob_train, test_choice, prob_test, train_offered_set = Mix_exp_mkv_model.ProbPerturb(perturb)
        # ini_lambda, ini_rho, ini_utility = Mix_exp_mkv_model.GetIniParam()
    else:
        print("Input Type is mistaken！！")

    filename_lst_assort = [[] for n in T_assor_list]
    #choose assortments
    chosen_assort = [[] for i in range(len(T_assor_list))]
    T_assort_list = sorted(T_assor_list)
    for idx in range(len(T_assort_list) - 1, -1, -1):
        T_assort = T_assort_list[idx]
        T_total = int(sum(Data_set_size[l] for l in chosen_assort[idx]))
        if not idx == len(T_assort_list) - 1:
            chosen_assort[idx] = sorted(random.sample(chosen_assort[idx + 1], T_assort_list[idx]))
        else:
            chosen_assort[idx] = sorted(random.sample(range(Num_assort), T_assort_list[idx]))

        train_choice, prob_train, train_offered_set = Select_Partial_Data(ini_train_choice, ini_prob_train,
                                                                                      ini_train_offered_set, chosen_assort[idx],
                                                                                      Data_set_size, N_prod)

        # filename_lst=[]
        for t in range(n_samples):
            Sumdict = {}
            Sumdict['amount_products'] = N_prod + 1

            result = []
            for j in range(T_assort):
                for k in range(Data_set_size[j]):
                    offered_products = []
                    tmp_dict = {}
                    index = sum(Data_set_size[0:j]) + k
                    for m in range(N_prod + 1):
                        if train_offered_set[t][index][m] == 1:
                            offered_products.append(int(m))
                    tmp_dict['product'] = int(train_choice[t][index])
                    tmp_dict['offered_products'] = offered_products
                    result.append(tmp_dict)

            result2 = []
            for j in range(T_assort):
                for k in range(N_sample_each_assort):
                    offered_products = []
                    tmp_dict = {}
                    offered_prob = []
                    index = int(N_sample_each_assort*j + k)
                    for m in range(N_prod + 1):
                        if train_offered_set[t][index][m] == 1:
                            offered_prob.append(prob_train[t][index][m])
                            offered_products.append(int(m))
                    tmp_dict['prob'] = offered_prob
                    tmp_dict['offered_products'] = offered_products
                    result2.append(tmp_dict)

            # print(len(result))
            test = []
            for i in range(T_test):
                offered_products = []
                tmp_dict = {}
                for m in range(N_prod + 1):
                    if new_set_binary_expand[i][m] == 1:
                        offered_products.append(int(m))
                tmp_dict['product'] = int(test_choice[i])
                tmp_dict['offered_products'] = offered_products
                test.append(tmp_dict)

            test2 = []
            for i in range(T_test):
                offered_products = []
                offered_prob = []
                tmp_dict = {}
                for m in range(N_prod + 1):
                    if new_set_binary_expand[i][m] == 1:
                        offered_prob.append(prob_test[i][m])
                        offered_products.append(int(m))
                tmp_dict['prob'] = offered_prob
                tmp_dict['offered_products'] = offered_products
                test2.append(tmp_dict)

            test_out = []
            # get the assortment out of the train set
            for i in range(T_test):
                offered_products = []
                offered_prob = []
                tmp_dict = {}
                if not list(new_set_binary_expand[i]) in train_offered_set[t].tolist():
                    for m in range(N_prod + 1):
                        if new_set_binary_expand[i][m] == 1:
                            offered_prob.append(prob_test[i][m])
                            offered_products.append(int(m))
                    tmp_dict['prob'] = offered_prob
                    tmp_dict['offered_products'] = offered_products
                    test_out.append(tmp_dict)



            transaction = {}

            ini_param = {}
            if Type == 'mnl' or Type == 'exp':
                ini_param['ini_utility'] = ini_utility.tolist()

            elif Type == 'lc':
                ini_param['ini_utility'] = ini_utility.tolist()
                ini_param['ini_lambda'] = ini_lambda.tolist()

            elif Type == 'mkv':
                ini_param['ini_lambda'] = ini_lambda.tolist()
                ini_param['ini_rho'] = ini_rho.tolist()

            elif Type == 'rl':
                ini_param['ini_perm'] = ini_perm.tolist()
                # ini_param['ini_perm'] = ini_perm
                ini_param['ini_beta'] = ini_beta.tolist()

            elif Type == 'nl':
                ini_param['ini_utility'] = ini_utility.tolist()
                ini_param['ini_lambda'] = ini_lambda.tolist()

            elif Type == 'emnl':
                ini_param['ini_utility'] = ini_utility.tolist()
                ini_param['ini_u0'] = ini_u0.tolist()
            elif Type.find('mix') >= 0:
                # ini_param['ini_lambda'] = ini_lambda.tolist()
                # ini_param['ini_rho'] = ini_rho.tolist()
                ini_param['ini_utility'] = ini_utility.tolist()

            # print(result2)
            transaction['in_sample'] = result
            transaction['in_sample_prob'] = result2
            transaction['all_sample'] = test
            transaction['all_sample_prob'] = test2
            transaction['out_of_sample_prob'] = test_out
            # print(len(test), len(test_out))

            Sumdict['Ground'] = Type
            Sumdict['ini_param'] = ini_param
            Sumdict['transactions'] = transaction


            js = json.dumps(Sumdict, cls=UserEncoder, indent=4)
            # print(js)
            filename = 'data_generate/instance/%s_%d_%d_%d_%d_%d_%.2f_%s_model%d_sample%d.dt' % (
            Type, N_prod, T_assort, T_assort*N_sample_each_assort, lb, ub, perturb, DataDistribute, index_model, t)
            # print(filename)
            with open(filename, 'w') as file_obj:
                file_obj.write(js)
            filename_lst_assort[idx].append(filename)
    return filename_lst_assort


