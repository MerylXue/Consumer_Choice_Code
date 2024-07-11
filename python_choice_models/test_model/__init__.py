# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


import numpy as np
from python_choice_models.data_generate.Data_transaction_Multiple_sample import DataProbTransferJson


## number of samples given a ground truth model
K_sample = 10
## number of ground truth models
K_train = 10


# generate the data from mixing two models
def generate_mix_data(N_prod_list, T_assor_list, T_total_list, distr_set, ground_model, lb,ub):

    for N_prod in N_prod_list:
        for T_assor in T_assor_list:
            for T_total in T_total_list:
                for distr in distr_set:
                    for i in range(K_train):
                        # randomly generate two models from the lists of ground truth model
                        models = np.random.choice(ground_model, 2, replace=False,
                                                  p=[1 / len(ground_model) for i in
                                                     range(len(ground_model))])
                        ground = 'mix_%s_%s' % (models[0], models[1])
                        input_file_lst = DataProbTransferJson(ground, N_prod, T_assor, T_total, lb, ub,
                                                              distr, 0, i, K_sample)
                        ## output the name lists of the data files
                        file = 'data_generate/RawDataFileName/Model_mix-Products_%d--Assortments_%d-T_Data-%d-Train_%d|%d-Sample_%d.txt' % (N_prod, T_assor, T_total,i,K_train, K_sample)
                        with open(file, 'w') as output_file:
                            for name in input_file_lst:
                                outline = ['%s' % name]
                                output_file.write(','.join(outline) + '\n')

# generate the data from the ground truth model
def generate_singe_data(N_prod_list, T_assor_list, T_total_list, distr_set, ground_model, lb,ub):
    for N_prod in N_prod_list:
        for T_assor in T_assor_list:
            for T_total in T_total_list:
                for distr in distr_set:
                    for ground in ground_model:
                        for i in range(K_train):
                            input_file_lst = DataProbTransferJson(ground, N_prod, T_assor, T_total, lb, ub,
                                                                  distr, 0, i, K_sample)

                            ## output the name lists of the data files
                            file = 'data_generate/RawDataFileName/Model_%s-Products_%d--Assortments_%d-T_Data-%d-Train_%d|%d-Sample_%d.txt' % (
                            ground, N_prod, T_assor, T_total, i, K_train, K_sample)
                            with open(file, 'w') as output_file:
                                for name in input_file_lst:
                                    outline = ['%s' % name]
                                    output_file.write(','.join(outline) + '\n')