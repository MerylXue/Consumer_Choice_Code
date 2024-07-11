# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


from python_choice_models.test_model import K_sample, K_train, beta, K_train_os

from python_choice_models.integrated_oda_estimate.Estimators import oda_estimators
from python_choice_models.integrated_oda_estimate.ODA_structual_model import Validate_MinMaxRegret_Model_alphadict,Validate_MinMaxRegret_Model_str
from datetime import date
from python_choice_models.utils import update_error_by_dict_lst, statistics_error_samples
import os
import sys
# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/python_choice_models/')

## this function is used for testing the ODA framework with only using structurel models (no interpolation with empirical results)

def test_oda_str( N_prod, T_assor, T_total, model_name):
    avg_error_dict = dict()
    for i in range(K_train):
        input_file_lst = []
        ## the file that records the name lists of data files
        FileName = open(
            'data_generate/RawDataFileName/Model_%s-Products_%d--Assortments_%d-T_Data-%d-Train_%d|%d-Sample_%d.txt'
            % (model_name, N_prod, T_assor, T_total, i, 10, 10))

        for line in FileName.readlines():
            line = line.rstrip("\n")
            input_file_lst.append(line)

        for t in range(K_sample):
            input_file = input_file_lst[t]

            oda_str = Validate_MinMaxRegret_Model_str(input_file, beta, K_train_os, oda_estimators)
            # print(bootstrap_error_dict)
            # avg_error_dict = update_error_by_dict('Bootstrap_m', 'structual', bootstrap_error_dict, avg_error_dict)
            avg_error_dict = update_error_by_dict_lst('Oda_str', 'min_max_regret',
                                                      oda_str, avg_error_dict)


    statistics_dict = statistics_error_samples(avg_error_dict, K_train, K_sample)

    with open(
            'result/TestModel_ODAStr-Data_%s-Products_%d-Model_%s-Assortments_%d--Sample_%d.csv'
            % (date.today().strftime("%Y_%m_%d"), N_prod, model_name, T_assor, T_total), 'w') as file:
        out_line_col0 = ['0']
        key0 = next(iter(statistics_dict))
        key01 = next(iter(statistics_dict[key0]))
        for key2 in statistics_dict[key0][key01]:
            out_line_col0 += [key2]

        for key4 in ['avg', 'all_var', 'avg_var', 'max', 'min']:
            out_line_col0[0] = key4
            file.writelines(','.join(out_line_col0) + '\n')
            for key in statistics_dict:
                for key2 in statistics_dict[key]:
                    out_line = [key + ': ' + key2]
                    for key3, item in statistics_dict[key][key2].items():
                        # print(key3, item)
                        if key4 in ['all_var', 'avg_var']:
                            out_line += ['%.3e' % item[key4]]
                        else:
                            out_line += ['%.4f' % item[key4]]

                    file.writelines(','.join(out_line) + '\n')
            file.writelines('\n')


## this function is used for testing the ODA framework  with interpolation

def test_oda( N_prod, T_assor, T_total, model_name):
    avg_error_dict = dict()
    for i in range(K_train):
        input_file_lst = []
        ## the file that records the name lists of data files
        FileName = open('data_generate/RawDataFileName/Model_%s-Products_%d--Assortments_%d-T_Data-%d-Train_%d|%d-Sample_%d.txt'
                                  % (model_name, N_prod, T_assor, T_total, i, 10, 10) )

        for line in FileName.readlines():
            line = line.rstrip("\n")
            input_file_lst.append(line)

        for t in range(K_sample):
            input_file = input_file_lst[t]


            # Min max chosen structual and optimal alpha
            oda_model_dict = Validate_MinMaxRegret_Model_alphadict(input_file, beta,
                                                                   K_train_os, oda_estimators)

            avg_error_dict = update_error_by_dict_lst('ODA', 'model_then_alpha',
                                                      oda_model_dict,
                                                      avg_error_dict)

    statistics_dict = statistics_error_samples(avg_error_dict, K_train, K_sample)

    with open(
            'result/TestModel_ODA-Date_%s-Products_%d-Model_%s-Assortments_%d--Sample_%d.csv'
            % (date.today().strftime("%Y_%m_%d"),  N_prod, model_name, T_assor, T_total), 'w') as file:
        out_line_col0 = ['0']
        key0 = next(iter(statistics_dict))
        key01 = next(iter(statistics_dict[key0]))
        for key2 in statistics_dict[key0][key01]:
            out_line_col0 += [key2]

        for key4 in ['avg', 'all_var', 'avg_var', 'max', 'min']:
            out_line_col0[0] = key4
            file.writelines(','.join(out_line_col0) + '\n')
            for key in statistics_dict:
                for key2 in statistics_dict[key]:
                    out_line = [key + ': ' + key2]
                    for key3, item in statistics_dict[key][key2].items():
                        # print(key3, item)
                        if key4 in ['all_var', 'avg_var']:
                            out_line += ['%.3e' % item[key4]]
                        else:
                            out_line += ['%.4f' % item[key4]]

                    file.writelines(','.join(out_line) + '\n')
            file.writelines('\n')

