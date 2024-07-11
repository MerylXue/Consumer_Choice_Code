# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


from datetime import date
from python_choice_models.utils import update_error_by_dict_lst, statistics_error_samples
from python_choice_models.test_model import K_sample, K_train, beta, K_train_os
from python_choice_models.integrated_oda_estimate.RunAllOS import compare_all
from python_choice_models.integrated_oda_estimate.Estimators import oda_estimators
from python_choice_models.decision_forest.run_decision_forest import runDecisionForest
from python_choice_models.integrated_oda_estimate.RandomForest import run_with_rf


 ## test the binary choice forest in "The Use of Binary Choice Forests to Model and Estimate Discrete Choices"
#

def test_RF(N_prod, T_assor, T_total, model_name):
    avg_error_dict = dict()
    for i in range(K_train):
        input_file_lst = []
        FileName = open('data_generate/RawDataFileName/Model_%s-Products_%d--Assortments_%d-T_Data-%d-Train_%d|%d-Sample_%d.txt'
                                  % (model_name, N_prod, T_assor, T_total, i, 10, 10) )

        for line in FileName.readlines():
            line = line.rstrip("\n")
            input_file_lst.append(line)

        for t in range(K_sample):
            input_file = input_file_lst[t]
            # empirical_error_dict = run_with_empirical(input_file)

            # ## decision forest
            # df_error_dict = runDecisionForest(input_file)
            # avg_error_dict = update_error_by_dict_lst('forest', 'decision_forest',
            #                                           df_error_dict,
            #                                           avg_error_dict)
            # random forest
            rf_error_dict = run_with_rf(input_file, N_prod)
            avg_error_dict = update_error_by_dict_lst('bcf', 'classification',
                                                      rf_error_dict,
                                                      avg_error_dict)


            # # ##ccp true prob point
            # ccp_t_error_dict = runCCPInteriorIni('lst', 'ccp', 't',
            #                                      input_file)
            #
            # avg_error_dict = update_error_by_dict_lst('CCP', 'true_prob', ccp_t_error_dict,
            #                                           avg_error_dict)
            #


    statistics_dict = statistics_error_samples(avg_error_dict, K_train, K_sample)
        # for key in boundary_dict.keys():
        #     boundary_dict[key] = boundary_dict[key] / (K_train * K_sample)
        # print(boundary_dict)
    with open(
            'result/TestModel_RF-Date_%s-Products_%d-Model_%s-Assortments_%d--Sample_%d.csv'
            % (date.today().strftime("%Y_%m_%d"),  N_prod, model_name,  T_assor, T_total), 'w') as file:
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




# test decision forest method from "Decision Forest: A Nonparametric Approach to Modeling Irrational Choice, Chen and Misic 2020"

def test_DF(N_prod, T_assor, T_total, model_name):
    avg_error_dict = dict()
    for i in range(K_train):
        input_file_lst = []
        FileName = open('data_generate/RawDataFileName/Model_%s-Products_%d--Assortments_%d-T_Data-%d-Train_%d|%d-Sample_%d.txt'
                                  % (model_name, N_prod, T_assor, T_total, i, 10, 10) )

        for line in FileName.readlines():
            line = line.rstrip("\n")
            input_file_lst.append(line)

        for t in range(K_sample):
            input_file = input_file_lst[t]
            # empirical_error_dict = run_with_empirical(input_file)

            ## decision forest
            df_error_dict = runDecisionForest(input_file)
            avg_error_dict = update_error_by_dict_lst('forest', 'decision_forest',
                                                      df_error_dict,
                                                      avg_error_dict)


    statistics_dict = statistics_error_samples(avg_error_dict, K_train, K_sample)

    with open(
            'result/TestModel_DF-Date_%s-Products_%d-Model_%s-Assortments_%d--Sample_%d.csv'
            % (date.today().strftime("%Y_%m_%d"),  N_prod, model_name,  T_assor, T_total), 'w') as file:
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


## run all the structural choice models, see oda_estimators in /integrated_oda_estimate/Estimators.py
def test_Str(N_prod, T_assor, T_total, model_name):
    avg_error_dict = dict()
    detail_dict = dict()
    for i in range(K_train):
        input_file_lst = []
        FileName = open('data_generate/RawDataFileName/Model_%s-Products_%d--Assortments_%d-T_Data-%d-Train_%d|%d-Sample_%d.txt'
                                  % (model_name, N_prod, T_assor, T_total, i, 10, 10) )
        for line in FileName.readlines():
            line = line.rstrip("\n")
            input_file_lst.append(line)

        for t in range(K_sample):
            input_file = input_file_lst[t]

            # structural model
            avg_error_dict, detail_dict = compare_all(input_file,  oda_estimators,avg_error_dict,detail_dict)


    statistics_dict = statistics_error_samples(avg_error_dict, K_train, K_sample)
    ## output the average results
    with open( 'result/Details/TestModel_Str-Date_%s-Products_%d-Model_%s-Assortments_%d--Sample_%d.csv'
            % (date.today().strftime("%Y_%m_%d"),  N_prod, model_name,  T_assor, T_total), 'w'
              ) as write_file:
        key_model = next(iter(detail_dict))
        key_method = next(iter(detail_dict[key_model]))
        key_indices = next(iter(detail_dict[key_model][key_method]))
        out_line_col0 = ['Details']
        out_line_col0 += ['index']
        for key in detail_dict[key_model][key_method][key_indices]:
            out_line_col0 += [key]
        write_file.writelines(','.join(out_line_col0) + '\n')
        for key in detail_dict:
            for key2 in detail_dict[key]:
                for key_index in detail_dict[key][key2]:
                    out_line = [key + ': ' + key2]
                    out_line += ['%d' % key_index]
                    for key3, item in detail_dict[key][key2][key_index].items():
                        # print(key3, item)
                        out_line += ['%.4f' % item]
                    # print(out_line)
                    write_file.writelines(','.join(out_line) + '\n')

    ## output the results in each sample
    with open(
            'result/TestModel_Str-Date_%s-Products_%d-Model_%s-Assortments_%d--Sample_%d.csv'
            % (date.today().strftime("%Y_%m_%d"),  N_prod, model_name,  T_assor, T_total), 'w') as file:
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

            #
            # for key in boundary_dict.keys():
            #     out_line = [key]
            #     out_line += ['%.4f' % boundary_dict[key]]
            #     file.writelines(','.join(out_line) + '\n')
