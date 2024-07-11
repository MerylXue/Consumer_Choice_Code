# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.


import numpy as np
def Select_Partial_Data(train_choice, prob_train, train_offered_set,chosen_assort,Data_set_size, num_product):
    n_samples = len(train_choice)
    # print(chosen_assort)
    MAXIMUM_T = int(sum([Data_set_size[i] for i in chosen_assort]))
    new_train_choice = np.zeros((n_samples, MAXIMUM_T), int)
    new_prob_train = np.zeros((n_samples, MAXIMUM_T, num_product + 1))
    new_train_offered_set = np.zeros((n_samples, MAXIMUM_T, num_product + 1), int)
    T_assort = len(chosen_assort)

    for t in range(n_samples):
        for chosen_assort_index in range(T_assort):
            chosen_assort_num = chosen_assort[chosen_assort_index]
            pre_index_benchmark = int(sum(Data_set_size[0:chosen_assort_num]))
            pro_index_benchmark = int(sum([Data_set_size[j] for j in chosen_assort[0:chosen_assort_index]]))

            new_train_offered_set[t][pro_index_benchmark:pro_index_benchmark + int(Data_set_size[chosen_assort_num])] \
                = train_offered_set[t][pre_index_benchmark:pre_index_benchmark + int(Data_set_size[chosen_assort_num])]
            new_train_choice[t][pro_index_benchmark:pro_index_benchmark + int(Data_set_size[chosen_assort_num])] \
                = train_choice[t][pre_index_benchmark:pre_index_benchmark + int(Data_set_size[chosen_assort_num])]
            new_prob_train[t][pro_index_benchmark:pro_index_benchmark + int(Data_set_size[chosen_assort_num])] \
                = prob_train[t][pre_index_benchmark:pre_index_benchmark + int(Data_set_size[chosen_assort_num])]

    return new_train_choice, new_prob_train, new_train_offered_set