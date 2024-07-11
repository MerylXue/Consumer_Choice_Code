# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.

from collections import Counter
import numpy as np

## function for finding the optimal solution of the min-max regret optimization in ODA framework
def Min_Max_Regret_OS( K_sample, max_regret_error):

    min_max_regret_os = np.zeros(K_sample)
    for k in range(K_sample):
        min_max_regret = min(max_regret_error[k])
        min_max_regret_os[k] = list(max_regret_error[k]).index(min_max_regret)


    d2 = Counter(min_max_regret_os)
    sorted_x = sorted(d2.items(), key=lambda x: x[1], reverse=True)
    optimal_os_index = int(sorted_x[0][0])


    return optimal_os_index