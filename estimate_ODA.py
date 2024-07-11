# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.

# Electronic companion of the paper:"Consumer Choice Models and Estimation: A Review and Extensions" by
#
# - Qi Feng: Mitchell E.Daniels, Jr. School of Management, Purdue University, [annabellefeng@purdue.edu](mailto:annabellefeng@purdue.edu)
# - J.George Shanthikumar: Mitchell E.Daniels, Jr. School of Management, Purdue University [shanthikumar@purdue.edu](mailto:shanthikumar@purdue.edu)
# - Mengying Xue: International Institute of Finance, School of Management, University of Science and Technology of China, [mengying.xue@gmail.com

# This code was written in Python and tested on Mac OSX and Linux  operating system.

## set the working directory
import os
print("Current working directory: {0}".format(os.getcwd()))
os.chdir('python_choice_models/')
print("Current working directory: {0}".format(os.getcwd()))


import sys
from python_choice_models.test_model.test_oda import test_oda, test_oda_str, test_bootstrap


def main():
    #parameter setting
    N_prod_list = [8]
    T_assor_list = [10]
    T_total_list = [100]


    ground_model = ['rl' ]




    for N_prod in N_prod_list:
        for T_assor in T_assor_list:
            for T_total in T_total_list:

                for model in ground_model:
                    test_oda(N_prod, T_assor, T_total,  model)

main()

