# This code is from the paper:
# Qi Feng, J. George Shanthikumar, Mengying Xue, “Consumer choice models and estimation: A review and extension”, Production and Operations Management, 2022, 31(2): 847-867.

# Electronic companion of the paper:"Consumer Choice Models and Estimation: A Review and Extensions" by
#
# - Qi Feng: Mitchell E.Daniels, Jr. School of Management, Purdue University, [annabellefeng@purdue.edu](mailto:annabellefeng@purdue.edu)
# - J.George Shanthikumar: Mitchell E.Daniels, Jr. School of Management, Purdue University [shanthikumar@purdue.edu](mailto:shanthikumar@purdue.edu)
# - Mengying Xue: International Institute of Finance, School of Management, University of Science and Technology of China, [mengying.xue@gmail.com

# This code was written in Python and tested on Mac OSX and Linux  operating system.

import os
## set the working directory
print("Current working directory: {0}".format(os.getcwd()))
os.chdir('python_choice_models/')
print("Current working directory: {0}".format(os.getcwd()))


from python_choice_models.test_model import generate_singe_data, generate_mix_data

## set the number of products
N_prod_list = [8]
## the ground truth model, you can set the name
ground_model = ["rl"]
## total number of samples
N_sample_list = [100]
## number of choice sets in the data
T_assor_list = [10]
## if distr = even, the data samples are distributed evenly into each choice set
distr_set = ['even']
# if the ground truthe model follows the random utiltiy model, then lb and ub are the lower and upper bounds of the radomly generated utitliy values
lb = 0.0
ub = 1.0

## generate data from the single ground truth model
## if you want to generate data from a mixture of structural models, see function generate_mix_data in test_model.__init__.py
generate_singe_data(N_prod_list, T_assor_list, N_sample_list, distr_set, ground_model, lb, ub)
