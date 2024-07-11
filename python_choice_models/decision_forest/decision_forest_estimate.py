## This code is from the paper
## Chen and Misic (2021), Decision Forest: A Nonparametric Approach to Modeling Irrational Choice
## https://ssrn.com/abstract=3376273
## The python code is written based on their code in Julia



from python_choice_models.estimation import Estimator
from copy import deepcopy
from gurobipy import *
import time
import numpy as np
from python_choice_models.decision_forest import DecisionTree, DecisionForestModel
from python_choice_models.models import RankedListModel
from python_choice_models.estimation.ranked_list import RankedListEstimator
from python_choice_models.estimation.market_explore.ranked_list import MIPMarketExplorer
from python_choice_models.estimation.maximum_likelihood.ranked_list import RankedListMaximumLikelihoodEstimator
from python_choice_models.integrated_oda_estimate.Bootstrap import GenerateOutofSampleTransactions
time_limit = 1200

perTol = 1e-6
K_train = 5

class DecisionForestEMEstimator(Estimator):
    def estimate(self, forest_model, transactions):
        ## use bootstrap to choose the depth

        depth_list = [3,  5, 7]
        # n_product= len(forest_model.products)
        #use rank list models to do the warm start
        initial_model = RankedListModel.simple_deterministic_independent(forest_model.products)
        forest_model.sample_assortments(transactions)

        result = RankedListMaximumLikelihoodEstimator.with_this(MIPMarketExplorer()).estimate_with_market_discovery(initial_model, transactions)

        n_iter = 0

        forest_model.convertRankingToForest(result.ranked_lists)


        # print(forest_model.num_tree)


        mrmse_lst = [[] for k in range(len(depth_list))]
        models = []
        empirical_dict = forest_model.GenerateEmpricalDict(transactions)

        for k in range(K_train):
            validating_transactions, validating_transactions_prob = forest_model.GenerateOutofSampleTransactions(transactions, empirical_dict)
            # print(validating_transactions_prob)
            for depth_limit in depth_list:
                forest_model_tmp = deepcopy(forest_model)
                transaction_counts = self.transactionstoCount(forest_model_tmp, validating_transactions)
                forest_model_tmp, loglik, elapsed_time, n_iter = self.estimateForest(forest_model_tmp, transaction_counts, time_limit, perTol, depth_limit)
                test_forest_predict, test_forest_predict_by_class = forest_model_tmp.predictForest()
                forest_model_tmp.probabilities = test_forest_predict
                mrmse_out = forest_model_tmp.mrmse_known_prob(validating_transactions_prob)
                mrmse_lst[depth_list.index(depth_limit)].append(mrmse_out)


        mrmse_avg = [sum(mrmse_lst[k])/K_train for k in range(len(depth_list))]

        opt_depth = depth_list[mrmse_avg.index(min(mrmse_avg))]

        ##for test
        # opt_depth = 3
        transaction_counts = self.transactionstoCount(forest_model, transactions)
        forest_model, loglik, elapsed_time, n_iter = self.estimateForest(forest_model, transaction_counts, time_limit,
                                                                     perTol, opt_depth)
        test_forest_predict, test_forest_predict_by_class = forest_model.predictForest()
        forest_model.probabilities = test_forest_predict
        print(mrmse_avg)
        print("the optimal depth is %d" %opt_depth)
        # print(forest_model.probabilities)
        return forest_model, n_iter
        # return result, n_iter



    def transactionstoCount(self, model, transactions):
        T = len(transactions)
        num_assort = len(model.assort_set)
        num_product = len(model.products)
        transaction_counts = np.zeros((num_assort, num_product))
        for transaction in transactions:
            assort_index = model.assortment_index(transaction)
            for product in transaction.offered_products:
                transaction_counts[assort_index][product] += 1
        return transaction_counts

    # function tcm_estimateForest in tcm_estimateForest.jl
    def estimateForest(self, initial_forest, transaction_counts, time_limit_overall, perTol, depth_limit):
        time_limit_EM = 60

        start_time = time.time()
        forest = deepcopy(initial_forest)


        forest.tree_distribution = np.array([1/ forest.num_tree for i in range( forest.num_tree)])

        forest, v, loglik, elapsed_time = self.forest_EM(forest, transaction_counts, time_limit_EM)

        u = np.array([[transaction_counts[m][i]/v[m][i] for i in range(len(forest.products))]
                      for m in range(len(forest.assort_set))])
        u[np.isnan(u)] = 0
        u[np.isinf(u)] = 0

        iter = 0
        print("CG iter %d ----- -- time elapsed = %f, log likelihood = %f" %(iter, 0.0, loglik))
        while (time.time() - start_time < time_limit_overall):
            iter += 1
            u = np.array([[transaction_counts[m][i] / v[m][i] for i in range(len(forest.products))]
                          for m in range(len(forest.assort_set))])
            u[np.isnan(u)] = 0
            u[np.isinf(u)] = 0
            ## solve tree subproblem
            single_tree, objval = self.solveTreeSubProblem(forest, u, depth_limit)
            # print(forest.num_tree, len(forest.tree_distribution))
            ## update the forest
            forest.trees.append(single_tree)
            forest.num_tree += 1

            # num_tree += 1

            initial_fudge = 0.999
            # update the probability distribution for forests
            forest.tree_distribution = np.array(list(initial_fudge * forest.tree_distribution)  + [1-initial_fudge])

            # print("len of trees %d, len of tree_distribution %d" % (forest.num_tree, len(forest.tree_distribution)))
            forest, v, new_loglik, elapsed_time = self.forest_EM(forest, transaction_counts, time_limit_EM)

            print("CG iter %d ----- -- time elapsed = %f, log likelihood = %f" % (iter, time.time() - start_time, new_loglik))
            if ((new_loglik - loglik) / abs(loglik) < perTol):
                temp = (new_loglik - loglik) / abs(loglik)
                # print(temp)
                # print(new_loglik)
                # print(loglik)
                loglik = new_loglik
                break

            loglik = new_loglik
        end_time = time.time()

        elapsed_time = end_time - start_time

        return forest, loglik, elapsed_time, iter
    ## function tcm_forest_M_update_lambda in tcm_estimateForest.jl
    # update choice probabilities
    def forest_M_update_lambda(self, num_trees, h, transaction_counts):
        sum_of_h = np.zeros(num_trees)

        for k in range(num_trees):
            sum_of_h[k] = np.sum(h[k] * transaction_counts)
        lambda_ = sum_of_h/sum(sum_of_h)
        return lambda_

    ## function tcm_forest_E in tcm_estimateForest.jl
    ## input: DecisionForestModel: forest, lambda_: probability_distribution

    def forest_E(self, forest):
        ## v: n_assortments * (1+num_product)
        ## v_by_class: num_tree * n_assortments * (1+num_product)

        ## update the probabilities of customer types
        v, v_by_class = forest.predictForest()

        # h: num_tree * n_assortments * (1+num_product)
        h = []

        for k in range(forest.num_tree):
            single_h = np.array([[forest.tree_distribution[k] * v_by_class[k][m][i]/v[m][i] for i in range(len(forest.products))]
            for m in range(len(forest.assort_set))])

            single_h[ np.isnan(single_h)] = 0.0
            h.append(single_h)
        h = np.array(h)
        return h, v, v_by_class

    def forest_EM(self, forest, transaction_counts, time_limit_EM):

        # forest.tree_distribution = deepcopy(initial_lambda) #1/K * ones(K)
        v, v_by_class = forest.predictForest()
        logv = np.log(v)
        logv[np.isnan(logv)] = 0.0
        logv[np.isinf(logv)] = 0.0

        loglik = np.sum(transaction_counts * logv)

        v = np.zeros(np.shape(transaction_counts))

        perTol_EM = 1e-5
        iter = 0

        start_time = time.time()

        while (time.time() - start_time < time_limit_EM):
            iter += 1
            h, v, v_by_class = self.forest_E(forest)
            forest.tree_distribution = self.forest_M_update_lambda(forest.num_tree, h, transaction_counts)

            v, v_by_class = forest.predictForest()
            logv = np.log(v)
            logv[np.isinf(logv)] = 0.0
            logv[np.isnan(logv)] = 0.0
            new_loglik = float(np.sum(transaction_counts * logv))

            print("\t Ranking EM: iter %d, -- time elpased = %f, log likelihood = %f"
                  % (iter, time.time() - start_time, new_loglik))

            if (new_loglik - loglik) / abs(loglik) < perTol_EM:
                temp = (new_loglik - loglik) / abs(loglik)
                loglik = new_loglik
                break

            loglik = new_loglik
        elapsed_time = time.time() - start_time

        return forest, v, loglik, elapsed_time

    ## function tcm_solverTreeSubproblem in tcm_estimateForest.jl
    def solveTreeSubProblem(self, forest, u, depth_limit):
        # tol_RC = 1e-4
        M = len(forest.assort_set)
        N = len(forest.products) - 1
        #initialize the tree with no purchase option as the root
        single_tree = DecisionTree([-1],[-1],[0],[True])
        single_depth = [0]
        single_include_set = [[0 for i in range(N)]]
        # each list in this one has dimension of M
        single_assortment_binary = [[1 for i in range(M)]]
        single_reduced_cost = [0.0]
        single_isClosed = [False]
        iter = 0

        while (True):
            iter += 1
            current_reduced_cost = sum([single_reduced_cost[i] for i in range(len(single_reduced_cost)) if single_tree.isLeaf[i]])
            print("\t\t Subproblem Iter : %d  -- reduced cost: %f  -- num leaves: %d"%(iter,current_reduced_cost, sum(single_tree.isLeaf)))

            candidate_leaves =  [ d for d in range(len(single_tree.isLeaf)) if single_tree.isLeaf[d] and
                                  (single_depth[d] < depth_limit) and not single_isClosed[d]]

            if len(candidate_leaves) == 0:
                print("\t\t\t No more eligible leaves to split; exiting procedure...")
                break

            num_candL = len(candidate_leaves)
            candidate_reduced_costs = -np.inf* np.ones(num_candL)
            best_split_p_by_leaf = [0 for i in range(num_candL)]
            best_left_leaf_p_by_leaf  = [0 for i in range(num_candL)]
            best_right_leaf_p_by_leaf  =  [0 for i in range(num_candL)]
            best_left_reduced_cost_by_leaf  = [0 for i in range(num_candL)]
            best_right_reduced_cost_by_leaf  =  [0 for i in range(num_candL)]
            best_left_assortment_binary_by_leaf = [ np.array([]) for i in range(num_candL)]
            best_right_assortment_binary_by_leaf =[  np.array([]) for i in range(num_candL)]

            for ell_ind in range(num_candL):
                leaf = candidate_leaves[ell_ind]
                assortment_binary = deepcopy(single_assortment_binary[leaf])
                candidate_split_products = list(set(range(1, N)).difference(set(single_include_set[leaf])))

                for split_p in candidate_split_products:
                    candidate_left_leaf_products = set(single_include_set[leaf]).union({split_p})
                    best_left_reduced_cost = -np.inf
                    best_left_p = -1
                    best_left_assortment_binary = np.zeros(M)
                    for left_p in candidate_left_leaf_products:
                        left_assortment_binary = np.array([assortment_binary[i] * forest.assort_set[i][split_p - 1] for i in range(M)])
                        temp = np.dot(left_assortment_binary, u[:,left_p])
                        if temp > best_left_reduced_cost:
                            best_left_reduced_cost = temp
                            best_left_p = left_p
                            best_left_assortment_binary = deepcopy(left_assortment_binary)

                    candidate_right_leaf_products = deepcopy(single_include_set[leaf])
                    best_right_reduced_cost = -np.inf
                    best_right_p = -1
                    best_right_assortment_binary = np.zeros(M)
                    for right_p in candidate_right_leaf_products:
                        right_assortment_binary = np.array([assortment_binary[i] * (1 - forest.assort_set[i][split_p - 1]) for i in range(M)])
                        temp = np.dot(right_assortment_binary, u[:, right_p])
                        if temp > best_right_reduced_cost:
                            best_right_reduced_cost = temp
                            best_right_p = right_p
                            best_right_assortment_binary = deepcopy(right_assortment_binary)
                    new_reduced_cost = best_left_reduced_cost + best_right_reduced_cost


                    if new_reduced_cost > candidate_reduced_costs[ell_ind]:
                        candidate_reduced_costs[ell_ind] = new_reduced_cost
                        best_split_p_by_leaf[ell_ind] = split_p
                        best_left_leaf_p_by_leaf[ell_ind] = best_left_p
                        best_right_leaf_p_by_leaf[ell_ind] = best_right_p
                        best_left_reduced_cost_by_leaf[ell_ind] = best_left_reduced_cost
                        best_right_reduced_cost_by_leaf[ell_ind] = best_right_reduced_cost
                        best_left_assortment_binary_by_leaf[ell_ind] = best_left_assortment_binary
                        best_right_assortment_binary_by_leaf[ell_ind] = best_right_assortment_binary

            incumbent_reduced_costs = [single_reduced_cost[i] for i in candidate_leaves]
            tol_RC = 1e-4

            for ell_ind in range(num_candL):
                if incumbent_reduced_costs[ell_ind] + tol_RC < candidate_reduced_costs[ell_ind]:
                    leaf = int(candidate_leaves[ell_ind])

                    numNodes = len(single_tree.tree_left)
                    #update the tree
                    single_tree.tree_left += [-1, -1]
                    single_tree.tree_right += [-1, -1]
                    single_tree.isLeaf += [True, True]
                    single_tree.product += [best_left_leaf_p_by_leaf[ell_ind], best_right_leaf_p_by_leaf[ell_ind]]
                    single_depth += [single_depth[leaf] + 1, single_depth[leaf] + 1]
                    single_include_set += [list(set(single_include_set[leaf]).union({best_split_p_by_leaf[ell_ind]})),
                                           deepcopy(single_include_set[leaf])]
                    single_assortment_binary += [deepcopy(best_left_assortment_binary_by_leaf[ell_ind]),
                                                 deepcopy(best_right_assortment_binary_by_leaf[ell_ind])]
                    single_reduced_cost += [best_left_reduced_cost_by_leaf[ell_ind], best_right_reduced_cost_by_leaf[ell_ind]]
                    single_isClosed += [False, False]

                    #update the leaf
                    single_tree.tree_left[leaf] = numNodes + 1
                    single_tree.tree_right[leaf] = numNodes + 2
                    single_tree.product[leaf] = best_split_p_by_leaf[ell_ind]
                    single_isClosed[leaf] = True
                    single_tree.isLeaf[leaf] = False
                else:
                    leaf = candidate_leaves[ell_ind]
                    single_isClosed[leaf] = True

        optimal_reduced_cost = sum([single_reduced_cost[i] for i in range(len(single_reduced_cost)) if single_tree.isLeaf[i]])
        return single_tree, optimal_reduced_cost