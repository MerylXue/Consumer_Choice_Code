## This code is from the paper
## Chen and Misic (2021), Decision Forest: A Nonparametric Approach to Modeling Irrational Choice
## https://ssrn.com/abstract=3376273
## The python code is written based on their code in Julia

from python_choice_models.models import Model
from copy import deepcopy
import numpy as np

class DecisionTree(object):
    # four arrays to construct a tree
    # the length of all four arrays = number of nodes in the tree (including the splits and leaf)
    def __init__(self, initial_left, initial_right, initial_product, initial_isLeaf):
        # tree-left[s] stores the left child node of  node[s]
        self.tree_left = initial_left
        self.tree_right = initial_right
        # product[s]: the correpsonding product of node s
        self.product = initial_product
        #whether node s is a leaf node
        self.isLeaf = initial_isLeaf

    ## function tcm_randomCmopleteTree in tcm_randomCompleteTree.jl
    def randomCompleteTree(self, N, depth):
        self.tree_left = []
        self.tree_right = []
        self.product = []
        self.isLeaf = []

        tree_unused_products = [ [] for k in range(2 ** (depth + 1) - 1)]
        tree_include_set = [[] for k in range(2 ** (depth + 1) - 1)]

        tree_unused_products[0] = np.array(range(1, N))
        tree_include_set[0] = [N]

        counter = 0
        num_nodes = 0

        for d in range(depth):
            num_d_nodes = 2**d
            for i in range(num_d_nodes):
                self.tree_left.append(counter + 1)
                self.tree_right.append(counter + 2)

                temp = np.random.choice(tree_unused_products[num_nodes])
                self.product.append(temp)
                self.isLeaf.append(False)

                tree_unused_products[counter + 1] = list(set(tree_unused_products[num_nodes]).difference(set([temp])))
                tree_unused_products[counter + 2] = list(set(tree_unused_products[num_nodes]).difference(set([temp])))

                tree_include_set[counter + 1] = tree_include_set[num_nodes] + [temp]
                tree_include_set[counter + 2] = deepcopy(tree_include_set[num_nodes])

                counter += 2
                num_nodes += 1

        num_d_nodes = 2 ** depth
        for i in range(num_d_nodes):
            self.tree_left.append(-1)
            self.tree_right.append(-1)
            self.product.append(np.random.choice(tree_include_set[num_nodes]))
            self.isLeaf.append(True)
            counter += 2
        return self.tree_left, self.tree_right, self.product, self.isLeaf


class DecisionForestModel(Model):
    def __init__(self, products):
        # trees: list of DecisionTree objects
        super().__init__(products)
        self.trees = []
        self.num_tree = 0
        # the probability distribution of trees, numpy array
        self.tree_distribution = []


    # function tcm_leafStats in tcm_code/tcm_leafStats.jl
    def leafStats(self):
        # record the number of leaves in each tree
        num_leaves_by_tree = np.array([sum(self.trees[k].isLeaf) for k in range(self.num_tree)])
        simple_avg_leaves = np.mean(num_leaves_by_tree)
        weighted_avg_leaves = np.dot(self.tree_distribution, num_leaves_by_tree)
        max_leaves = np.max(num_leaves_by_tree)

        return simple_avg_leaves, weighted_avg_leaves, max_leaves

    # function tcm_depthStats in tcm_code/tcm_leafStats.jl
    def depthStats(self):
        depth_by_tree = np.zeros(self.num_tree)
        for k in range(self.num_tree):
            depth_vec = np.zeros(len(self.trees[k].tree_left))
            depth_vec[0] = 1

            for cn in range(len(depth_vec)):
                if not self.trees[k].isLeaf[cn]:
                    depth_vec[self.trees[k].tree_left[cn]] = depth_vec[cn] + 1
                    depth_vec[self.trees[k].tree_right[cn]] = depth_vec[cn] + 1

            depth_by_tree[k] = np.max(depth_vec)

        simple_avg_depth = np.mean(depth_by_tree)
        weighted_avg_depth = np.dot(self.tree_distribution, depth_by_tree)
        max_depth = np.max(depth_by_tree)

        return simple_avg_depth, weighted_avg_depth, max_depth

    ## function tcm_predictForest in tcm_predictForest.jl

    def predictForest(self):
        # number of assortments
        n_assortments = len(self.assort_set)
        n_products = len(self.products)
        v = np.zeros((n_assortments, n_products))
        v_by_class = np.zeros((self.num_tree, n_assortments, n_products))

        for t in range(self.num_tree):
            v_one_class = np.zeros((n_assortments, n_products))
            for m in range(n_assortments):
                cn = 0
                while not self.trees[t].isLeaf[cn]:
                    # print(cn, len(self.trees[t].isLeaf))
                    # print(len(self.assort_set[m]), self.trees[t].product[cn])
                    if self.assort_set[m][self.trees[t].product[cn] - 1] > 0:
                        cn = self.trees[t].tree_left[cn]
                    else:
                        cn = self.trees[t].tree_right[cn]
                    if cn >= len(self.trees[t].isLeaf):
                        break
                    # print("single tree is Leaf %d cn: %d" % ( len(self.trees[t].isLeaf),  cn))
                    # print(self.trees[t].isLeaf[cn])
                if cn < len(self.trees[t].isLeaf):
                    v_one_class[m][self.trees[t].product[cn]] = 1
            # print(self.tree_distribution[t])
            # print(v_one_class)
            v += self.tree_distribution[t] * v_one_class

            v_by_class[t] = v_one_class
        return v, v_by_class

    ## function tcm_concertRankingtoForest in tcm_estimateForest.l;
    ## input: orderings: the position of product 0-N
    def convertRankingToForest(self, orderings):
        K = len(orderings)
        self.num_tree = K
        self.trees = []
        num_products = len(self.products) - 1

        for k in range(K):
            cn = 0
            tree = DecisionTree([], [], [], [])
            for i in range(num_products + 1):
                # if the preferred choice is the zero-option
                if orderings[k][i] == 0:
                    # -1 means no child node
                    tree.tree_left.append(-1)
                    tree.tree_right.append(-1)
                    tree.product.append(orderings[k][i])
                    tree.isLeaf.append(True)
                    break
                else:
                    tree.tree_left += [cn + 1, -1]
                    tree.tree_right += [cn + 2, -1]
                    tree.product += [orderings[k][i], orderings[k][i]]
                    tree.isLeaf += [False, True]
                    cn += 2
            # print("------------------------------------")
            # print(orderings[k])
            # print(tree.tree_left)
            # print(tree.tree_right)
            # print(tree.product)
            # print(tree.isLeaf)
            self.trees.append(tree)
        self.tree_distribution = [1/K for k in range(K)]

    def creatBasicForest(self):
        num_products = len(self.products) - 1
        self.num_tree = num_products
        orderings = np.zeros((self.num_tree, num_products + 1))
        for k in range(self.num_tree - 1):
            single_ordering = [k, 0]+ list(np.random.permutation(range(k))) + list(np.random.permutation(range(k+1, num_products)))
            # print(single_ordering)
            orderings[k] = single_ordering
        orderings[self.num_tree-1] = [0] + list(np.random.permutation(range(1, num_products+1)))\

        return self.convertRankingToForest(orderings)

    def probability_of(self, transaction):
        idx = self.assortment_index(transaction)
        return self.probabilities[idx][transaction.product]