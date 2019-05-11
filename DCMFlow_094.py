import abc
import numpy as np
import copy
import pandas as pd
import time
import random
import tensorflow as tf
import math
import collections
import sys
import platform
import os
from textwrap import dedent
"""
 The difference between DCMFlow_093 and DCMFlow_08 is the use 
 of long (a.k.a case-alternative) format
 """
class Optimizer:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

class LbfgsbOptimizer(Optimizer):
    """ A class holds arguments for some of TensorFlow's L-BFGS-B optimizer
        and others specific to `DCMFlow`"""
    model_type = "LbfgsbOptimizer"
    def __init__(self, n_steps=10000,
                 improvement_threshold=1e-30,
                 objective='loglikelihood_sum'):
        """Constructs a new arguments class for Adam optimizer
        Args:
        n_steps: The number of optimization steps to run.
        improvement_threshold: This hyper-parameter controls when to stop 
            optimization. Optimization stops when the reduction in cost 
            (likelihood) does not exceed the improvement threshold.
        """
        self.n_steps = n_steps
        self.improvement_threshold = improvement_threshold
        self.objective = objective
        self.name = 'l-bfgs-b'
class AdamOptimizer(Optimizer):
    """ A class holds arguments for some of TensorFlow's Adam optimizer
        and others specific to `DCMFlow`"""
    model_type = "AdamOptimizer"
    def __init__(self, n_steps=10000,
                 step_size=None,
                 learning_rate=1e-3,
                 beta1=0.9, beta2=0.999, epsilon=1e-8,
                 improvement_threshold=1e-30,
                 patience=100, 
                 log_every_n_epochs=1,
                 objective='loglikelihood_sum',
                 constraints=None,
                 interior_point_penalty_term=0.1):
        """Constructs a new arguments class for Adam optimizer
        Args:
        n_steps: The number of optimization steps to run.
        step_size: The number of cases randomly sampled from the data in each 
            optimization step. The default value is None, which means the 
            whole data will be used for each parameter update iteration 
            (optimization step). For large datasets and/or large trees, you 
            may run out of memory and therefore you will need to set 
            `step_size` to something small (e.g., 10000). If you are sill 
            running out of memory, choose smaller step sizes (e.g., 2000, 
            1000, or 500, etc).
        learning_rate: The initial learning rate used by Adam.
        beta1: A float value or a constant float tensor.
            The exponential decay rate for the 1st moment estimates.
        beta2: A float value or a constant float tensor.
            The exponential decay rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
        improvement_threshold: This hyper-parameter controls when to stop 
            optimization. Optimization stops when the reduction in cost 
            (likelihood) does not exceed the improvement threshold.
        patience: Allows the reduction in cost to fall 
            under `improvement_threshold` value for a number of `patience` 
            steps before exiting the optimization loop.
        log_every_n_epochs: Print the value of cost and other stats every 
            `log_every_n_epochs` epochs. The number of epochs for a data 
            set ~ number of cases / `step_size`
        objective: The objective function to optimize. Valid values for 
            `objective` are 'likelihood_mean' and 'likelihood_sum' for the 
            mean value of negative log likelihood across all examples in the 
            given batch or their sum, respectively. The mean value might 
            be useful for the Adam optimizer because, if there are any 
            constraints on the model's parameters, any deviation for the 
            parameters from the constraints will be significant relative to 
            the mean value of negative log likelihood. (need to be confirmed)
        interior_point_penalty_term: This hyper-parameter is added to penalize
            parameters that approach the upper and/or lower boundaries
            provided in the constraints dictionary. For example, if the logsum
            parameters, say L1C1 & L1C2 in the vector L1, are to be constrained
            between 0 and 1, the cost function will be defined as:
            cost = - LL - interior_point_penalty_term*sum(log(L1)) 
                  -  interior_point_penalty_term*sum(log(1-L1)), where LL is
            the mean loglikelihood. To indicate that there is no bound, use
            -np.inf or np.inf in the constraints dictionary depending whether 
            there is no lower bound or upper bound, respectively.
            Note that the optimization process is very
            sensitive to this term. Too small values my let the parameters inch
            closer and closer to the boundaries and if they jump them we get
            nan values due to the negative number under the logs. If the 
            optimization process terminated with Divergence message with 'nan'
            value for the cost (objective), consider increasing the value of 
            `interior_point_penalty_term`. Keep in mind that too large values 
            may push parameters too far away from the boundaries and as a 
            result you may never find the optimal set of parameters. In the 
            future, this hyper-parameter may be updated during optimization 
            (e.g., decays).
        """
        self.n_steps = n_steps
        self.step_size = step_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.improvement_threshold = improvement_threshold
        self.patience = patience
        self.log_every_n_epochs = log_every_n_epochs
        self.objective = objective
        self.interior_point_penalty_term = interior_point_penalty_term
        self.name = 'adam'

class DCMFlow:
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def compute_choice_probabilities(self):
        pass

    @abc.abstractmethod
    def compute_choices(self):
        pass

    @abc.abstractmethod
    def compute_likelihood(self):
        pass



class NLFlow(DCMFlow):
    model_type = "NLFlow"
    def __init__(self, nltree_dict,
                 utilities_dict,
                 dtype=tf.float32,
                 pin_data_to_gpu=False):
        DCMFlow.__init__(self)
        self.nltree_dict = nltree_dict
        assert type(self.nltree_dict) in [collections.OrderedDict, dict]
        self.n_levels = len(nltree_dict) - 1
        self.n_nests = 0
        for l in range(1, self.n_levels):
            self.n_nests += len(self.nltree_dict['l' + str(l)])
        self.n_choices = len(self.nltree_dict['l' + str(self.n_levels)])

        self.utilities = utilities_dict
        assert type(self.utilities) is collections.OrderedDict 
        assert len(self.utilities) == self.n_choices
        # Check whether the keys in the utilities are the same as
        # the name of choices (leaf nodes in the lowest level of the tree)
        assert list(self.utilities) == self.nltree_dict['l'+str(self.n_levels)]

        self.sess = None
        self.dtype = dtype
        assert self.dtype in [tf.float32, tf.float64]
        self.global_time = 0
        self.global_steps = 0
        self.pin_data_to_gpu = pin_data_to_gpu
        assert self.pin_data_to_gpu in [True, False]

        self.parse_utilities_and_create_maps(self.utilities)
        self.objective = 'loglikelihood_sum'
        self.unavilable_alternatives_epsilon = np.exp(11) if self.dtype==tf.float32 else np.exp(20)
        self.est_parameters_dict = None
        # Initialize tensors to None so that we can debug them
        #self.normalized_ftr_price_adj_t = tf.constant([0], self.dtype)
        #self.euti = None
        #self.val_eq = None
        #self.cond_pr = None
        #self.choice_probabilities = None
        #self.chosen_choice_probabilities = None
    #---------------------------------------------------------------------------
    def parse_utilities(self, utilities):
        """
        Reads the utility 
        for each alternative choice
        to find all constant and coefficient 
        paramters and covarites.
        
        Args:
            utilities: A dictionary of utilities. The key is the choice_id
                       and the value is the linear equation formula of
                       the choice's utility in terms of constant 
                       paramters (e.g., A1 and A2 in the
                       example below), coefficient paramters (B1,
                       B2, and B3), and covaraites (X1, X2, and X3).

                       #Example:
                       utilities['C1'] = 'A1 + B1*X1 + A2 + B2*X2'
                       utilities['C2'] = 'A1 + B1*X1 + B3*X3'
        Returns:
            choice_names: A list of choice IDs or names (e.g., [C1, C2])
            const_names: A list of constant paramters (e.g., [A1, A2])
            coef_names: A list of coefficient paramters (e.g., [B1, B2, B3])
            covariate_names: A list of covariates (e.g., [X1, X2, X3])    
        """
        choice_names = []
        constant_names = []
        coefficient_names = []
        covariate_names = []
        for ut in utilities:
            choice_names.append(ut)
            equation = utilities[ut].replace(' ', '')
            terms = equation.split('+')
            for term in terms:
                if '*' in term:
                    tokens = term.split('*')
                    if len(tokens) == 2:
                        coefficient_name, covariate_name = tokens
                        if coefficient_name not in coefficient_names:
                            coefficient_names.append(coefficient_name)
                        if covariate_name not in covariate_names:
                            #print(ut, covariate_name)
                            covariate_names.append(covariate_name)
                    else:
                        print('Incorrect term %s in utility %s'%(term, ut))
                else:
                    if term not in constant_names:
                        constant_names.append(term)
        # logsum names
        logsum_names = []
        n_levels = len(self.nltree_dict) - 1
        for level in range(1, n_levels):
            nest_choices = self.nltree_dict['l'+str(level)]
            logsum_names.extend(nest_choices)
        return [choice_names, logsum_names, constant_names, coefficient_names, covariate_names]
    #---------------------------------------------------------------------------
    def create_index_dictionaries(self, unique_vals_list):
        """
        Creates dictionaries to convert names to indices
        and indices to names.
        
        Args:
            unique_vals_list: A list of values to create indices for. 
            E.g., ['C1', 'C2'].
        
        Returns a list with the following two dictionaries:
            vals_to_indices: A dictionary where keys are the items in the list
                and values are the indices
            indices_to_vals: A dictionary where keys are the indices for the values
                in the list.
        """
        vals_to_indices = {k:i for i,k in enumerate(unique_vals_list)}
        indices_to_vals = {i:k for i,k in enumerate(unique_vals_list)}
        return [vals_to_indices, indices_to_vals]
    #---------------------------------------------------------------------------
    def parse_utilities_and_create_maps(self, utilities):
        """
        Reads the utility 
        for each alternative choice
        to find all constant and coefficient 
        paramters and covarites.
        
        Args:
            utilities: A dictionary of utilities. The key is the choice_id
                       and the value is the linear equation formula of
                       the choice's utility in terms of constant 
                       paramters (e.g., 'A1' and 'A2' in the
                       example below), coefficient paramters ('B1',
                       'B2', and 'B3'), and covaraites ('X1', 'X2', and 'X3').

                       #Example:
                       utilities['C1'] = 'A1 + B1*X1 + A2 + B2*X2'
                       utilities['C2'] = 'A1 + B1*X1 + B3*X3'
        Returns:
            choice_names: A list of choice IDs or names (e.g., ['C1', 'C2'])
            const_names: A list of constant paramters (e.g., ['A1', 'A2'])
            coef_names: A list of coefficient paramters (e.g., ['B1', 'B2', 'B3'])
            covariate_names: A list of covariates (e.g., ['X1', 'X2', 'X3'])    
        """
        choice_names, logsum_names, constant_names, coefficient_names, covariate_names = self.parse_utilities(utilities)

        choice_names2indices, choice_indices2names = self.create_index_dictionaries(choice_names)
        logsum_names2indices, logsum_indices2names = self.create_index_dictionaries(logsum_names)
        constant_names2indices, constant_indices2names = self.create_index_dictionaries(constant_names)
        coefficient_names2indices, coefficient_indices2names = self.create_index_dictionaries(['_null_coef_'] + coefficient_names)
        covariate_names2indices, covariate_indices2names = self.create_index_dictionaries(covariate_names)
        
        n_choices = len(choice_names)
        n_logsums = len(logsum_names)
        n_constants = len(constant_names)
        n_coefficients = len(coefficient_names)
        n_covariates = len(covariate_names)
        
        choices2covariates_map = np.zeros(shape=[n_choices, n_covariates], dtype=np.int)
        choices2constants_map = np.zeros(shape=[n_choices, n_constants], dtype=np.float32)
        
        for ut in utilities:
            choice_ix = choice_names2indices[ut]
            equation = utilities[ut].replace(' ', '')
            terms = equation.split('+')
            for term in terms:
                if '*' in term:
                    tokens = term.split('*')
                    if len(tokens) == 2:
                        coefficient_name, covariate_name = tokens
                        coefficient_ix = coefficient_names2indices[coefficient_name]
                        covariate_ix = covariate_names2indices[covariate_name]
                        choices2covariates_map[choice_ix, covariate_ix] = coefficient_ix
                    else:
                        print('Incorrect term %s in utility %s'%(term, ut))
                else:
                    constant_ix = constant_names2indices[term]
                    #print(term, choice_ix, constant_ix, choices2constants_map.shape)
                    choices2constants_map[choice_ix, constant_ix] = 1.0
        
        # Return list of lists
        assert n_choices == self.n_choices
        self.n_logsums = n_logsums
        self.n_constants = n_constants
        self.n_coefficients = n_coefficients
        self.n_covariates = n_covariates

        self.choice_names = choice_names
        self.constant_names = constant_names
        self.logsum_names = logsum_names
        self.coefficient_names = coefficient_names
        self.covariate_names = covariate_names

        self.choice_names2indices = choice_names2indices
        self.choice_indices2names = choice_indices2names
        self.constant_indices2names = constant_indices2names
        self.constant_names2indices = constant_names2indices
        self.logsum_indices2names = logsum_indices2names
        self.logsum_names2indices = logsum_names2indices
        self.coefficient_names2indices = coefficient_names2indices
        self.coefficient_indices2names = coefficient_indices2names
        self.covariate_names2indices = covariate_names2indices
        self.covariate_indices2names = covariate_indices2names
        
        self.choices2covariates_map = choices2covariates_map
        self.choices2covariates_map_df = pd.DataFrame(choices2covariates_map,
                                                      index=choice_names,
                                                      columns=covariate_names)
        np_dtype = np.float32 if self.dtype==tf.float32 else np.float64
        self.choices2constants_map = choices2constants_map.astype(np_dtype)
        self.choices2constants_map_df = pd.DataFrame(choices2constants_map,
                                                  index=choice_names,
                                                  columns=constant_names)
    #---------------------------------------------------------------------------
    def build_graph(self):
        nltree_dict = self.nltree_dict
        self.graph = tf.Graph()
        with self.graph.as_default():
            parent1, parent2 = self.build_indices(nltree_dict)
            # n_covariates X n_choices
            self.covariates2choices_map_t = tf.constant(self.choices2covariates_map, tf.int32, name='covariates2choices_map')
            # n_choices X n_constants
            self.choices2constants_map_t = tf.constant(self.choices2constants_map, self.dtype, name='covariates2choices_map')

            if self.pin_data_to_gpu: # for better performance when data is small enough to fit into GPU
                # A matrix with dimensions = n_choices*n_price_levels X n_features
                self.covariates_t = tf.constant(self.covariates, self.dtype, name='features_prices')
                self.case_choice_idxs_t = tf.constant(self.case_choice_idxs, tf.int32, name='case_choice_idxs_t')
                # A matrix with dimensions = n_observations X 2
                self.choices_y_t = tf.constant(self.choices_y, tf.int32, name='choices_y')
                self.n_cases_per_batch_t = tf.constant(self.n_cases_per_batch, tf.int32, name='n_cases_per_batch_t')
            else:
                self.covariates_t = tf.placeholder(self.dtype, shape=[None, None], name='covariates')
                # A matrix with dimensions = n_observations X n_choices
                self.case_choice_idxs_t = tf.placeholder(tf.int32, shape=[None, None], name='case_choice_idxs_t')
                # A matrix with dimensions = n_observations X 2
                self.choices_y_t = tf.placeholder(tf.int32, shape=[None, None], name='choices_y')
                self.n_cases_per_batch_t = tf.placeholder(tf.int32, shape=None, name='n_cases_per_batch_t')
                
            self.learning_rate = tf.placeholder(self.dtype, name='learning_rate')
            self.logsum_parameters_reg = tf.placeholder(self.dtype, name='logsum_parameters_regularizer')
                        
            if self.task == 'training':
                n_logsums = self.n_nests
                
                #Initial values of the logsum parameters
                init_logsum_parameters = self.get_initial_values(self.logsum_indices2names,
                                                                 self.initial_values_dict,
                                                                 self.constraints,
                                                                 is_for_logsums=True)               
                # Initial values of the constant parameters
                init_constants = self.get_initial_values(self.constant_indices2names,
                                                         self.initial_values_dict,
                                                         self.constraints,
                                                         is_for_logsums=False)
                # Initial values of the main coefs
                init_coefficients = self.get_initial_values(self.coefficient_indices2names,
                                                         self.initial_values_dict,
                                                         self.constraints,
                                                         is_for_logsums=False)
                
                self.logsums_t = tf.Variable(initial_value=init_logsum_parameters, dtype=self.dtype, name='logsum_parameters')
                self.constants_t = tf.Variable(initial_value=init_constants, dtype=self.dtype, name='constants')
                self.coefficients_t = tf.Variable(initial_value=init_coefficients, dtype=self.dtype, name='coefficients')
            elif self.task == 'prediction':
                self.logsums_t = tf.placeholder(dtype=self.dtype, shape=[None], name='logsum_parameters')
                self.constants_t = tf.placeholder(dtype=self.dtype, shape=[None], name='constants')
                self.coefficients_t = tf.placeholder(dtype=self.dtype, shape=[None], name='coefficients')

            self.coefficients_with_zprfx = tf.concat([tf.constant([0.0], self.dtype), self.coefficients_t], axis=0, name='coefficients_with_zprfx') 
            self.loaded_cov2chc_map_t = tf.gather(self.coefficients_with_zprfx, self.covariates2choices_map_t)
            #self.sampled_cov2chc_map_t = tf.gather(tf.transpose(self.loaded_cov2chc_map_t), self.case_choice_idxs_t[:,1])
            self.sampled_cov2chc_map_t = tf.gather(self.loaded_cov2chc_map_t, self.case_choice_idxs_t[:,1])
            self.cov_times_coefs = tf.multiply(self.covariates_t, self.sampled_cov2chc_map_t)

            self.consts_sum_per_choice = tf.reduce_sum(tf.multiply(self.choices2constants_map_t, self.constants_t), axis=1)
            self.sampled_consts_sum_per_choice = tf.gather(self.consts_sum_per_choice, self.case_choice_idxs_t[:,1])

            self.uts_per_case_choice_long = tf.reduce_sum(self.cov_times_coefs, axis=1) + self.sampled_consts_sum_per_choice
            self.uts_per_case_choice_wide = tf.scatter_nd(self.case_choice_idxs_t, 
                                                     self.uts_per_case_choice_long,
                                                     [self.n_cases_per_batch_t, self.n_choices])
 
            raw_euti = tf.exp(self.uts_per_case_choice_wide)

            self.val_eq = dict()

            self.choice_availability_mask_t = tf.scatter_nd(self.case_choice_idxs_t, 
                                                 tf.ones_like(self.uts_per_case_choice_long, dtype=self.dtype),
                                                 [self.n_cases_per_batch_t, self.n_choices])


            self.masked_euti = tf.multiply(raw_euti, self.choice_availability_mask_t)
            self.min_euti = tf.reduce_min(raw_euti)
            euti_epsilon = self.min_euti/self.unavilable_alternatives_epsilon
            self.euti = self.masked_euti + euti_epsilon * (1.0-self.choice_availability_mask_t)

            self.val_eq['l'+str(self.n_levels)] = self.euti
            theta_end_i = self.n_nests
            for i in range(self.n_levels-1, 0, -1):
                theta_st_i = theta_end_i - len(nltree_dict['l'+str(i)])
                l_theta = tf.transpose(tf.gather(tf.transpose(self.logsums_t), tf.range(theta_st_i, theta_end_i)))
                theta_end_i = theta_st_i
                tmps = []
                for j in range(len(nltree_dict['l'+str(i)])):
                    src = self.val_eq['l'+str(i+1)]
                    indices = [k for k in range(len(parent1['l'+str(i+1)])) 
                                               if parent1['l'+str(i+1)][k]==j]
                    s_coef = tf.transpose(tf.gather(tf.transpose(l_theta), [j]))
                    tmp1 = tf.gather(tf.transpose(src), indices) # slice specific columns
                    tmp2 = tf.reduce_sum(tf.pow(tmp1, 1.0/s_coef), axis=0) # sum selected columns
                    tmp3 = tf.pow(tmp2, s_coef) 
                    tmps.append(tf.reshape(tmp3, [-1,1]))
                self.val_eq['l'+str(i)] = tf.concat(tmps, 1)
            self.cond_pr = dict()
            theta_end_i = self.n_nests
            for i in range(self.n_levels, 1, -1):
                indices = parent1['l' + str(i)]              
                theta_st_i = theta_end_i - len(nltree_dict['l'+str(i-1)])
                l_theta = tf.transpose(tf.gather(tf.transpose(self.logsums_t), tf.range(theta_st_i, theta_end_i)))
                s_coef = tf.transpose(tf.gather(tf.transpose(l_theta), indices))
                theta_end_i = theta_st_i
                src = self.val_eq['l'+str(i-1)]
                tmp1 = tf.transpose(tf.gather(tf.transpose(src), indices))
                denom = tf.pow(tmp1, 1.0/s_coef)
                numer = tf.pow(self.val_eq['l'+str(i)], 1.0/s_coef, name='numer_'+str(i))
                self.cond_pr['l'+str(i)] = tf.divide(numer, denom)
            denom = tf.reshape(tf.reduce_sum(self.val_eq['l1'], axis=1), [-1,1])
            self.cond_pr['l1'] = tf.divide(self.val_eq['l1'], denom)
            confg_pr = self.cond_pr['l'+str(self.n_levels)]
            for i in range(self.n_levels, 1, -1):
                src = self.cond_pr['l'+str(i-1)]
                indices = parent2['l' + str(i)]
                tmp1 = tf.transpose(tf.gather(tf.transpose(src), indices))
                confg_pr = tf.multiply(confg_pr, tmp1)
            self.choice_probabilities = confg_pr
            self.chosen_choice_probabilities = tf.gather_nd(self.choice_probabilities, self.choices_y_t)

            if self.objective == 'loglikelihood_mean':
                self.loglikelihood = tf.reduce_mean(tf.log(self.chosen_choice_probabilities))
            elif self.objective == 'loglikelihood_sum':
                self.loglikelihood = tf.reduce_sum(tf.log(self.chosen_choice_probabilities))

            if self.task == 'training':
                if self.optimizer.name == 'l-bfgs-b':
                    logsum_bounds = (np.zeros(n_logsums), np.ones(n_logsums))
                    constant_bounds = (np.zeros(self.n_constants), np.inf*np.ones(self.n_constants))
                    coefficient_bounds = (-np.inf*np.ones(self.n_coefficients), np.zeros(self.n_coefficients))
                    var_to_bounds = self.get_var_to_bounds_constraints(self.constraints)
                    #print(var_to_bounds)
                    self.opt = tf.contrib.opt.ScipyOptimizerInterface(-self.loglikelihood, 
                                                   var_to_bounds = var_to_bounds,
                                                    method='L-BFGS-B',
                                                    options={'maxiter': self.optimizer.n_steps,
                                                             'ftol': self.optimizer.improvement_threshold})
                elif self.optimizer.name == 'adam':
                    var_list = []
                    var_list = var_list + [self.logsums_t]
                    var_list = var_list + [self.constants_t]
                    var_list = var_list + [self.coefficients_t]

                    self.cost = -self.loglikelihood
                    near_boundary_cost = self.get_bound_constraints(self.constraints)
                    if near_boundary_cost is not None:
                        ip_penalty = self.optimizer.interior_point_penalty_term
                        self.cost -= (ip_penalty*near_boundary_cost) 
                    self.opt = tf.train.AdamOptimizer(learning_rate=self.optimizer.learning_rate,
                                    beta1=self.optimizer.beta1, 
                                    beta2=self.optimizer.beta2,
                                    epsilon=self.optimizer.epsilon)
                    #self.gvs_t = self.opt.compute_gradients(self.cost, var_list)
                    self.train_op = self.opt.minimize(self.cost, var_list=var_list)
            self.init_op = tf.global_variables_initializer()
    #---------------------------------------------------------------------------
    def build_indices(self, nltree_dict):
        n_levels = len(nltree_dict) - 1
        parent1 = dict()
        for lvl in range(n_levels, 1, -1):
            parent1['l'+str(lvl)] = [j for i in range(len(nltree_dict['l'+str(lvl)])) 
                                       for j in range(len(nltree_dict['l'+str(lvl-1)])) 
                                       if nltree_dict['l'+str(lvl-1)][j] in nltree_dict['l'+str(lvl)][i]]
        
        parent2 = dict()
        for lvl in range(n_levels, 1, -1):
            parent2['l'+str(lvl)] = [j for i in range(len(nltree_dict['l'+str(n_levels)])) 
                                       for j in range(len(nltree_dict['l'+str(lvl-1)])) 
                                       if nltree_dict['l'+str(lvl-1)][j] in nltree_dict['l'+str(n_levels)][i]]
        return [parent1, parent2]
    #---------------------------------------------------------------------------
    def get_bound_vectors(self, constraints, param_names_to_constrain, 
                  n_params, param_names2indices, index_adjust=0):
        param_lower_bounds = -np.inf*np.ones(n_params)
        param_upper_bounds = np.inf*np.ones(n_params)
        for param_name_to_constrain in param_names_to_constrain:
            lower_bound, upper_bound = constraints[param_name_to_constrain]
            param_index = param_names2indices[param_name_to_constrain] + index_adjust
            #print(param_name_to_constrain, param_index)
            param_lower_bounds[param_index] = lower_bound
            param_upper_bounds[param_index] = upper_bound
        param_bounds = (param_lower_bounds, param_upper_bounds)
        #print(param_bounds)
        return param_bounds
    #---------------------------------------------------------------------------
    def enforce_constraints(self, lu_constraint, 
                            param_name=None,
                            requested_init_value=None,
                            default_init_value=None):
        l,u = lu_constraint
        mn, mx = -1e7, 1e7
        l_hat = l if l != 0 else 0.0001
        u_hat = u if u != 0 else -0.0001
        if (requested_init_value is not None and 
            requested_init_value > l_hat and
            requested_init_value < u_hat):
            init_value = requested_init_value*1.0
        elif (default_init_value is not None and 
            default_init_value > l_hat and
            default_init_value < u_hat):
            init_value = default_init_value*1.0
        else:
            if l < mn and u > mx:
                l_hat, u_hat = [0, 0]
            elif l < mn and u <= mx:
                l_hat = u_hat - 2*abs(u_hat)
            elif l >= mn and u > mx:
                u_hat = l_hat + 2*abs(l_hat)
            c = (u_hat - l_hat)/2.0 + l_hat
            l_hat2 = c - (u_hat - l_hat)/10.0
            u_hat2 = c + (u_hat - l_hat)/10.0
            init_value = np.random.uniform(l_hat2, u_hat2)
            if (requested_init_value is not None and 
                (requested_init_value < l or requested_init_value > u)):
                msg = ("The requested initial value of %g for the parameter '%s'\n"
                       + "is out of its lower-upper bound [%g, %g].\n"
                       + "Defaulting to the initial value of %g instead.")
                print(msg%(requested_init_value, param_name, l, u, init_value))
        return init_value
    #---------------------------------------------------------------------------
    def get_initial_values(self, indices2names, initial_vals_dict=None, 
                            constraints_dict=None,
                            is_for_logsums=False):
        initial_vals_dict = {} if initial_vals_dict is None else initial_vals_dict
        constraints_dict = {} if constraints_dict is None else constraints_dict

        np_dtype = np.float32 if self.dtype==tf.float32 else np.float64
        n_params = len(indices2names)
        index_adj = 0
        if indices2names[0] == '_null_coef_':
            n_params -= 1
            index_adj = 1
        init_params = np.zeros(n_params, np_dtype)
        if is_for_logsums:
            init_params += 0.9
        for p_index in range(n_params):
            p_name = indices2names[p_index+index_adj]
            if p_name in initial_vals_dict and p_name in constraints_dict:
                init_val = self.enforce_constraints(constraints_dict[p_name],
                                                   p_name,
                                                   initial_vals_dict[p_name])
                init_params[p_index] = init_val
            elif p_name not in initial_vals_dict and p_name in constraints_dict:
                init_val = self.enforce_constraints(constraints_dict[p_name],
                                   p_name,
                                   None,
                                   init_params[p_index])
                init_params[p_index] = init_val
            elif p_name in initial_vals_dict and p_name not in constraints_dict:
                init_params[p_index] = initial_vals_dict[p_name]
        return init_params
    #---------------------------------------------------------------------------
    def get_var_to_bounds_constraints(self, constraints):
        var_to_bounds = None
        if constraints is not None:
            var_to_bounds = {}
            logsum_names_to_constrain = set(constraints).intersection(self.logsum_names)
            if len(logsum_names_to_constrain) > 0:
                logsum_bounds = self.get_bound_vectors(constraints, logsum_names_to_constrain,
                                                       self.n_logsums, self.logsum_names2indices)
                var_to_bounds[self.logsums_t] = logsum_bounds

            constant_names_to_constrain = set(constraints).intersection(self.constant_names)
            if len(constant_names_to_constrain) > 0:
                constant_bounds = self.get_bound_vectors(constraints, constant_names_to_constrain,
                                                       self.n_constants, self.constant_names2indices)
                var_to_bounds[self.constants_t] = constant_bounds

            coefficient_names_to_constrain = set(constraints).intersection(self.coefficient_names)
            if len(coefficient_names_to_constrain) > 0:
                #print(self.coefficients_t.get_shape(), self.n_coefficients)
                coefficient_bounds = self.get_bound_vectors(constraints, coefficient_names_to_constrain,
                                                       self.n_coefficients, self.coefficient_names2indices,-1)
                var_to_bounds[self.coefficients_t] = coefficient_bounds
        return var_to_bounds
    #---------------------------------------------------------------------------
    def get_bound_cost(self, constraints, param_names_to_constrain, 
                  n_params, param_names2indices, param_tensor, 
                  index_adjust=0):
        lower_bounds = []
        upper_bounds = []
        lower_indices = []
        upper_indices = []
        for param_name_to_constrain in param_names_to_constrain:
            lower_bound, upper_bound = constraints[param_name_to_constrain]
            param_index = param_names2indices[param_name_to_constrain] + index_adjust
            if lower_bound != -np.inf:
                lower_bounds.append(lower_bound)
                lower_indices.append(param_index)
            if upper_bound != np.inf:
                upper_bounds.append(upper_bound)
                upper_indices.append(param_index)
        boundary_cost = 0.0
        np_dtype = np.float32 if self.dtype==tf.float32 else np.float64
        if len(lower_bounds) > 0:
            lower_bounds = np.array(lower_bounds, np_dtype)
            tensor_subset = tf.gather(param_tensor, lower_indices)
            boundary_cost += tf.reduce_sum(tf.log(tensor_subset - lower_bounds))
        if len(upper_bounds) > 0:
            upper_bounds = np.array(upper_bounds, np_dtype)
            tensor_subset = tf.gather(param_tensor, upper_indices)
            boundary_cost += tf.reduce_sum(tf.log(upper_bounds - tensor_subset))
        return boundary_cost
    #---------------------------------------------------------------------------
    def get_bound_constraints(self, constraints):
        boundary_cost = None
        if constraints is not None:
            p_list_to_constraint = list(constraints)

            logsum_names_to_constrain = set(constraints).intersection(self.logsum_names)
            if len(logsum_names_to_constrain) > 0:
                boundary_cost = 0 if boundary_cost is None else boundary_cost
                boundary_cost += self.get_bound_cost(constraints, logsum_names_to_constrain,
                                                    self.n_logsums, self.logsum_names2indices,
                                                    self.logsums_t)

            constant_names_to_constrain = set(constraints).intersection(self.constant_names)
            if len(constant_names_to_constrain) > 0:
                boundary_cost = 0 if boundary_cost is None else boundary_cost
                boundary_cost += self.get_bound_cost(constraints, constant_names_to_constrain,
                                                       self.n_constants, self.constant_names2indices,
                                                       self.constants_t)
 
            coefficient_names_to_constrain = set(constraints).intersection(self.coefficient_names)
            if len(coefficient_names_to_constrain) > 0:
                boundary_cost = 0 if boundary_cost is None else boundary_cost
                boundary_cost += self.get_bound_cost(constraints, coefficient_names_to_constrain,
                                                       self.n_coefficients, self.coefficient_names2indices,
                                                       self.coefficients_t,-1)
        return boundary_cost
    #---------------------------------------------------------------------------
    def convert_param_dic_to_vectors(self, true_params_dict):
        # logsums: order is important (top-to-bottom & left-to-right)
        logsum_parameters = []
        for logsum_name in self.logsum_names:
            logsum_parameters.append(true_params_dict[logsum_name])
        logsum_parameters = np.array(logsum_parameters)
        # constants:
        constants = []
        for const_name in self.constant_names:
            constants.append(true_params_dict[const_name])
        constants = np.array(constants)
        # coefficients
        coefficients = []
        for coef_name in self.coefficient_names:
            coefficients.append(true_params_dict[coef_name])
        coefficients = np.array(coefficients)
        return [logsum_parameters, constants, coefficients]
    #---------------------------------------------------------------------------
    def fit(self, data,
            optimizer='l-bfgs-b',
            true_params_dict=None,
            initial_values_dict=None,
            constraints_dict=None,
            unavilable_alternatives_epsilon='auto',
            start_over=False,
            verbose=1):
        self.task = 'training'
        if verbose > 0:
            print('Digesting the data...')
        X, choices_y, case_choice_idxs = self.parse_data(data)
        self.covariates = X.values
        self.case_choice_idxs = case_choice_idxs.values
        self.choice_matrix_indices = np.stack((np.arange(choices_y.shape[0]), 
                                               np.array(choices_y.reshape(-1))), axis=-1)
        self.n_cases_per_batch = self.n_cases
        self.constraints = constraints_dict

        np_dtype = np.float32 if self.dtype==tf.float32 else np.float64

        if unavilable_alternatives_epsilon=='auto':
            self.unavilable_alternatives_epsilon = np.exp(11) if self.dtype==tf.float32 else np.exp(20)
        elif unavilable_alternatives_epsilon > 100 and unavilable_alternatives_epsilon < 1e50:
            self.unavilable_alternatives_epsilon = unavilable_alternatives_epsilon
        else:
            print('Invalid value for the unavilable_alternatives_epsilon')
            print("It must be 'auto' or a number between 100 and 1e50")
            sys.exit()
        if start_over:
            self.est_parameters_dict = None
            self.initial_values_dict = initial_values_dict
        else:
            self.initial_values_dict = self.est_parameters_dict
        # Set up optimizer
        opt_name = None
        optimizer_ = optimizer
        self.optimizer = None
        if isinstance(optimizer_, str):
            opt_name = optimizer_.lower()
            if opt_name == 'l-bfgs-b':
                self.optimizer = LbfgsbOptimizer()
            elif opt_name == 'adam': 
                self.optimizer = AdamOptimizer()
            else:
                print('ERROR: Invalid optimizer %s'%(optimizer_))
                print('Valid Optimizers are: Adam, L-BFGS-B')
                return     
        elif 'LbfgsbOptimizer' in str(type(optimizer_)):
            self.optimizer = optimizer_
            opt_name = 'l-bfgs-b'
        elif 'AdamOptimizer' in str(type(optimizer_)):
            self.optimizer = optimizer_
            opt_name = 'adam'
        else:
            print('ERROR: Invalid optimizer %s', str(type(optimizer_)))
            print('Valid Optimizers are: AdamOptimizer and LbfgsbOptimizer')
            return

        self.objective = self.optimizer.objective
        self.build_graph()
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init_op)
        t1 = time.time()
        self.fd = {self.covariates_t: self.covariates,
                   self.case_choice_idxs_t: self.case_choice_idxs,
                   self.choices_y_t: self.choice_matrix_indices,
                   self.n_cases_per_batch_t: self.n_cases}
        #if not self.pin_data_to_gpu:
        #    self.fd[self.covariates_x_t] = self.covariates_x
        #    self.fd[self.choices_y_t] = self.choice_matrix_indices
        if verbose > 0:
            print('\nStarting the optimization process using %s'%(opt_name))
        exit_status = ['STILL CONVERGING', 'CONVERGED', 'DIVERGED'][0]   
        t1 = time.time()
        if self.optimizer.name == 'l-bfgs-b':
            try:
                self.opt.minimize(self.sess, feed_dict=self.fd)  
                ops = [self.loglikelihood, 
                       self.constants_t, self.coefficients_t, 
                       self.logsums_t]
                ops_results = self.sess.run(ops, feed_dict=self.fd)
            except Exception as e:
                if 'Dst tensor is not initialized' in str(e):
                    print('\nERROR: unable to optimize using L-BFGS-B.')
                    print('This could be a memory limitation problem, to solve')
                    print('either try to down sample the data or use the Adam')
                    print('optimizer with small mini-batches as in:')
                    print('fit(..., opt=AdamOptimizer(step_size=1000))')
                return
            except:
                print('Some other error appeared')
            (loglikelihood_val, self.opt_const_vals, self.opt_coef_vals, 
                self.opt_logsum_vals) = ops_results
            tx = time.time()
            self.est_parameters_dict = self.param_vectors_to_dict(self.opt_logsum_vals,
                            self.opt_const_vals, self.opt_coef_vals)
            est_accuracy = None
            if true_params_dict is not None:
                est_accuracy = self.get_param_est_acc_summary(true_params_dict)
            ll_sum, ll_mean = self.compute_ll_sum_and_mean(loglikelihood_val, self.n_cases)
            stat_names = ('Loglikelihood (Sum/Mean)=%.1f/%.3f, T=%ds')
            stat_values =(-ll_sum, -ll_mean, tx-t1)
            summary_line = stat_names%stat_values 
            if est_accuracy is not None:
                summary_line = summary_line + ', ' + est_accuracy
            print(summary_line)
        elif self.optimizer.name == 'adam':
            abrv_msg = """\n
                           Abbreviations key:
                           S: Step, E: Epoch, L: Loglikelihood, S/M: Sum/Mean
                           C: Cost (loglikehood + penalty terms for constrained 
                              parameters, printed when constraints are used)
                           T: Time, Cf/Ls/Cn: Coefficient/Logsum/Constant
                           MAPE: Mean Absolute Percentage Error (printed when true
                                 parameters are provided)
                           RMSE: Root Mean Square Error (printed when true parameters
                                 are provided)\n
                       """
            print(dedent(abrv_msg))
            step_size = self.optimizer.step_size
            if verbose > 1:
                print('N.cases = %d, step_size = %d'%(self.n_cases, step_size))
            step_size = self.n_cases if step_size is None else min(self.n_cases, step_size)
            step_start_indices = np.arange(0, self.n_cases, step_size)
            n_steps_per_epoch = len(step_start_indices)
            random_indices = np.random.choice(self.n_cases, self.n_cases, replace=False)
            unq_case_idxs = case_choice_idxs['case_idxs'].unique()
            random_case_idxs = np.random.choice(unq_case_idxs, self.n_cases, replace=False)
            self.n_violations = 0
            epoch = 0
            exit_status, exit_report = 'UNKNOWN', 'PREMATURE EXIT'
            lowest_cost = np.inf
            highest_ll = -np.inf
            self.cost_per_epoch = []
            self.ll_per_epoch = []
            agg_samp_time = 0
            self.p_logsums, self.p_consts, self.p_coefs = None, None, None
            n_cases_per_epoch = 0
            for step in range(self.optimizer.n_steps):
                if step_size < self.n_cases:
                    ts_st = time.time()
                    step_start_index = step_start_indices[step%n_steps_per_epoch]
                    step_end_index = min(step_start_index + step_size, self.n_cases)
                    batch_case_idxs = random_case_idxs[step_start_index: step_end_index]
                    batch_cace_choice_idxs = np.flatnonzero(case_choice_idxs['case_idxs'].isin(batch_case_idxs))
                    self.n_cases_per_batch = len(batch_case_idxs)
                    self.fd[self.covariates_t] = self.covariates[batch_cace_choice_idxs,]
                    batch_case_choice_idxs = self.case_choice_idxs[batch_cace_choice_idxs,]
                    batch_case_choice_idxs[:,0] = pd.match(batch_case_choice_idxs[:,0], batch_case_idxs)
                    self.fd[self.case_choice_idxs_t] = batch_case_choice_idxs
                    #self.fd[self.case_choice_idxs_t] = self.case_choice_idxs[batch_cace_choice_idxs,]
                    self.fd[self.choices_y_t] = np.stack((np.arange(batch_case_idxs.shape[0]), 
                                                self.choice_matrix_indices[batch_case_idxs,1]), axis=-1)
                    self.fd[self.n_cases_per_batch_t] = self.n_cases_per_batch
                    ts_en = time.time()
                    #np.flatnonzero(pd.Series([0,0,0,1,1,2,2,2,3,3,3,3,3,4,4,4,5,5]).isin([1,4]))
                    #array([ 3,  4, 13, 14, 15], dtype=int64)
                ops = [self.train_op, self.loglikelihood, self.cost, 
                       self.constants_t, self.coefficients_t, 
                       self.logsums_t]
                if (step+1)%n_steps_per_epoch==0:
                    epoch += 1
                    random_case_idxs = np.random.choice(unq_case_idxs, self.n_cases, replace=False)
                    #agg_samp_time += (ts_en - ts_st)
                    #print(agg_samp_time)
                try:
                    ops_results = self.sess.run(ops, feed_dict=self.fd)
                except Exception as e:
                    if 'Dst tensor is not initialized' in str(e):
                        try_step_size = 1000 if step_size > 1000 else step_size/2
                        print('\nERROR: unable to optimize using Adam with step_size=%d'%(step_size))
                        print('This could be a memory limitation problem, to solve,')
                        print('consider using smaller mini-batches of data as in:')
                        print('fit(..., opt=AdamOptimizer(step_size=%d))'%(try_step_size))
                    return
                (_, loglikelihood_val, cost_val, self.opt_const_vals, self.opt_coef_vals, 
                    self.opt_logsum_vals) = ops_results
                ll_sum, ll_mean = self.compute_ll_sum_and_mean(loglikelihood_val, self.n_cases_per_batch)
                self.ll_per_epoch.append(ll_sum)
                cost_sum, cost_mean = self.compute_ll_sum_and_mean(cost_val, self.n_cases_per_batch)
                n_cases_per_epoch += self.n_cases_per_batch
                self.cost_per_epoch.append(cost_sum)
                if (epoch > 0 and (step+1)%n_steps_per_epoch == 0) or (step == self.optimizer.n_steps-1):
                    epoch_cost_sum = sum(self.cost_per_epoch)
                    epoch_cost_mean = epoch_cost_sum/n_cases_per_epoch
                    epoch_ll_sum = sum(self.ll_per_epoch)
                    epoch_ll_mean = epoch_ll_sum/n_cases_per_epoch
                    self.cost_per_epoch = []
                    self.ll_per_epoch = []
                    n_cases_per_epoch = 0
                    tx = time.time()
                    self.est_parameters_dict = self.param_vectors_to_dict(self.opt_logsum_vals,
                                                self.opt_const_vals, self.opt_coef_vals)
                    est_accuracy = None
                    if true_params_dict is not None:
                        est_accuracy = self.get_param_est_acc_summary(true_params_dict)
                    if constraints_dict is None:
                        stat_names = ('S/E=%d/%d, L(S/M)=%.1f/%.3f, T=%ds')
                        stat_values =(step+1, epoch, -epoch_ll_sum, -epoch_ll_mean, 
                            tx-t1)
                    else:
                        stat_names = ('S/E=%d/%d, L(S/M)=%.1f/%.3f, C(S/M)=%.1f/%.3f, T=%ds')
                        stat_values =(step+1, epoch, -epoch_ll_sum, -epoch_ll_mean, 
                            epoch_cost_sum, epoch_cost_mean, tx-t1)
                    summary_line = stat_names%stat_values 
                    if est_accuracy is not None:
                        summary_line = summary_line + ', ' + est_accuracy
                    if verbose > 0 and (epoch==1 or epoch%self.optimizer.log_every_n_epochs == 0 
                                        or step==self.optimizer.n_steps-1):
                        print(summary_line)
                    exit_status, exit_report = self.check_convergence(epoch_cost_sum, lowest_cost, 
                                                                      self.optimizer.improvement_threshold, 
                                                                      self.optimizer.patience,
                                                                      verbose)
                    if exit_status == 'CONVERGED' or exit_status == 'DIVERGED':
                        break
                    if epoch_cost_sum < lowest_cost:
                        lowest_cost = epoch_cost_sum
                    if epoch_ll_sum > highest_ll:
                        highest_ll = epoch_ll_sum
                        best_step = step
                elif math.isnan(cost_val) or cost_val == np.inf or cost_val == -np.inf:
                        exit_status = 'DIVERGED'
                        exit_report = 'Cost value is ' + str(cost_val)
                        break
            if exit_status == 'STILL CONVERGING' and step == self.optimizer.n_steps - 1:
                exit_report += '\n Max number of steps (%d) has been reached.'%(step+1)
            stars = ''.join(['*']*100)
            msg = stars + '\n' + exit_status + ': ' + exit_report + '\n' 
            print(msg)
    #---------------------------------------------------------------------------
    def compute_ll_sum_and_mean(self, loglikelihood_val, n_cases):
        if self.objective=='loglikelihood_sum':
            ll_sum = loglikelihood_val
            ll_mean = loglikelihood_val/(1.0*n_cases)
        else:
            ll_sum = loglikelihood_val*n_cases
            ll_mean = loglikelihood_val
        return [ll_sum, ll_mean]
    #---------------------------------------------------------------------------
    def check_convergence(self, curr_cost, lowest_cost, improvement_threshold, patience, verbose):    
        improvement = None
        if (lowest_cost != np.inf and not math.isnan(curr_cost) 
            and curr_cost != np.inf and curr_cost != -np.inf): 
            improvement = lowest_cost-curr_cost         
        ric = 'None' if improvement is None else '%g'%(improvement)
        exit_report_msg = ('Improvement in Cost of %s fell under the Improvement Threshold (%g)'
                           + ' %d/%d times')
        exit_report_vals = (ric, improvement_threshold, self.n_violations, patience)
        exit_report = exit_report_msg%exit_report_vals
        exit_status = 'STILL CONVERGING'
        if (math.isnan(curr_cost) or 
            curr_cost == np.inf or curr_cost == -np.inf or
            (improvement is not None and improvement < improvement_threshold)):
            if improvement is not None and self.n_violations < patience:
                if self.n_violations > 0 and self.n_violations%100==0 and verbose > 1:
                    msg = ('Cost = %2.7f, '
                           + 'improvment = %g, '
                           + 'n. violations = %d, patience = %d')
                    print(msg%(curr_cost, improvement, self.n_violations, patience))
                self.n_violations +=1
            else:
                if improvement is None or improvement < -0.01:
                    exit_status = 'DIVERGED'
                    if improvement is None:
                        exit_report = 'Improvement of Cost is None'
                    else:
                        exit_report_msg = ('Improvement of Cost = %g < -0.01')
                        exit_report_vals = (improvement)
                        exit_report = exit_report_msg%exit_report_vals
                elif (improvement is not None 
                      and improvement < improvement_threshold
                      and improvement > -0.01):
                    exit_status = 'CONVERGED'
                    exit_report_msg = ('Improvement in Cost (%g) < Improvement Threshold (%g)'
                                       + ' for at least %d opt. steps (Patience)')
                    exit_report_msg += ("\n(consider increasing the value of the"
                                        + " hyper-parameter 'Patience' if you would" 
                                        + " like to further explore the parameter search" 
                                        + " space.)")
                    exit_report_vals = (improvement, improvement_threshold, patience)
                    exit_report = exit_report_msg%exit_report_vals
        else:
            if verbose > 2:
                print('Resetting the number of violations variable (n_violations) from %d to 0'%(self.n_violations))
            self.n_violations = 0
        return [exit_status, exit_report]
    #---------------------------------------------------------------------------
    def get_param_est_acc_summary(self, true_params_dict):
        sdf = self.get_estimated_parameters(true_params_dict)
        cov_mape, cov_rmse, logsum_mape, logsum_rmse, const_mape, const_rmse = [-1]*6
        true_coefs = sdf[sdf['param_type']=='Coefficient']['param_true_val'].values
        est_coefs = sdf[sdf['param_type']=='Coefficient']['param_est_val'].values
        cov_mape, cov_rmse = self.get_mape_rmse(true_coefs, est_coefs)

        true_logsums = sdf[sdf['param_type']=='Logsum']['param_true_val'].values
        est_logsums = sdf[sdf['param_type']=='Logsum']['param_est_val'].values
        logsum_mape, logsum_rmse = self.get_mape_rmse(true_logsums, est_logsums)

        true_consts = sdf[sdf['param_type']=='Constant']['param_true_val'].values
        est_consts = sdf[sdf['param_type']=='Constant']['param_est_val'].values
        const_mape, const_rmse = self.get_mape_rmse(true_consts, est_consts)

        stat_names = 'Cf/Ls/Cn MAPE=%.1f/%.1f/%.1f%%, RMSE=%.3f/%.3f/%.3f'
        stat_values =(cov_mape, logsum_mape, 
                       const_mape, 
                       cov_rmse, logsum_rmse, 
                       const_rmse)
        summary_line = stat_names%stat_values
        return summary_line
    #---------------------------------------------------------------------------
    def get_mape_rmse(self,true_coefs, opt_coefs):
        mape, rmse = [-1.0, -1.0]
        if (true_coefs is not None and opt_coefs is not None
            and len(true_coefs) == len(opt_coefs)):
            mape = np.mean(100*np.abs((true_coefs[true_coefs!=0.0]-opt_coefs[true_coefs!=0.0])/true_coefs[true_coefs!=0.0]))
            rmse = np.sqrt(np.mean((true_coefs-opt_coefs)**2))
        return [mape, rmse]
    #---------------------------------------------------------------------------
    def print_params(self,true_coefs, opt_coefs):
        pe = 100*np.abs((true_coefs-opt_coefs)/true_coefs)
        for t, o, e in zip(true_coefs, opt_coefs, pe):
            print('%2.4f\t%2.4f\t%2.4f'%(t, o, e))
    #---------------------------------------------------------------------------
    def compute_choice_probabilities(self):
        #TODO
        print('TODO')
    #---------------------------------------------------------------------------
    def predict(self, data, ):
        #TODO
        print('hi')
    #---------------------------------------------------------------------------
    def parse_data(self, data):
        # Check whether caseid column is the data frame
        # Each case must have a unique case ID. The data is supposed to have
        # at least two records with the same case ID
        assert 'caseid' in data.columns
        # Check wheter choiceids column in the data frame
        # choiceid can be indices (when integers) of the choices 
        # or the choice names themselves (when data type is string)
        assert 'choiceid' in data.columns
        t1 = time.time()
        case_choice_info = data[['caseid', 'choiceid']].copy()
        t2 = time.time()
        #print('First Copy = %d'%(t2 - t1))

        if data['choiceid'].dtypes == int:
            case_choice_info['choice_idxs'] = case_choice_info['choiceid']
        else:
            case_choice_info['choice_idxs'] = case_choice_info['choiceid'].apply(lambda x: self.choice_names2indices[x])
        t3 = time.time()
        #print('Apply choiceid to choice_idxs = %d'%(t3-t2))

        unq_case_ids = pd.unique(data['caseid'])
        case_choice_info['case_idxs'] = pd.match(data['caseid'].values, unq_case_ids)
        t4 = time.time()
        #print('unique and finding the indices of unique case ids using pd.match = %d'%(t4-t3))

        case_choice_idxs = case_choice_info[['case_idxs', 'choice_idxs']]
        self.n_cases = unq_case_ids.shape[0]
        X = data.drop(['caseid', 'choiceid'], axis=1)
        t5 = time.time()
        #print('Copying X takes %d'%(t5-t4))
        y_wide = None
        y_long = None
        if 'chosen' in data.columns:
            y_wide = case_choice_info[data['chosen']==1]['choice_idxs'].values
            y_long = data['chosen'].values
            X = X.drop(['chosen'], axis=1)
        assert X.shape[1] == len(self.covariate_names) and set(X.columns) == set(self.covariate_names)
        t6 = time.time()
        #print('Dropping chosen and asserting takes %d'%(t6-t5))
        X = X[self.covariate_names]
        t7 = time.time()
        #print('Selecting X covariates takes %d'%(t7-t6))
        return [X, y_wide, case_choice_idxs]
    #---------------------------------------------------------------------------
    def covert_y_wide_2_long(self, y_wide, data, case_choice_idxs):
        t1 = time.time()
        y_long_left = data[['caseid', 'choiceid']].copy()
        t2 = time.time()
        #print('First copy = %d'%(t2-t1))
        y_long_left['case_idxs'] = case_choice_idxs['case_idxs'].values
        y_long_left['choice_idxs'] = case_choice_idxs['choice_idxs'].values
        n_cases = y_wide.shape[0]
        y_wide_mtx = np.concatenate([np.arange(n_cases).reshape(-1,1), y_wide], axis=1)
        t3 = time.time()
        #print('Concatenate = %d'%(t3-t2))
        right_columns = ['case_idxs_r', 'choice_idxs_r']
        y_wide_right = pd.DataFrame(y_wide_mtx, columns=right_columns)
        y_wide_right['chosen'] = 1
        t4 = time.time()
        #print('Create dataframe = %d'%(t4-t3))
        clms_to_drop = right_columns + ['case_idxs', 'choice_idxs']
        y_long = pd.merge(y_long_left, y_wide_right,  how='left', 
                  left_on=['case_idxs','choice_idxs'], 
                  right_on = right_columns).drop(clms_to_drop, axis=1).fillna(0)
        t5 = time.time()
        #print('Merging, dropping, and filling nans = %d'%(t5-t4))
        y_long['chosen'] = y_long['chosen'].astype('int')
        t6 = time.time()
        #print('Changing data type = %d'%(t6-t5))
        return y_long
    #---------------------------------------------------------------------------
    def compute_choices(self, data, true_params_dict):
        [choices, ll] = self.compute_choices_and_likelihood(data=data,
                                true_params_dict=true_params_dict,
                                choices_y=None)
        return choices
    #---------------------------------------------------------------------------
    def compute_likelihood(self, data, true_params_dict, choices_y=None):
        [choices, ll] = self.compute_choices_and_likelihood(data=data,
                                true_params_dict=true_params_dict,
                                choices_y=choices_y)
        return ll
    #---------------------------------------------------------------------------
    def compute_choices_and_likelihood(self, 
                                       data, 
                                       true_params_dict,
                                       choice_result_format='long',
                                       step_size=None):
        self.task='prediction'
        # logsums: order is important (top-to-bottom & left-to-right)
        p_vectors = self.convert_param_dic_to_vectors(true_params_dict)
        logsum_parameters, constants, coefficients = p_vectors

        X, y, case_choice_idxs = self.parse_data(data)
        self.covariates = X.values
        self.case_choice_idxs = case_choice_idxs.values
        np_dtype = np.float32 if self.dtype==tf.float32 else np.float64
        graph = self.build_graph()
        sess = tf.Session(graph=self.graph)
        sess.run(self.init_op)
        fd = {self.covariates_t: self.covariates,
              self.case_choice_idxs_t: self.case_choice_idxs,
              self.choices_y_t: [[-1.0,-1.0]],
              self.n_cases_per_batch_t: self.n_cases,
              self.logsums_t: logsum_parameters.astype(np_dtype),
              self.coefficients_t: coefficients.astype(np_dtype),
              self.constants_t: constants.astype(np_dtype),
              #self.gamma_t: np_dtype(gamma),
              self.learning_rate: -1.0,
              self.logsum_parameters_reg: -1.0}

        if self.pin_data_to_gpu:
            fd[self.covariates_t] = None
            fd[self.case_choice_idxs_t] = None
            fd[self.choices_y_t] = None

        choices_p_wide = np.array([]).reshape(-1,1)
        choices_p_long = pd.DataFrame()
        ll_epoch = 0
        n_steps = 0
        unq_case_idxs = case_choice_idxs['case_idxs'].unique()
        original_step_size = step_size
        step_size = self.n_cases if step_size is None else min(self.n_cases, step_size)
        for step_start_index in np.arange(0, self.n_cases, step_size):
            if step_size < self.n_cases:
                step_end_index = min(step_start_index + step_size, self.n_cases)
                batch_case_idxs = unq_case_idxs[step_start_index: step_end_index]
                batch_cace_choice_idxs = np.flatnonzero(case_choice_idxs['case_idxs'].isin(batch_case_idxs))
                self.n_cases_per_batch = len(batch_case_idxs)
                fd[self.covariates_t] = self.covariates[batch_cace_choice_idxs,]
                batch_case_choice_idxs = self.case_choice_idxs[batch_cace_choice_idxs,]
                batch_case_choice_idxs[:,0] = pd.match(batch_case_choice_idxs[:,0], batch_case_idxs)
                fd[self.case_choice_idxs_t] = batch_case_choice_idxs
                fd[self.n_cases_per_batch_t] = self.n_cases_per_batch
            if y is None:
                try:
                    cp = sess.run(self.choice_probabilities, 
                                           feed_dict=fd)
                except Exception as e:
                    if 'Dst tensor is not initialized' in str(e):
                        print('Error: unable to compute choice probabilities')
                        print('This could be a memory limitation problem')
                        if original_step_size is None:
                            print("You might overcome this by dividing the data into smaller sizes via the argument 'step_size' (e.g., 1000)")
                        else:
                            print("You might overcome this by dividing the data into smaller sizes than the one provided step_size=%d"%(step_size))
                    return
                cp = cp/cp.sum(axis=1)[:,None]
                choices_p_wide_batch = np.array([np.random.choice(np.arange(0, len(p)), 1, replace=False, p=p) for p in cp]).reshape(-1,1)
            else:
                choices_p_wide_batch = y[step_start_index:step_end_index].reshape(-1,1)

            cmi_batch = np.stack((np.arange(choices_p_wide_batch.shape[0]), 
                                  np.array(choices_p_wide_batch.reshape(-1))), axis=-1)

            choices_p_wide = np.concatenate((choices_p_wide, choices_p_wide_batch)) 
            fd[self.choices_y_t] = cmi_batch
            ll_batch = sess.run(self.loglikelihood, feed_dict=fd)
            n_steps += 1
            ll_epoch += ll_batch

        choices_p = choices_p_wide 
        if choice_result_format=='long':
            choices_p = self.covert_y_wide_2_long(choices_p_wide, data, case_choice_idxs)
        return [choices_p, ll_epoch]
    #---------------------------------------------------------------------------
    def get_opt_params(self):
        s, c, p = self.sess.run([self.logsums_t, self.constants_t, self.coefficients_t])
        return list(s) + list(c) + list(p)
    #---------------------------------------------------------------------------
    def save_params(self, out_fname, true_params_dict=None):
        esimated_params = self.get_estimated_parameters(true_params_dict)
        if esimated_params is not None and out_fname is not None:
            esimated_params.to_csv(out_fname, index=False)
    #---------------------------------------------------------------------------
    def save_tree(self, out_fname):
        if out_fname is not None:
            out_file = open(out_fname, 'w')
            for level in range(1, self.n_levels+1):
                node_names = ','.join(self.nltree_dict['l'+str(level)])
                out_file.write(node_names + '\n')
            out_file.close()
    #---------------------------------------------------------------------------
    def param_vectors_to_dict(self, logsums, constants, coefficients):
        param_dict = {}
        for logsum_name in self.logsum_names:
            param_dict[logsum_name] = logsums[self.logsum_names2indices[logsum_name]]    
        for const_name in self.constant_names:
            param_dict[const_name] = constants[self.constant_names2indices[const_name]]
        for coef_name in self.coefficient_names:
            param_dict[coef_name] = coefficients[self.coefficient_names2indices[coef_name]-1]  
        return param_dict
    #---------------------------------------------------------------------------
    def get_estimated_parameters(self, true_params_dict=None):
        np.seterr(divide='ignore', invalid='ignore')
        if self.est_parameters_dict is None:
            print("ERROR: Model's parameters have not been estimated yet. Run 'fit()' first to estimate them.")
            return
        param_types = []
        param_names = []
        param_est_vals = []
        param_true_vals = []
        # Logsums
        idx = 0
        for logsum_name in self.logsum_names:
            param_types.append('Logsum')
            param_names.append(logsum_name)
            param_est_vals.append(self.est_parameters_dict[logsum_name])
            if true_params_dict is not None and logsum_name in true_params_dict:
                param_true_vals.append(true_params_dict[logsum_name])
            else:
                param_true_vals.append(None)
        # Constants
        for const_name in self.constant_names:
            param_types.append('Constant')
            param_names.append(const_name)
            param_est_vals.append(self.est_parameters_dict[const_name])
            if true_params_dict is not None and const_name in true_params_dict:
                param_true_vals.append(true_params_dict[const_name])
            else:
                param_true_vals.append(None)
        # Coefficients
        for coefficient_name in self.coefficient_names:
            param_types.append('Coefficient')
            param_names.append(coefficient_name)
            param_est_vals.append(self.est_parameters_dict[coefficient_name])
            if true_params_dict is not None and coefficient_name in true_params_dict:
                param_true_vals.append(true_params_dict[coefficient_name])
            else:
                param_true_vals.append(None)
        summary_df = pd.DataFrame()
        summary_df['param_type'] = param_types
        summary_df['param_name'] = param_names
        summary_df['param_est_val'] = param_est_vals
        if true_params_dict is not None:
            summary_df['param_true_val'] = param_true_vals
            summary_df['relative_perc_error'] = summary_df[['param_est_val', 'param_true_val']].apply(lambda x: 100*np.abs((x[0]-x[1])/x[1]), axis=1)
        return summary_df
    #---------------------------------------------------------------------------
    def debug(self, 
               data, 
               true_params_dict,
               var_to_debug,
               task=None):
        self.task='prediction'
        p_vectors = self.convert_param_dic_to_vectors(true_params_dict)
        logsum_parameters, constants, coefficients = p_vectors
        X, choices_y, case_choice_idxs = self.parse_data(data)
        self.covariates = X.values
        self.case_choice_idxs = case_choice_idxs.values
        if choices_y is None:
            choices_y = [[-1, -1.0]]
        else: 
            choices_y = np.stack((np.arange(choices_y.shape[0]), 
                                               np.array(choices_y.reshape(-1))), axis=-1)
        np_dtype = np.float32 if self.dtype==tf.float32 else np.float64
        graph = self.build_graph()
        sess = tf.Session(graph=self.graph)
        sess.run(self.init_op)
        fd = {self.covariates_t: self.covariates,
              self.case_choice_idxs_t: self.case_choice_idxs,
              self.choices_y_t: choices_y,
              self.n_cases_per_batch_t: self.n_cases,
              self.logsums_t: logsum_parameters.astype(np_dtype),
              self.coefficients_t: coefficients.astype(np_dtype),
              self.constants_t: constants.astype(np_dtype),
              #self.gamma_t: np_dtype(gamma),
              self.learning_rate: 0.01,
              self.logsum_parameters_reg: -1.0}

        if self.pin_data_to_gpu:
            fd[self.covariates_t] = None
            fd[self.case_choice_idxs_t] = None
            fd[self.choices_y_t] = None

        #print(fd)
        #print(var_to_debug)
        if type(var_to_debug) is list:
            var_to_debug_ = []
            for vi in var_to_debug:
                var_to_debug_.append(eval('self.' + vi))
        else:
            var_to_debug_ = eval('self.' + var_to_debug)
        var_vals = sess.run(var_to_debug_, feed_dict=fd)
        if type(var_vals) is list:
            res_dic = {}
            for i, rkey in enumerate(var_to_debug):
                res_dic[rkey] = var_vals[i]
        else:
            res_dic = var_vals
        return res_dic
    #---------------------------------------------------------------------------
    def convert_tree_dict2expression(self, nltree):
        """ 
        Converts tree dictionary to text. 
        Example:
        Input = nltree:
            'l0': 'Root'
            'l1': ['L1C1', 'L1C2']
            'l2': ['L1C1_L2C1', 'L1C1_L2C2', 'L1C2_L2C3', 'L1C2_L2C4']
        Output = tree_expression:
            '((L1C1_L2C1,L1C1_L2C2)L1C1,(L1C2_L2C3,L1C2_L2C4)L1C2)Root;'
        """
        def find_children(parent, possible_children):
            children = []
            for possible_child in possible_children:
                if parent in possible_child:
                    children.append(possible_child)
            return children
        nlevels = len(nltree) - 1
        ndic = dict()
        ndic['l'+str(nlevels)] = collections.OrderedDict()
        for node in nltree['l'+str(nlevels)]:
            ndic['l'+str(nlevels)][node] = node
        for level in range(nlevels-1, 0, -1):
            ndic['l'+str(level)] = collections.OrderedDict()
            for node in nltree['l'+str(level)]:
                node_children = find_children(node, nltree['l'+str(level+1)])
                formatted_children = [ndic['l'+str(level+1)][nc] for nc in node_children]
                ndic['l'+str(level)][node] = '(' + ','.join(formatted_children) + ')' + node
        tree_expression= '(' + ','.join([ndic['l1'][k] for k in ndic['l1']]) + ')Root;'
        return tree_expression
    #---------------------------------------------------------------------------
    def print_tree(self, show_internal_choice_name=True):
        try:
            from ete3 import Tree, faces, AttrFace, TreeStyle, TextFace
            # Convert a dictionary to string
            tree_expression = self.convert_tree_dict2expression(self.nltree_dict)
            ete_tree = Tree(tree_expression, format=1)
            tree_image_in_text = ete_tree.get_ascii(show_internal=show_internal_choice_name)
            print(tree_image_in_text)
            return tree_image_in_text
        except:
            print('UNABLE to print the choice tree. Please install the ETE Toolkit')
            print('from http://etetoolkit.org/')
            print('Using PIP:')
            print('pip install --upgrade ete3')
            print('Using Conda:')
            print('conda install -c etetoolkit ete3')
            return None
    #---------------------------------------------------------------------------
    def draw_tree(self, file_name=None, rotation=0, fsize=8, 
                  name_position="branch-right", 
                  show_internal_choice_name=True,
                  w=None, h=None, units='px', dpi=90):
        try:
            os_name = platform.system()
            from ete3 import Tree, faces, AttrFace, TreeStyle, TextFace
            # Convert a dictionary to string
            tree_expression = self.convert_tree_dict2expression(self.nltree_dict)
            ete_tree = Tree(tree_expression, format=1)

            def my_layout(node):
              # Add name label to all nodes
              if node.is_leaf():
                  #F = TextFace(node.name, tight_text=True, fsize=fsize, ftype='Courier')
                  F = F = AttrFace("name", ftype='Courier', fsize=fsize, fgcolor='black')
                  faces.add_face_to_node(F, node, column=0, position=name_position)
              else:
                  #F = TextFace(node.name, tight_text=True, fsize=fsize, ftype='Courier')
                  F = AttrFace("name", ftype='Courier', fsize=fsize, fgcolor='black')
                  faces.add_face_to_node(F, node, column=0, position='branch-top')
            ts = TreeStyle()
            ts.rotation = rotation
            ts.show_scale = False
            if os_name == 'Linux':
                os.environ["QT_QPA_PLATFORM"] = "offscreen"
                print('WARNING: Tree labels may be invisible on Linux servers without GUI.')
                print('This is a limitation in the tree rendering engine ETE as of November 2017.')
                print('Use the print_tree() function to display it in text,')
                print('or run on a Windows machine to draw/save the tree as an image.\n')
                print('Action: hide internal choice names and print the tree in text:')
                print(ete_tree.get_ascii(show_internal=True))
                show_internal_choice_name=False

            ts.show_leaf_name = True
            if show_internal_choice_name:
                ts.show_leaf_name = False
                ts.layout_fn = my_layout
            if file_name is not None:
                ete_tree.render(file_name, w=w, units=units, h=h, dpi=dpi, tree_style=ts)
            return ete_tree.render('%%inline', w=w, units=units, h=h, dpi=dpi, tree_style=ts)
        except:
            print('UNABLE to print the choice tree. Please install the ETE Toolkit')
            print('from http://etetoolkit.org/')
            print('Using PIP:')
            print('pip install --upgrade ete3')
            print('Using Conda:')
            print('conda install -c etetoolkit ete3')
            return None
