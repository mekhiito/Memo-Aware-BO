from json_reader import read_json
from functions import generate_input_data, generate_ei_input_data, F, Cost_F, get_gen_bounds, get_dataset_bounds
from single_iteration import bo_iteration
from acquisition_funcs.EEIPU import EEIPU
import numpy as np
import torch
import csv
from typing import Iterable

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MS_ACQFS = ['EEIPU', 'MS_CArBO', 'LaMBO', 'MS_BO']

def taylor_trial(trial_number, acqf, params=None):

    # trial_logs = read_json('logs')
    bound_list = read_json('bounds')
    
    chosen_functions, h_ind, total_budget = params['obj_funcs'], params['h_ind'], params['total_budget']
    #input bounds are the bounds within which to generate data
    input_bounds = get_gen_bounds(h_ind, bound_list, funcs=chosen_functions)
    #get 1000 data points
    X, Y, C, C_inv = get_initial_data(params['n_init_data'], bounds=input_bounds, seed=trial_number*10000, acqf=acqf, params=params) 
    #bounds for normalizing and standardizing
    bounds = get_dataset_bounds(X, Y, C, C_inv, input_bounds)
    train_x = normalize(X, bounds=bounds['x_cube'])
    train_y = standardize(y, bounds['y'])
    
    #training cost GPs and inv cost GP
    iter = 0
    mll, gp_model = get_gp_models(train_x, train_y, iter, params=params)
    cost_mll, cost_gp = get_cost_models(train_x, C, iter, params['h_ind'], bounds, acqf_str = acqf)
    inv_cost_mll, inv_cost_gp = get_inv_cost_models(train_x, C_inv, iter, params['h_ind'], bounds, acqf_str = acqf)

    #cost_sampler is used for MC sampling
    cost_sampler = SobolQMCNormalSampler(sample_shape=params['cost_samples'], seed=params['rand_seed'])
    acqf = EEIPU(acq_type=acqf, model=gp_model, cost_gp=cost_gp, inv_cost_gp=inv_cost_gp, best_f=train_y.max(),
                        cost_sampler=cost_sampler, acq_objective=IdentityMCObjective(), unstandardizer=unstandardize,
                        unnormalizer=unnormalize, bounds=bounds, eta=None,
                        consumed_budget=0, iter=iter, params=params)
    #estimating expected inverse cost using the current method
    exp_inv_cost_eeipu = acqf.compute_expected_inverse_cost()
    print(exp_inv_cost_eeipu)
        
    return