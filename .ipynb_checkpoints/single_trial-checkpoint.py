from json_reader import read_json
from functions import generate_input_data, generate_ei_input_data, F, Cost_F, get_gen_bounds, get_dataset_bounds
from LaMBO.MSET import MSET, Node
from single_iteration import bo_iteration
import numpy as np
from copy import deepcopy
import torch
import csv
from typing import Iterable

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MS_ACQFS = ['EEIPU', 'MS_CArBO', 'LaMBO', 'MS_BO']

def bo_trial(trial_number, acqf, bo_iter_function, wandb, params=None):

    trial_logs = read_json('logs')
    bound_list = read_json('bounds')
    
    chosen_functions, h_ind, total_budget = params['obj_funcs'], params['h_ind'], params['total_budget']

    input_bounds = get_gen_bounds(h_ind, bound_list, funcs=chosen_functions)
    
    X, Y, C, C_inv = get_initial_data(params['n_init_data'], bounds=input_bounds, seed=trial_number*10000, acqf=acqf, params=params)
    
    best_fs = [Y.max().item()]
    total_budget = params['total_budget']
    cum_cost = 0
    iteration = 0
    
    while cum_cost < total_budget:
        
        bounds = get_dataset_bounds(X, Y, C, C_inv, input_bounds)
        new_x, n_memoised, E_c, E_inv_c, y_pred, acq_value = bo_iter_function(X, Y, C, C_inv, bounds=bounds, acqf_str=acqf, decay=eta, iter=iteration, consumed_budget=cum_cost, params=params)
        
        new_y = F(new_x, params).unsqueeze(-1)
        new_c = Cost_F(new_x, params)
        inv_cost = torch.tensor([1/new_c.sum()]).unsqueeze(-1)
        
        new_x, new_y, new_c, inv_cost = new_x.to(DEVICE), new_y.to(DEVICE), new_c.to(DEVICE), inv_cost.to(DEVICE)

        if acqf not in MS_ACQFS:
            new_c = new_c.sum()
        
        X = torch.cat([X, new_x])
        Y = torch.cat([Y, new_y])
        C_inv = torch.cat([C_inv, inv_cost])
        C = torch.cat([C, new_c])
        
        best_f = Y.max().item()
        for stage in range(n_memoised):
            new_c[:,stage] = torch.tensor([params['epsilon']])

        # stage_cost_list = new_c.tolist()
        sum_stages = new_c.sum().item()        
        cum_cost += sum_stages

        iteration += 1
        
        # wandb.log(log)

        best_fs.append(best_f)

        iteration_logs(acqf, trial_number, iteration, best_f, sum_stages, cum_cost)

        
    return