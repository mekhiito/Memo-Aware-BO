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

def get_initial_data(n, bounds=None, seed=0, acqf=None, params=None):
    # if acqf not in ['EEIPU', 'CArBO']:
    #     X = generate_ei_input_data(N=n, bounds=bounds, seed=seed, acqf=acqf, params=params)
    # else:
    X = generate_input_data(N=n, bounds=bounds, seed=seed, acqf=acqf, params=params)
    y = F(X, params).unsqueeze(-1)
    c = Cost_F(X, params)

    if acqf not in ['EEIPU', 'MS_CArBO']:
        c = c.sum(dim=1)
    return X, y, c

def print_iter_logs(trial_logs):

    print(f"Best F = {trial_logs['best f(x)'][-1]:>4.3f}", end = '\t\t')
    print(f"Predicted Y = {trial_logs['f^(x)'][-1]:>4.3f}", end = '\t\t')
    print(f"f(x) = {trial_logs['f(x)'][-1]:>4.3f}", end = '\t\t')
    print(f"c(x) = [" + ", ".join('{:.3f}'.format(val) for val in trial_logs['c(x)'][-1]) + "]", end = '\t\t')
    print(f"E[c(x)] = [" + ", ".join('{:.3f}'.format(val) for val in trial_logs['E[c(x)]'][-1]) + "]", end = '\t\t')
    print(f"sum(c(x)) = [{trial_logs['sum(c(x))'][-1]:>4.3f}]", end = '\t\t')
    print(f"sum(E[c(x)]) = {trial_logs['sum(E[c(x)])'][-1]:>4.3f}", end = '\t\t')
    print(f"Cum Costs = {trial_logs['cum(costs)'][-1]:>4.3f}", end = '\t\t')
    print(f"Inverse c(x) = {trial_logs['1/c(x)'][-1]:>4.3f}", end = '\t\t')
    print(f"E[inv_c(x)] = {trial_logs['E[1/c(x)]'][-1]:>4.3f}")
    print('\n')

def bo_trial(trial_number, acqf, wandb, params=None):

    trial_logs = read_json('logs')
    bound_list = read_json('bounds')

    params['total_budget'] = 8000
    params['budget_0'] = 2500
    
    chosen_functions, h_ind, total_budget = params['obj_funcs'], params['h_ind'], params['total_budget']

    input_bounds = get_gen_bounds(h_ind, bound_list, funcs=chosen_functions)
    
    X, Y, C = get_initial_data(params['n_init_data'], bounds=input_bounds, seed=trial_number*10000, acqf=acqf, params=params)
    C_inv = 1/C.sum(dim=1)
    C_inv = C_inv.to(DEVICE)
    C_inv = C_inv.unsqueeze(-1)

    
    print(f'{acqf}\tTrial {trial_number}\t{X.shape[0]}')
    
    eta = params['warmup_eta']
    
    best_fs = [Y.max().item()]
    total_budget = params['total_budget']
    cum_cost = 0
    iteration = 0
    
    while cum_cost < total_budget:
        
        bounds = get_dataset_bounds(X, Y, C, C_inv, input_bounds)
        new_x, n_memoised, E_c, E_inv_c, y_pred, acq_value = bo_iteration(X, Y, C, C_inv, bounds=bounds, acqf_str=acqf, decay=eta, iter=iteration, consumed_budget=cum_cost, params=params)
        
        new_x = new_x.to(DEVICE)
        new_y = F(new_x, params).unsqueeze(-1)
        new_y = new_y.to(DEVICE)
        new_c = Cost_F(new_x, params)
        new_c = new_c.to(DEVICE)
        inv_cost = torch.tensor([1/new_c.sum()]).unsqueeze(-1)
        inv_cost = inv_cost.to(DEVICE)

        if acqf not in ['EEIPU', 'MS_CArBO']:
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
        
        log = dict(
            acqf=acqf,
            trial=trial_number,
            iteration=iteration,
            best_f=best_f,
            sum_c_x=sum_stages,
            cum_costs=cum_cost,
            # c_x=dict(zip(map(str,range(len(stage_cost_list))) ,stage_cost_list))
        )

        iteration += 1

        dir_name = f"syn_logs_"
        csv_file_name = f"{dir_name}/{acqf}_trial_{trial_number}.csv"
        # Check if the file exists
        try:
            with open(csv_file_name, 'r') as csvfile:
                reader = csv.reader(csvfile)
                fieldnames = next(reader)  # Read the headers in the first row

        except FileNotFoundError:
            # If file does not exist, create it and write headers
            fieldnames = ['acqf', 'trial', 'iteration', 'best_f', 'sum_c_x', 'cum_costs']
            with open(csv_file_name, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        # Append data
        with open(csv_file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(log)
        
        # wandb.log(log)

        best_fs.append(best_f)

        
    return