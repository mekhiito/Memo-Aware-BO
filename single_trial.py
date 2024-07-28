from json_reader import read_json
from functions.processing_funcs import get_gen_bounds, get_dataset_bounds, get_initial_data
from functions.synthetic_functions import F, Cost_F
from functions.iteration_functions import iteration_logs
import torch
from time import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MS_ACQFS = ['EEIPU', 'LaMBO', 'MS_CArBO', 'MS_BO']

def bo_trial(trial_number, acqf, bo_iter_function, params=None):

    trial_logs = read_json('logs')
    bound_list = read_json('bounds')
    
    chosen_functions, h_ind = params['obj_funcs'], params['h_ind']

    input_bounds = get_gen_bounds(h_ind, bound_list, funcs=chosen_functions)
    
    X, Y, C, C_inv, init_cost = get_initial_data(params['n_init_data'], bounds=input_bounds, seed=trial_number*10000, acqf=acqf, params=params)
    
    params['budget_0'] = init_cost
    # params['total_budget'] = 10*init_cost
    
    best_f = Y.max().item()
    total_budget = params['total_budget']
    cum_cost = init_cost
    
    print(f'{acqf} at Trial {trial_number} used an initial cost of {init_cost:0,.2f} out of {total_budget:0,.2f}')
    iteration = 0
    count = 0
    init_eta, cool = 1, 0.9
    while cum_cost < total_budget:
        
        bounds = get_dataset_bounds(X, Y, C, C_inv, input_bounds)

        st = time()
        new_x, n_memoised, acq_value, count = bo_iter_function(X, Y, C, C_inv, bounds=bounds, acqf_str=acqf, decay=init_eta, iter=iteration, count=count, consumed_budget=cum_cost, params=params)
        en = time()
        init_eta *= cool

        iter_duration = en - st

        new_y = F(new_x, params).unsqueeze(-1)
        new_c = Cost_F(new_x, params)
        inv_cost = torch.tensor([1/new_c.sum()]).unsqueeze(-1)
        
        new_x, new_y, new_c, inv_cost = new_x.to(DEVICE), new_y.to(DEVICE), new_c.to(DEVICE), inv_cost.to(DEVICE)

        # if trial_number == 1:
        #     s = "BETTER than" if new_y.item() > best_f else "WORSE than"
        #     r = ":)" if new_y.item() > best_f else " :("
        #     m = f"Memoized {n_memoised} stages" if (new_y.item() > best_f and acqf == 'EEIPU' and n_memoised > 0) else ''
        #     print(f"{acqf} New Y {new_y.item():0,.2f} {s} {best_f:0,.2f} {r} Cost = {new_c.sum().item():0,.2f}. {m}\n\n")

        if acqf not in MS_ACQFS:
            new_c = new_c.sum(dim=1).unsqueeze(-1)
        
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

        # best_fs.append(best_f)

        eta = (params['total_budget'] - cum_cost) / (params['total_budget'] - params['budget_0'])

        log_params = {
            'acqf':acqf,
            'trial':trial_number,
            'iteration':iteration,
            'best_f':best_f,
            'sum_c_x':sum_stages,
            'cum_costs':cum_cost,
            'n_mem':n_memoised,
            'eta':eta,
            'duration':iter_duration,
            'n_prefs': params['n_prefs'],
        }

        iteration_logs(log_params)

    
    print(f'{acqf} Trial {trial_number} Final Data has {X.shape} datapoints with best_f {best_f:0,.2f}')

        
    return
