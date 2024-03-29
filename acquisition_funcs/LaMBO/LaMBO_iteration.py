import torch
from botorch.acquisition import UpperConfidenceBound
from functions.iteration_functions import get_gp_models
from functions.processing_funcs import normalize, unnormalize, standardize, get_gen_bounds
from optimize_mem_acqf import optimize_acqf_by_mem

def lambo_iteration(X, y, c, c_inv, bounds=None, acqf_str='', decay=None, iter=None, consumed_budget=None, params=None):
    
    train_x = normalize(X, bounds=bounds['x_cube'])
    train_y = standardize(y, bounds['y'])
    
    mll, gp_model = get_gp_models(train_x, train_y, iter, params=params)
    
    norm_bounds = get_gen_bounds(params['h_ind'], params['normalization_bounds'], bound_type='norm')
    
    acqf = UpperConfidenceBound(model=gp_model, beta=0.2, maximize=True)
    
    new_x, n_memoised, acq_value = optimize_acqf_by_mem(
        acqf=acqf, acqf_str=acqf_str, bounds=norm_bounds, 
        iter=iter, params=params, seed=iter)
    
    new_x = unnormalize(new_x, bounds=bounds['x_cube'])
    
    return new_x, n_memoised, acq_value