from botorch.acquisition import ExpectedImprovement
from functions.processing_funcs import normalize, standardize, get_gen_bounds
from functions.iteration_functions import get_gp_models
from optimize_mem_acqf import optimize_acqf_by_mem
import torch

def EI_iteration(X, y, c, c_inv, bounds=None, acqf_str='', decay=None, iter=None, consumed_budget=None, params=None):
    
    train_x = normalize(X, bounds=bounds['x_cube'])
    train_y = standardize(y, bounds['y'])
    
    mll, gp_model = get_gp_models(train_x, train_y, iter, params=params)
    
    norm_bounds = get_gen_bounds(params['h_ind'], params['normalization_bounds'], bound_type='norm')
    
    acqf = ExpectedImprovement(model=gp_model, best_f=train_y.max())
    
    new_x, n_memoised, acq_value = optimize_acqf_by_mem(
        acqf=acqf, acqf_str=acqf_str, bounds=bounds['x'], 
        iter=iter, params=params, seed=params['rand_seed'])
    
    E_c, E_inv_c, E_y = [0], torch.tensor([0]), 0
    # new_x = unnormalize(new_x, bounds=bounds['x_cube'])
    
    return new_x, n_memoised, E_c, E_inv_c, E_y, acq_value