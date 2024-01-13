from botorch.acquisition import UpperConfidenceBound

def lambo_iteration(X, y, c, c_inv, bounds=None, acqf_str='', decay=None, iter=None, consumed_budget=None, params=None):
    
    train_x = normalize(X, bounds=bounds['x_cube'])
    train_y = standardize(y, bounds['y'])
    
    mll, gp_model = get_gp_models(train_x, train_y, iter, params=params)
    
    norm_bounds = get_gen_bounds(params['h_ind'], params['normalization_bounds'], bound_type='norm')
    
    acqf = UpperConfidenceBound(model=gp_model, best_f=train_y.max(), maximize=True)
    
    new_x, n_memoised, acq_value = optimize_acqf_by_mem(acqf=acqf, acqf_str=acqf_str, bounds=norm_bounds, iter=iter, prefix_pool=None, params=params, seed=params['rand_seed'])
    
    E_c, E_inv_c, E_y = [0], torch.tensor([0]), 0
    new_x = unnormalize(new_x, bounds=bounds['x_cube'])
    
    return new_x, n_memoised, E_c, E_inv_c, E_y, acq_value