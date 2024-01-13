from EEIPU import EEIPU

def bo_iteration(X, y, c, c_inv, bounds=None, acqf_str='', decay=None, iter=None, consumed_budget=None, params=None):
    
    train_x = normalize(X, bounds=bounds['x_cube'])
    train_y = standardize(y, bounds['y'])
    
    mll, gp_model = get_gp_models(train_x, train_y, iter, params=params)
    
    norm_bounds = get_gen_bounds(params['h_ind'], params['normalization_bounds'], bound_type='norm')
    
    prefix_pool = None
    if params['use_pref_pool']:
        prefix_pool = generate_prefix_pool(train_x, acqf_str, params)
    
    cost_mll, cost_gp = get_cost_models(train_x, c, iter, params['h_ind'], bounds, acqf_str)
    inv_cost_mll, inv_cost_gp = get_inv_cost_models(train_x, c_inv, iter, params['h_ind'], bounds, acqf_str)
        
    cost_sampler = SobolQMCNormalSampler(sample_shape=params['cost_samples'], seed=params['rand_seed'])
    acqf = EEIPU(acq_type=acqf_str, model=gp_model, cost_gp=cost_gp, inv_cost_gp=inv_cost_gp, best_f=train_y.max(),
                        cost_sampler=cost_sampler, acq_objective=IdentityMCObjective(), unstandardizer=unstandardize,
                        unnormalizer=unnormalize, bounds=bounds, eta=decay,
                        consumed_budget=consumed_budget, iter=iter, params=params)
    
    new_x, n_memoised, acq_value = optimize_acqf_by_mem(acqf=acqf, acqf_str=acqf_str, bounds=norm_bounds, iter=iter, prefix_pool=prefix_pool, params=params, seed=params['rand_seed'])
    
    E_c, E_inv_c, E_y = [0], torch.tensor([0]), 0
    E_c = acqf.compute_expected_cost(new_x)
    E_inv_c = acqf.compute_expected_inverse_cost(new_x[:, None, :])

    new_x = unnormalize(new_x, bounds=bounds['x_cube'])
    
    
    return new_x, n_memoised, E_c, E_inv_c, E_y, acq_value