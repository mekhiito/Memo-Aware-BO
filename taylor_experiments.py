from json_reader import read_json
from functions.processing_funcs import get_gen_bounds, get_dataset_bounds, get_initial_data, normalize, standardize, unnormalize, unstandardize
from acquisition_funcs.EEIPU.EEIPU import EEIPU
from functions.iteration_functions import get_gp_models, get_multistage_cost_models, get_inv_cost_models
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import IdentityMCObjective
import torch
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
def taylor_trial(trial_number, acqf, params=None):
    print("I am here...")
    # trial_logs = read_json('logs')
    bound_list = read_json('bounds')
    chosen_functions, h_ind, total_budget = params['obj_funcs'], params['h_ind'], params['total_budget']
    #input bounds are the bounds within which to generate data
    input_bounds = get_gen_bounds(h_ind, bound_list, funcs=chosen_functions)
    #get 1000 data points
    acqf = 'EEIPU'
    X, Y, C, C_inv = get_initial_data(params['n_init_data'], bounds=input_bounds, seed=trial_number*10000, acqf=acqf, params=params)
    params['n_init_data'] = X.shape[0]
    print(f'Initial Data has {X.shape} points')
    #bounds for normalizing and standardizing
    bounds = get_dataset_bounds(X, Y, C, C_inv, input_bounds)
    train_x = normalize(X, bounds=bounds['x_cube'])
    train_y = standardize(Y, bounds['y'])
    #training cost GPs and inv cost GP
    iter = 0
    mll, gp_model = get_gp_models(train_x, train_y, iter, params=params)
    cost_mll, cost_gp = get_multistage_cost_models(train_x, C, iter, params['h_ind'], bounds, acqf = acqf)
    inv_cost_mll, inv_cost_gp = get_inv_cost_models(train_x, C_inv, iter, params['h_ind'], bounds, acqf = acqf)

    #cost_sampler is used for MC sampling
    cost_sampler = SobolQMCNormalSampler(sample_shape=params['cost_samples'], seed=params['rand_seed'])
    acqf = EEIPU(acq_type=acqf, model=gp_model, cost_gp=cost_gp, inv_cost_gp=inv_cost_gp, best_f=train_y.max(),
                        cost_sampler=cost_sampler, acq_objective=IdentityMCObjective(), unstandardizer=unstandardize,
                        unnormalizer=unnormalize, bounds=bounds, eta=None,
                        consumed_budget=0, iter=iter, params=params)
    #estimating expected inverse cost using the current method
    exp_inv_cost_eeipu = acqf.compute_expected_inverse_cost(train_x[:, None, :])
    print(exp_inv_cost_eeipu.shape)
    exp_inv_cost_taylor = acqf.compute_taylor_expansion(train_x[:, None, :])
    print(exp_inv_cost_taylor.shape)
    exp_inv_cost_gt = acqf.compute_ground_truth(train_x[:, None, :])
    print(exp_inv_cost_gt.shape)

    mse_eeipu = F.mse_loss(exp_inv_cost_gt, exp_inv_cost_eeipu)
    mse_taylor = F.mse_loss(exp_inv_cost_gt, exp_inv_cost_taylor)
    print("MSE of EEIPU: ", mse_eeipu.item())    
    print("MSE of Taylor: ", mse_taylor.item()) 
    
    return