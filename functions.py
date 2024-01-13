import torch
import copy
import random
import math
from collections import deque
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, RQKernel, MaternKernel, PeriodicKernel, ScaleKernel, AdditiveKernel, ProductKernel
from botorch.test_functions import Beale, Branin, Hartmann, EggHolder, StyblinskiTang, Rosenbrock, Levy, Shekel, Ackley,HolderTable, Michalewicz

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SYNTHETIC_FUNCTIONS = {
    # Stage 1
    'branin2': Branin(negate=True, bounds=[[0,10], [0,10]]),
    'michale2': Michalewicz(negate=True),
    'styblinski2': StyblinskiTang(dim=2, negate=True),
    'beale2': Beale(negate=True),
    
    # Stage 2
    'ackley3': Ackley(dim=3, negate=True),
    'hartmann3':Hartmann(dim=3, negate=True),
    'styblinski3': StyblinskiTang(dim=3, negate=True),
    
    # Stage 3
    'rosenbrock2': Rosenbrock(negate=True),
    'levy2': Levy(negate=True),
    'holdertable2': HolderTable(negate=True)
}

def normalize(data, bounds=None):
    data_ = data + 0
    dims = data_.shape[1]
    
    for dim in range(dims):
        mn, mx = bounds[0][dim].item(), bounds[1][dim].item()
        data_[:, dim] = (data_[:,dim] - mn) / (mx - mn)
    return data_ 

def unnormalize(data, bounds=None):
    data_ = data + 0
    dims = data_.shape[1]
    for dim in range(dims):
        mn, mx = bounds[0][dim].item(), bounds[1][dim].item()
        data_[:, dim] = (data_[:, dim] * (mx-mn)) + mn
    return data_ 

def normalize_cost(data, params):
    mn, mx = 0, 1000
    try:
        assert data.min().item() > 1 and data.max().item() < 1000
    except:
        print(f"BEFORE NORMALIZING COSTS, EXCEPTION RAISED BECAUSE THE MINIMUM DATAPOINT IS = {data.min().item()}, MAXIMUM FOR SOME REASON IS = {data.max().item()}, SHAPE IS = {data.shape}, AND NUMBER OF NANS IS {torch.isnan(data.view(-1)).sum().item()}")
    
    data = (data - mn) / (mx-mn)
    alpha, eps = params['alpha'], params['norm_eps']
    data = alpha * data + eps
    try:
        assert data.min().item() > 1
    except:
        print(f"AFTER NORMALIZING COSTS, EXCEPTION RAISED BECAUSE THE MINIMUM DATAPOINT IS = {data.min().item()}, MAXIMUM FOR SOME REASON IS = {data.max().item()}, SHAPE IS = {data.shape}, AND NUMBER OF NANS IS {torch.isnan(data.view(-1)).sum().item()}")
    return data

def standardize(data, bounds=None):
    data_ = data + 0
    mean, std = bounds[0].item(), bounds[1].item()
    data_ = (data_ - mean) / std
    return data_ 

def unstandardize(data, bounds=None):
    data_ = data + 0
    mean, std = bounds[0].item(), bounds[1].item()
    data_ = (data_ * std) + mean
    return data_ 

def logistic(x, params): # logistic function
    log = ( 1./ (1 + torch.exp(-10*x))  )
    return log 

def sin(x, params): # sin
    sine = torch.sin(x)
    return sine 

def cos(x, params): # cosine
    cosine = torch.cos(x)
    return cosine 

def poly(x, params): # polynomial function
    nomial = x**params['power']
    return nomial 

def apply(f, x, params={}, synthetic = False):
    if synthetic:
        val = (params['scale'] * f(x) + params['shift'])**params['power']
    else:
        val = (params['scale'] * f(x, params) + params['shift'])**params['power']
    return val 

def stage_cost_func(X, dims=3, stage=1):
    funcs = [cos, sin, logistic, poly]
    strs = ['cos', 'sin', 'log', 'pol']
    cost = 0
    used_costs = []
    for d in range(dims):
        random.seed(stage*100 + d)
        idx = random.randint(0,3)
        scale, shift = random.randint(10, 50), random.randint(50, 100)
        used_costs.append(strs[idx])

        cost += apply(funcs[idx], X[:,d], {'scale':scale, 'shift':shift, 'power':2})
    cost = cost/(dims*500)

    assert cost.min().item() > 0   
    cost = cost.unsqueeze(-1)
    return cost, used_costs

def cost3D(X, ctype=1):
    if ctype==1:
        cost = (apply(logistic, X[:,0], {'scale':20, 'shift':50, 'slope':4}) + apply(sin, X[:,1], {'scale':30, 'shift':40}) + apply(cos, X[:,2], {'scale':5, 'shift':30}))
    elif ctype==2:
        cost =  (apply(sin, X[:,0], {'scale':20, 'shift':50}) + apply(logistic, X[:,2], {'slope':8,'scale':15,'shift':5}))
    elif ctype==3:
        cost = (apply(logistic, X[:,0], {'scale':22, 'shift':15, 'slope':3}) + apply(sin, X[:,1], {'scale':5, 'shift':10}) + apply(cos, X[:,2], {'scale':20, 'shift':25}))
    else:
        raise ValueError('Only cost types 1 to 3 acceptable')
    assert cost.min().item() > 0
    cost = cost.unsqueeze(-1)
    return cost
    
def get_gen_bounds(param_idx, func_bounds, funcs=None, bound_type=''):
    lo_bounds, hi_bounds = [], []
    
    for stage in range(len(param_idx)):
        if bound_type == 'norm':
            f_bounds = func_bounds
        else:
            f = funcs[stage]
            f_bounds = func_bounds[f]
        stage_size = len(param_idx[stage])
        
        lo_bounds += [f_bounds[0]]*stage_size
        hi_bounds += [f_bounds[1]]*stage_size
    
    bounds = torch.tensor([lo_bounds, hi_bounds], device=DEVICE, dtype=torch.double)
    return bounds

def get_dataset_bounds(X, Y, C, C_inv, gen_bounds):
    bounds = {}
    bounds['x'] = gen_bounds + 0.

    x_cube_bounds = [[], []]
    for i in range(X.shape[1]):
        x_cube_bounds[0].append(X[:,i].min().item())
        x_cube_bounds[1].append(X[:,i].max().item())
    bounds['x_cube'] = torch.tensor(x_cube_bounds, device=DEVICE)

    bounds['y'] = torch.tensor([[Y.mean().item()], [Y.std().item()]], device=DEVICE, dtype=torch.double)
    invc = C_inv + 0
    invc = torch.log(invc)
    bounds['1/c'] = torch.tensor([[invc.mean().item()], [invc.std().item()]], device=DEVICE, dtype=torch.double)

    std_c_bounds = [[], []]
    for s in range(C.shape[1]):
        stage_costs = C[:,s]
        log_sc = torch.log(stage_costs)
        std_c_bounds[0].append(log_sc.mean().item())
        std_c_bounds[1].append(log_sc.std().item())
    bounds['c'] = torch.tensor(std_c_bounds, device=DEVICE)
    
    return bounds

def get_random_observations(N=None, bounds=None, seed=None):
    torch.manual_seed(seed=seed)
    X = None
    # Generate initial training data, one dimension at a time
    for dim in range(len(bounds[0])):
        lo_bounds, hi_bounds = bounds[0][dim], bounds[1][dim]
        
        if torch.is_tensor(X):
            temp = torch.distributions.uniform.Uniform(lo_bounds, hi_bounds).sample([N, 1])
            X = torch.cat((X, temp), dim=1)
        else:
            X = torch.distributions.uniform.Uniform(lo_bounds, hi_bounds).sample([N, 1])
    X = X.to(DEVICE)
    return X

def remove_row(x, idx):
    return torch.cat((x[:idx], x[idx+1:]))

def find_costliest_cand(costs):
    return torch.argmax(costs)

def find_closest_cand(X, cand):
    distances = torch.matmul(cand, X.T)
    distance_from_set, _ = torch.max(distances,dim=1)
    closest = torch.argmax(distance_from_set)
    return closest

def generate_input_data(N=None, bounds=None, seed=0, acqf=None, params=None):
    
    X = []
    init_cost = 1
    i=0
    while init_cost < params['budget_0']:
        torch.manual_seed(seed=seed+init_cost)
        candidates = get_random_observations(N, bounds, seed=seed+init_cost)
        costs = Cost_F(candidates, params)
        while candidates.shape[0] > 1:
            expensive = find_costliest_cand(costs.sum(dim=1))
            candidates = remove_row(candidates, expensive)
            costs = remove_row(costs, expensive)

            if X != [] and candidates.shape[0] > 1:
                closest = find_closest_cand(X, candidates)
                candidates = remove_row(candidates, closest)
                costs = remove_row(costs, closest)
        
        cand_cost = costs.sum().item()
        init_cost += cand_cost
        
        X = candidates if X == [] else torch.cat((X, candidates))
        i += 1
    return X

def generate_ei_input_data(N=None, bounds=None, seed=0, acqf=None, params=None):
    torch.manual_seed(seed=seed)
    init_cost = 0
    X = []
    while init_cost < params['budget_0']:
        candidate = get_random_observations(1, bounds)
        cand_cost = Cost_F(candidate, params)
        init_cost += cand_cost.sum().item() #sum(cost.item() for cost in cand_cost)
        X = candidate if X == [] else torch.cat((X, candidate))
    return X

def F(X, params):
    
    funcs = params['obj_funcs']
    param_idx = params['h_ind']
    n_stages = len(param_idx)
    
    F = 0

    for stage in range(n_stages):
        stage_params = param_idx[stage]
        idx = stage_params[0]
        f = funcs[stage]
        obj = SYNTHETIC_FUNCTIONS[f]
        if '2' in f:
            F += obj(X[:, idx:idx+2])
        elif '3' in f:
            F += obj(X[:, idx:idx+3])
    return F.to(DEVICE)

def Cost_F(X, params):
    
    costs = []
    n_stages = len(params['h_ind'])
    used_costs = []
    for stage in range(n_stages):
        stage_c, used_f = stage_cost_func(X[:,params['h_ind'][stage]], dims=len(params['h_ind'][stage]), stage=stage)
        costs = stage_c if costs == [] else torch.cat([costs, stage_c], dim=1)
        used_costs += used_f
    # print(used_costs)
    costs = costs.to(DEVICE)
    return costs

def initialize_GP_model(X, y, params=None):
    X_, y_ = X + 0, y + 0
    gp_model = SingleTaskGP(X_, y_).to(X_)
    gp_model = gp_model.to(DEVICE)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model).to(DEVICE)
    return mll, gp_model

def generate_prefix_pool(X, acqf, params):
    prefix_pool = []
    first_idx = params['n_init_data']

    if acqf not in ['EEIPU', 'EIPU-MEMO']:
        prefix_pool.append([])
        return prefix_pool
        
    for i, param_config in enumerate(X[first_idx:]):
        prefix = []
        n_stages = len(params['h_ind'])
        for j in range(n_stages - 1):
            stage_params = params['h_ind'][j]
            prefix.append(list(param_config[stage_params].cpu().detach().numpy()))
            prefix_pool.append(copy.deepcopy(prefix))
    
    random.shuffle(prefix_pool)
    
    # Constant complexity to append at beginning of list
    prefix_pool = deque(prefix_pool)
    prefix_pool.appendleft([])
    prefix_pool = list(prefix_pool)
    
    if len(prefix_pool) > params['prefix_thresh']:
        prefix_pool = prefix_pool[:params['prefix_thresh']]
    return prefix_pool

