import torch
import random
from botorch.test_functions import Beale, Branin, Hartmann, StyblinskiTang, Rosenbrock, Levy, Ackley,HolderTable, Michalewicz

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