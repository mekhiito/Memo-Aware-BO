import wandb
from argparse import ArgumentParser
import time
from copy import deepcopy
from EEIPU.EEIPU_iteration import EEIPU_iteration
from EI.EI_iteration import EI_iteration
from cost_aware_acqf.CArBO_iteration import CArBO_iteration
from cost_aware_acqf.EIPS_iteration import EIPS_iteration
import botorch
from optimizer.optimize_acqf_funcs import optimize_acqf, _optimize_acqf_batch, gen_candidates_scipy, gen_batch_initial_conditions
from json_reader import read_json
from single_trial import bo_trial
import torch
import random
import numpy as np

def arguments():
    parser = ArgumentParser()
    parser.add_argument("--obj-funcs", nargs="+", help="Objective functions", default=["beale2", "hartmann3", "beale2", "ackley3", "branin2"])
    parser.add_argument("--init-eta", type=float, help="Initial ETA", default=1)
    parser.add_argument("--decay-factor", type=float, help="Decay factor", default=1)
    parser.add_argument("--cost-types", nargs="+", help="Cost types", default=[1,2,3])
    parser.add_argument("--warmup-eta", type=float, help="Warm up", default=1e-2)
    parser.add_argument("--trial-num", type=int, help="Trial number")
    parser.add_argument("--exp-group", type=str, help="Group ID")
    parser.add_argument("--acqf", type=str, help="Acquisition function", choices=['EEIPU', 'EIPU', 'EIPU-MEMO', 'EI', 'CArBO', 'EIPS', 'MS_CArBO', 'MS_BO', 'LaMBO'])
    
    params:dict = read_json("params")
    
    args = parser.parse_args()
    
    args_dict = deepcopy(vars(args))
    args_dict.pop("trial_num")
    args_dict.pop("exp_group")
    params.update(args_dict)
    
    return args, params

if __name__=="__main__":
    args, params = arguments()
    trial = args.trial_num
    # wandb.init(
    #     entity="cost-bo",
    #     project="memoised-cost-aware-bo-organized",
    #     group=f"{args.exp_group}--acqf_{args.acqf}|-obj-func_{'-'.join(args.obj_funcs)}|-dec-fac_{args.decay_factor}"
    #             f"|init-eta_{args.init_eta}|-cost-typ_{'-'.join(list(map(str,args.cost_types)))}",
    #     name=f"{time.strftime('%Y-%m-%d-%H%M')}-trial-number_{trial}",
    #     config=params
    # )
    botorch.optim.optimize.optimize_acqf = optimize_acqf
    botorch.optim.optimize._optimize_acqf_batch = _optimize_acqf_batch
    botorch.generation.gen.gen_candidates_scipy = gen_candidates_scipy
    botorch.optim.initializers.gen_batch_initial_conditions = gen_batch_initial_conditions
    
    
    torch.manual_seed(seed=params['rand_seed'])
    np.random.seed(params['rand_seed'])
    random.seed(params['rand_seed'])
    botorch.utils.sampling.manual_seed(seed=params['rand_seed'])
    
    logs = read_json('logs')

    params['EEIPU_iteration'] = EEIPU_iteration
    params['EI_iteration'] = EI_iteration
    params['CArBO_iteration'] = CArBO_iteration
    params['EIPS_iteration'] = EIPS_iteration

    # Use this line to customize the initial budget_0 (only counted for the initial data generation)
    params['budget_0'] = 2500
    
    # Use this line to customize the total optimization budget used by the BO process
    params['total_budget'] = 8000

    # Use this line to customize the number of optimizable hyperparameters per stage for this synthetic experiment
    params_per_stage = [4, 6, 10, 7, 3]
    params['h_ind'] = []
    i = 0
    for stage_params in params_per_stage:
        stage = []
        for j in range(stage_params):
            stage.append(i)
            i += 1
        params['h_ind'].append(stage)
    # params['h_ind'] = [[0,1,2,3], [4,5,6,7,8,9], [10,11,12,13,14,15,16,17,18,19], [20,21,22,23,24,25,26], [27,28,29]]
    
    trial = args.trial_num

    if args.acqf == 'LaMBO':
        lambo_trial(trial_number=trial, acqf=args.acqf, wandb=wandb, params=params)
    else:
        bo_trial(trial_number=trial, acqf=args.acqf, iter_function=params[f'{args.acqf}_iteration'], wandb=wandb, params=params)
    
