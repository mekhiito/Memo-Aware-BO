import wandb
from argparse import ArgumentParser
import time
import botorch
from copy import deepcopy
from acquisition_funcs.EEIPU.EEIPU_iteration import eeipu_iteration
from acquisition_funcs.EI.EI_iteration import ei_iteration
from acquisition_funcs.cost_aware_acqf.CArBO_iteration import carbo_iteration
from acquisition_funcs.cost_aware_acqf.EIPS_iteration import eips_iteration
from acquisition_funcs.MS_BO.MS_BO_iteration import msbo_iteration
from acquisition_funcs.LaMBO.LaMBO_iteration import lambo_iteration
from acquisition_funcs.LaMBO.LaMBO import LaMBO
from optimizer.optimize_acqf_funcs import optimize_acqf, _optimize_acqf_batch, gen_candidates_scipy, gen_batch_initial_conditions
from json_reader import read_json
from single_trial import bo_trial
import torch
import random
import numpy as np

def arguments():
    
    parser = ArgumentParser()
    parser.add_argument("--obj-funcs", nargs="+", help="Objective functions", default=["levy2", "hartmann3", "beale2"])
    parser.add_argument("--init-eta", type=float, help="Initial ETA", default=1)
    parser.add_argument("--decay-factor", type=float, help="Decay factor", default=1)
    parser.add_argument("--cost-types", nargs="+", help="Cost types", default=[1,2,3,2])
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

    params['EEIPU_iteration'] = eeipu_iteration
    params['LaMBO_iteration'] = lambo_iteration
    params['EI_iteration'] = ei_iteration
    params['CArBO_iteration'] = carbo_iteration
    params['MS_CArBO_iteration'] = carbo_iteration
    params['EIPS_iteration'] = eips_iteration
    params['MS_BO_iteration'] = msbo_iteration

    params['total_budget'] = 400
    
    params_per_stage = [2, 3, 2]

    params['h_ind'] = []
    i = 0
    for stage_params in params_per_stage:
        stage = []
        for j in range(stage_params):
            stage.append(i)
            i += 1
        params['h_ind'].append(stage)
    
    trial = args.trial_num

    if args.acqf == 'LaMBO':
        params['lambo_eta'] = 1
        lambo = LaMBO(params['lambo_eta'])
        lambo.lambo_trial(trial_number=trial, acqf=args.acqf, wandb=wandb, params=params)
    else:
        bo_trial(trial_number=trial, acqf=args.acqf, bo_iter_function=params[f'{args.acqf}_iteration'], wandb=wandb, params=params)
    
