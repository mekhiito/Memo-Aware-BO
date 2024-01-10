from json_reader import read_json
from functions import generate_input_data, generate_ei_input_data, F, Cost_F, get_gen_bounds, get_dataset_bounds
from LaMBO.MSET import MSET, Node
from single_iteration import bo_iteration
import numpy as np
from copy import deepcopy
import torch
import csv
from typing import Iterable
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_initial_data(n, bounds=None, seed=0, acqf=None, params=None):
    # if acqf not in ['EEIPU', 'CArBO']:
    #     X = generate_ei_input_data(N=n, bounds=bounds, seed=seed, acqf=acqf, params=params)
    # else:
    X = generate_input_data(N=n, bounds=bounds, seed=seed, acqf=acqf, params=params)
    y = F(X, params).unsqueeze(-1)
    c = Cost_F(X, params)

    if acqf not in ['EEIPU', 'MS_CArBO']:
        c = c.sum(dim=1)
    return X, y, c

def print_iter_logs(trial_logs):

    print(f"Best F = {trial_logs['best f(x)'][-1]:>4.3f}", end = '\t\t')
    print(f"Predicted Y = {trial_logs['f^(x)'][-1]:>4.3f}", end = '\t\t')
    print(f"f(x) = {trial_logs['f(x)'][-1]:>4.3f}", end = '\t\t')
    print(f"c(x) = [" + ", ".join('{:.3f}'.format(val) for val in trial_logs['c(x)'][-1]) + "]", end = '\t\t')
    print(f"E[c(x)] = [" + ", ".join('{:.3f}'.format(val) for val in trial_logs['E[c(x)]'][-1]) + "]", end = '\t\t')
    print(f"sum(c(x)) = [{trial_logs['sum(c(x))'][-1]:>4.3f}]", end = '\t\t')
    print(f"sum(E[c(x)]) = {trial_logs['sum(E[c(x)])'][-1]:>4.3f}", end = '\t\t')
    print(f"Cum Costs = {trial_logs['cum(costs)'][-1]:>4.3f}", end = '\t\t')
    print(f"Inverse c(x) = {trial_logs['1/c(x)'][-1]:>4.3f}", end = '\t\t')
    print(f"E[inv_c(x)] = {trial_logs['E[1/c(x)]'][-1]:>4.3f}")
    print('\n')

def bo_trial(trial_number, acqf, wandb, params=None):

    trial_logs = read_json('logs')
    bound_list = read_json('bounds')

    params['total_budget'] = 8000
    params['budget_0'] = 2500
    
    chosen_functions, h_ind, total_budget = params['obj_funcs'], params['h_ind'], params['total_budget']


    # FROM HERE
    if acqf != 'LaMBO':
        input_bounds = get_gen_bounds(h_ind, bound_list, funcs=chosen_functions)
        # else: build the tree and choose an arm
    
    X, Y, C = get_initial_data(params['n_init_data'], bounds=input_bounds, seed=trial_number*10000, acqf=acqf, params=params)
    C_inv = 1/C.sum(dim=1)
    C_inv = C_inv.to(DEVICE)
    C_inv = C_inv.unsqueeze(-1)
    # TO HERE SHOULD BE IN A LOOP TO GENERATE SEPARATE DATA PER LEAF

    
    print(f'{acqf}\tTrial {trial_number}\t{X.shape[0]}')
    
    eta = params['warmup_eta']
    
    best_fs = [Y.max().item()]
    total_budget = params['total_budget']
    cum_cost = 0
    iteration = 0
    
    while cum_cost < total_budget:

        if acqf == 'LaMBO':
        
            bounds, acq_value = get_dataset_bounds(X[arm_idx], Y[arm_idx], C[arm_idx], C_inv[arm_idx], arm.value)

            new_x, n_memoised, E_c, E_inv_c, y_pred = bo_iteration(X[arm_idx], Y[arm_idx], C[arm_idx], C_inv[arm_idx], bounds=bounds, acqf_str=acqf, decay=eta, iter=iteration, consumed_budget=cum_cost, params=params)
        else:
            bounds = get_dataset_bounds(X, Y, C, C_inv, input_bounds)
            new_x, n_memoised, E_c, E_inv_c, y_pred = bo_iteration(X, Y, C, C_inv, bounds=bounds, acqf_str=acqf, decay=eta, iter=iteration, consumed_budget=cum_cost, params=params)
        
        new_x = new_x.to(DEVICE)
        new_y = F(new_x, params).unsqueeze(-1)
        new_y = new_y.to(DEVICE)
        new_c = Cost_F(new_x, params)
        new_c = new_c.to(DEVICE)
        inv_cost = torch.tensor([1/new_c.sum()]).unsqueeze(-1)
        inv_cost = inv_cost.to(DEVICE)

        if acqf not in ['EEIPU', 'MS_CArBO']:
            new_c = new_c.sum()
        
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
        
        log = dict(
            acqf=acqf,
            trial=trial_number,
            iteration=iteration,
            best_f=best_f,
            sum_c_x=sum_stages,
            cum_costs=cum_cost,
            # c_x=dict(zip(map(str,range(len(stage_cost_list))) ,stage_cost_list))
        )

        iteration += 1

        dir_name = f"syn_logs_"
        csv_file_name = f"{dir_name}/{acqf}_trial_{trial_number}.csv"
        # Check if the file exists
        try:
            with open(csv_file_name, 'r') as csvfile:
                reader = csv.reader(csvfile)
                fieldnames = next(reader)  # Read the headers in the first row

        except FileNotFoundError:
            # If file does not exist, create it and write headers
            fieldnames = ['acqf', 'trial', 'iteration', 'best_f', 'sum_c_x', 'cum_costs']
            with open(csv_file_name, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        # Append data
        with open(csv_file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(log)
        
        # wandb.log(log)

        best_fs.append(best_f)

        
    return

def build_partitions(input_bounds, h_ind, n_stages):
    partitions = []
    for i in range(n_stages-1):
        stage_partition = []
        for stage_idx in h_ind[i]:
            lo, hi = input_bounds[0][stage_idx], input_bounds[1][stage_idx]
            mid = (lo + hi) / 2.0
            p = [[lo, mid], [mid, hi]]
            stage_partition.append(p)
        partitions.append(stage_partition)

    last_stage_partition = []
    for stage_idx in h_ind[-1]:
        lo, hi = input_bounds[0][stage_idx], input_bounds[1][stage_idx]
        p = [lo, hi]
        last_stage_partition.append(p)

    return partitions, last_stage_partitions

def get_pdf(n_leaves):
    unif_prob = 1.0/n_leaves
    probs = np.array([unif_prob for i in range(n_leaves)])
    return probs

def build_tree(partitions, depths, last_stage_partitions):
    root = Node(None, 0)
    mset = MSET(partitions, depths, last_stage_partition)
    
    left = mset.ConstructMSET(root, 0, 0, 1, [], [[], []])
    right = mset.ConstructMSET(root, 0, 1, 2, [], [[], []])
    root.add_child(left, right)

    mset.assign_leaf_ranges(root)

    return mset, root

def get_subtree_arms(root, prev_arm_idx, prev_h):
    
    node = copy.deepcopy(root)
    curr_depth = 0
    
    while curr_depth < prev_h:
        if node.left.leaf_ranges[0] <= prev_arm_idx <= node.left.leaf_ranges[1]:
            node = node.left
        elif node.right.leaf_ranges[0] <= prev_arm_idx <= node.right.leaf_ranges[1]:
            node = node.right
            
        curr_depth += 1

    return node.leaf_ranges

def select_arm(root, leaves, prev_h, prev_arm_idx, n_leaves):
    

    arm_choices = np.array([i for i in range(n_leaves)])
    valid_arm_idx = get_subtree_arms(root, prev_h, prev_arm_idx)

    valid_arm_choices = arm_choices[valid_arm_idx[0]:valid_arm_idx[1]+1]
    valid_probs = probs[valid_arm_idx[0]:valid_arm_idx[1]+1]
    
    arm_idx = random.choices(valid_arm_choices, weights=valid_probs)[0]

    return leaves[arm_idx], arm_idx

def update_loss_estimators(loss, root, arm_idx, sigma, eta, H, acq_value):
    loss[arm_idx][0] = acq_value
    node = copy.deepcopy(root)
    for height in range(1, H):
    
        if node.left.leaf_ranges[0] <= arm_idx <= node.left.leaf_ranges[1]:
            node = node.left
        elif node.right.leaf_ranges[0] <= arm_idx <= node.right.leaf_ranges[1]:
            node = node.right
    
        nominator = 0
        for leaf_idx in range(node.leaf_ranges[0], node.leaf_ranges[1]):
            nominator += (probs[leaf_idx] * np.exp(-eta * (1 + sigma[height-1]) * loss[leaf_idx][height-1]))
    
        denominator = probs[node.leaf_ranges[0]:node.leaf_ranges[1]+1].sum()
        
        loss_i = np.log( nominator / denominator )**(-int(1/eta))

        loss[arm_idx][height] = sigma[height] * loss_i

    return loss

def update_arm_probability(loss, arm_idx, n_leaves, eta):
    
    nominator = probs[arm_idx] * np.exp(-eta*loss[arm_idx,:].sum())
    
    denominator = 0
    for leaf_idx in range(n_leaves):
        denominator += probs[leaf_idx] * np.exp(-eta*loss[leaf_idx,:].sum())

    probs[arm_idx] = nominator/denominator

    return probs

def remove_invalid_partitions(input_bounds, probs, h_ind, n_leaves, n_stages):
    prob_thres = 0.1/n_leaves
                                                        
    invalid_partitions = np.where(probs < prob_thres)[0]
    
    if invalid_partitions.shape[0] > 0:
        invalid_arm = invalid_partitions[0]
        for i in range(n_stages-1):
            for stage_idx in h_ind[i]:
                if invalid_arm[i] == 0:
                    input_bounds[stage_idx][0] = (input_bounds[stage_idx][0] + input_bounds[stage_idx][1]) / 2.0
                else:
                    input_bounds[stage_idx][1] = (input_bounds[stage_idx][0] + input_bounds[stage_idx][1]) / 2.0

    return input_bounds

def lambo_trial(trial_number, acqf, wandb, params=None):
    
    chosen_functions, h_ind, total_budget = params['obj_funcs'], params['h_ind'], params['total_budget']
    
    global_input_bounds = get_gen_bounds(h_ind, bound_list, funcs=chosen_functions)

    n_stages = len(h_ind)
    n_leaves = 2**(n_stages-1)

    probs = get_pdf(n_leaves)

    depths = [ 1 for i in range(n_stages - 1) ]

    H = sum(depths)
    h = H + 0
    
    arm_idx = random.randint(n_leaves)
    
    loss = np.zeros([n_leaves, H])
    
    for iter in iters:
    
        partitions, last_stage_partitions = build_partitions(global_input_bounds, h_ind, n_stages)
    
        mset, root = build_tree(partitions, depths, last_stage_partitions)
            
        leaves = mset.leaves

        input_bounds, arm_idx = select_arm(root, leaves, h, arm_idx, n_leaves)
        
        # Call bo_iteration & return (x, acq_value)

        sigma = np.array(random.choices([-1, 1], k=H))
        sigma[-1] = -1

        h = np.where(sigma == -1)[0][0]

        # Updating the loss estimators for the selected arm

        loss = update_loss_estimators(loss, root, arm_idx, sigma, eta, H, acq_value)

        # Updating the probability associated with the selected arm

        probs = update_arm_probability(loss, arm_idx, n_leaves, eta)

        global_input_bounds = remove_invalid_partitions(input_bounds, probs, h_ind, n_leaves, n_stages)
        
        












