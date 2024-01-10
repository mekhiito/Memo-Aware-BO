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

        loss = update_loss_estimators(loss, root, arm_idx, sigma, eta, H, acq_value)

        probs = update_arm_probability(loss, arm_idx, n_leaves, eta)

        global_input_bounds = remove_invalid_partitions(input_bounds, probs, h_ind, n_leaves, n_stages)
        