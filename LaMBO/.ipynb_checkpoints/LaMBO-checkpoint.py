class LaMBO:
    def __init__():
        return

    def build_partitions(self, input_bounds, h_ind, n_stages):
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
    
    def get_pdf(self, n_leaves):
        unif_prob = 1.0/n_leaves
        probs = np.array([unif_prob for i in range(n_leaves)])
        return probs
    
    def build_tree(self, partitions, depths, last_stage_partitions):
        root = Node(None, 0)
        mset = MSET(partitions, depths, last_stage_partition)
        
        left = mset.ConstructMSET(root, 0, 0, 1, [], [[], []])
        right = mset.ConstructMSET(root, 0, 1, 2, [], [[], []])
        root.add_child(left, right)
    
        mset.assign_leaf_ranges(root)
    
        return mset, root
    
    def get_subtree_arms(self, root, prev_arm_idx, prev_h):
        
        node = copy.deepcopy(root)
        curr_depth = 0
        
        while curr_depth < prev_h:
            if node.left.leaf_ranges[0] <= prev_arm_idx <= node.left.leaf_ranges[1]:
                node = node.left
            elif node.right.leaf_ranges[0] <= prev_arm_idx <= node.right.leaf_ranges[1]:
                node = node.right
                
            curr_depth += 1
    
        return node.leaf_ranges
    
    def select_arm(self, root, leaves, prev_h, prev_arm_idx, n_leaves):
        
    
        arm_choices = np.array([i for i in range(n_leaves)])
        valid_arm_idx = get_subtree_arms(root, prev_h, prev_arm_idx)
    
        valid_arm_choices = arm_choices[valid_arm_idx[0]:valid_arm_idx[1]+1]
        valid_probs = probs[valid_arm_idx[0]:valid_arm_idx[1]+1]
        
        arm_idx = random.choices(valid_arm_choices, weights=valid_probs)[0]
    
        return leaves[arm_idx], arm_idx
    
    def update_loss_estimators(self, loss, root, arm_idx, sigma, eta, H, acq_value):
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
    
    def update_arm_probability(self, loss, arm_idx, n_leaves, eta):
        
        nominator = probs[arm_idx] * np.exp(-eta*loss[arm_idx,:].sum())
        
        denominator = 0
        for leaf_idx in range(n_leaves):
            denominator += probs[leaf_idx] * np.exp(-eta*loss[leaf_idx,:].sum())
    
        probs[arm_idx] = nominator/denominator
    
        return probs
    
    def remove_invalid_partitions(self, input_bounds, probs, h_ind, n_leaves, n_stages, leaf_partitions):
        prob_thres = 0.1/n_leaves
                                                            
        invalid_partitions = np.where(probs < prob_thres)[0]
        
        if invalid_partitions.shape[0] > 0:
            first_invalid_idx = invalid_partitions[0]
            invalid_partition = leaf_partitions[first_invalid_idx]
            
            for i in range(n_stages-1):
                for stage_idx in h_ind[i]:
                    if invalid_partition[i] == 0:
                        input_bounds[0][stage_idx] = (input_bounds[0][stage_idx] + input_bounds[1][stage_idx]) / 2.0
                    else:
                        input_bounds[1][stage_idx] = (input_bounds[0][stage_idx] + input_bounds[1][stage_idx]) / 2.0
    
        return input_bounds
    
    def build_datasets(self, acqf, leaf_bounds, trial_number, n_leaves, params):
    
        X_tree, Y_tree, C_tree, C_inv_tree = [], [], []
        best_f = -1e9
        for leaf in range(n_leaves):
            x, y, c, c_inv = get_initial_data(
                params['lambo_init_data'], bounds=leaf_bounds[leaf], seed=trial_number*10000, acqf=acqf, params=params)
            
            X_tree.append(x)
            y_tree.append(y)
            c_tree.append(c)
            C_inv_tree.append(c_inv)
            best_f = max(best_f, y.max().item())
    
        return X_tree, Y_tree, C_tree, C_inv_tree, best_f
    
    def lambo_trial(self, trial_number, acqf, wandb, params=None):
        
        chosen_functions, h_ind, total_budget = params['obj_funcs'], params['h_ind'], params['total_budget']
        
        global_input_bounds = get_gen_bounds(h_ind, bound_list, funcs=chosen_functions)
    
        n_stages = len(h_ind)
        n_leaves = 2**(n_stages-1)
    
        partitions, last_stage_partitions = build_partitions(global_input_bounds, h_ind, n_stages)
        
        mset, root = build_tree(partitions, depths, last_stage_partitions)
    
        depths = [ 1 for i in range(n_stages - 1) ]
    
        X_tree, Y_tree, C_tree, C_inv_tree, best_f = build_datasets(acqf, mset.leaves, trial_number, n_leaves, params)
    
        probs = get_pdf(n_leaves)
    
        H = sum(depths)
        h = H + 0
        
        arm_idx = random.randint(n_leaves)
        
        loss = np.zeros([n_leaves, H])
        
        best_fs = -1e9
        for idx in arm_idx:
            best_fs = max(best_fs, Y[idx].max().item())
            
        total_budget = params['total_budget']
        cum_cost = 0
        iteration = 0
        
        while cum_cost < total_budget:
                
            leaf_bounds = mset.leaves
    
            input_bounds, arm_idx = select_arm(root, leaves, h, arm_idx, n_leaves)
    
            X, Y, C, C_inv = X_tree[arm_idx], Y_tree[arm_idx], C_tree[arm_idx], C_inv_tree[arm_idx]
    
            bounds = get_dataset_bounds(X, Y, C, C_inv, input_bounds)
    
            new_x, n_memoised, E_c, E_inv_c, y_pred, acq_value = lambo_iteration(X, Y, C, C_inv, bounds=bounds, acqf_str=acqf, decay=eta, iter=iteration, consumed_budget=cum_cost, params=params)
    
            sigma = np.array(random.choices([-1, 1], k=H))
            sigma[-1] = -1
    
            h = np.where(sigma == -1)[0][0]
    
            loss = update_loss_estimators(loss, root, arm_idx, sigma, eta, H, acq_value)
    
            probs = update_arm_probability(loss, arm_idx, n_leaves, eta)
    
            global_input_bounds = remove_invalid_partitions(input_bounds, probs, h_ind, n_leaves, n_stages, mset.leaf_partitions)
        
            partitions, last_stage_partitions = build_partitions(global_input_bounds, h_ind, n_stages)
        
            mset, root = build_tree(partitions, depths, last_stage_partitions)
        
            new_y = F(new_x, params).unsqueeze(-1)
            new_c = Cost_F(new_x, params)
            inv_cost = torch.tensor([1/new_c.sum()]).unsqueeze(-1)
            
            new_x, new_y, new_c, inv_cost = new_x.to(DEVICE), new_y.to(DEVICE), new_c.to(DEVICE), inv_cost.to(DEVICE)
            
            X[arm_idx] = torch.cat([X[arm_idx], new_x])
            Y[arm_idx] = torch.cat([Y[arm_idx], new_y])
            C[arm_idx] = torch.cat([C[arm_idx], new_c])
            C_inv[arm_idx] = torch.cat([C_inv[arm_idx], inv_cost])
            
            best_f = max(best_f, new_y.item())
            for stage in range(n_memoised):
                new_c[:,stage] = torch.tensor([params['epsilon']])
    
            sum_stages = new_c.sum().item()        
            cum_cost += sum_stages

            log = dict(
                acqf=acqf,
                trial=trial_number,
                iteration=iteration,
                best_f=best_f,
                sum_c_x=sum_stages,
                cum_costs=cum_cost,
            )
    
            iteration += 1
    
            dir_name = f"syn_logs_"
            csv_file_name = f"{dir_name}/{acqf}_trial_{trial_number}.csv"

            try:
                with open(csv_file_name, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    fieldnames = next(reader)
    
            except FileNotFoundError:
                fieldnames = ['acqf', 'trial', 'iteration', 'best_f', 'sum_c_x', 'cum_costs']
                with open(csv_file_name, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
    
            with open(csv_file_name, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(log)
            
            # wandb.log(log)
    
            best_fs.append(best_f)
        