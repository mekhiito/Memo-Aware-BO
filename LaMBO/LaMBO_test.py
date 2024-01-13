import numpy as np
import random
import torch
from MSET import MSET, Node
        
def Loss_Test(root):
    node = copy.deepcopy(root)
    probs = np.array([1.0/8 for i in range(8)])
    loss = np.zeros([8, 4])
    sigma = random.choices([-1, 1], k=4)
    sigma[-1] = -1
    arm_idx = 3
    eta = 0.9
    for h in range(1, 3):
        
        if node.left.leaf_ranges[0] <= arm_idx <= node.left.leaf_ranges[1]:
            node = node.left
        elif node.right.leaf_ranges[0] <= arm_idx <= node.right.leaf_ranges[1]:
            node = node.right
        
        nominator = 0
        for leaf_idx in range(node.leaf_ranges[0], node.leaf_ranges[1]):
            nominator += (probs[leaf_idx] * np.exp(-eta * (1 + sigma[h-1]) * loss[leaf_idx][h-1]))
    
        denominator = probs[node.leaf_ranges[0]:node.leaf_ranges[1]+1].sum()
        
        loss_i = np.log( nominator / denominator )**(-int(1/eta))

        loss[arm_idx][h] = sigma[h] * loss_i

    nominator = probs[arm_idx] * np.exp(-eta*loss[arm_idx,:].sum())
    denominator = 0
    for leaf_idx in range(8):
        denominator += probs[leaf_idx] * np.exp(-eta*loss[leaf_idx,:].sum())
    
    arm_choices = [i for i in range(8)]
    print('Loss Test: ', random.choices(arm_choices, probs)[0])

def MSET_Test():
    n_stages = 3

    h_ind = [ [0,1,2], [3,4,5], [6,7,8], [9, 10]]
    input_bounds = [[0,0,0,0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1,1,1,1]]
    

    for dd in range(2):
        print('Global Bounds this iteration are: ')
        print(input_bounds)
        partitions = []
        n_stages = len(h_ind)
        n_leaves = 2**(n_stages-1)
        
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
    
        depths = [ 1 for i in range(n_stages - 1) ]
        
        root = Node(None, 0, 0)
        tree_const = MSET(partitions, depths, last_stage_partition)
    
        left = tree_const.ConstructMSET(root, 0, 0, 1, [], [[], []])
        right = tree_const.ConstructMSET(root, 0, 1, 2, [], [[], []])
        root.add_child(left, right)

        print('Printing Tree Now')
        tree_const.assign_leaf_ranges(root)
        tree_const.print_MSET(root)

        probs = np.array([0.000000001, 0.5, 0.4, 0.01, 0.1, 0.2, 0.15, 0.44])

        prob_thres = 0.1/n_leaves
                                                        
        invalid_partitions = np.where(probs < prob_thres)[0]
        
        print('Invalid Partitions: ', invalid_partitions)
        if invalid_partitions.shape[0] > 0:
            
            invalid_partition = tree_const.leaf_partitions[0]
            print('Invalid Partition: ', invalid_partition)
            
            for i in range(n_stages-1):
                for stage_idx in h_ind[i]:
                    if invalid_partition[i] == 0:
                        input_bounds[0][stage_idx] = (input_bounds[0][stage_idx] + input_bounds[1][stage_idx]) / 2.0
                    else:
                        input_bounds[1][stage_idx] = (input_bounds[0][stage_idx] + input_bounds[1][stage_idx]) / 2.0

    Loss_Test(root)

MSET_Test()









