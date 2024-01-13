import copy
import numpy as np
import random

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Node:
    def __init__(self, parent, depth, idx):
        self.parent = parent
        self.depth = depth
        self.idx = idx
        self.left = None
        self.right = None
        self.value = None
        self.leaf_partitions = None
        self.leaf_ranges = None

    def add_parent(self, parent):
        self.parent = parent

    def add_child(self, left, right):
        self.left = left
        self.right = right

    def add_value(self, value):
        self.value = value

    def add_leaf_partition(self, leaf_partitions):
        self.leaf_partitions = leaf_partitions

    def get_depth(self):
        return self.depth

    def get_idx(self):
        return self.idx

class MSET:
    def __init__(self, partitions, depths, last_stage_bounds):
        self.partitions = partitions
        self.depths = depths
        self.stage = 0
        self.last_stage_bounds = last_stage_bounds
        self.leaves = []
        self.leaf_partitions = []

    def update_bounds(self, curr_stage_partitions, p_bounds, p_idx):
        bounds = copy.deepcopy(p_bounds)
        curr_partition = []

        for hp_partitions in curr_stage_partitions:
            curr_partition.append(hp_partitions[p_idx])

        for ps in curr_partition:
            bounds[0].append(ps[0])
            bounds[1].append(ps[1])

        return bounds

    def create_leaf(self, node, leaf_partition, bounds):
        for ps in self.last_stage_bounds:
            bounds[0].append(ps[0])
            bounds[1].append(ps[1])
        # bounds = torch.tensor(bounds, device=DEVICE, dtype=torch.double)
        node.add_value(bounds)
        node.add_leaf_partition(leaf_partition)
        
        self.leaves.append(bounds)
        self.leaf_partitions.append(leaf_partition)
        
        return node

    def build_children(self, node, stage_idx, node_idx, leaf_partition, bounds):
        
        left_range, right_range = 1e9, -1e9
        
        left = self.ConstructMSET(node, stage_idx + 1, 0, 2*node_idx + 1, leaf_partition, bounds)
        right = self.ConstructMSET(node, stage_idx + 1, 1, 2*node_idx + 2, leaf_partition, bounds)
        node.add_child(left, right)

        return node

    def ConstructMSET(self, parent, stage_idx, p_idx, node_idx, leaf_partitions=None, p_bounds=None):

        leaf_partition = copy.deepcopy(leaf_partitions)

        curr_stage_partitions = self.partitions[stage_idx]

        leaf_partition.append(p_idx)
        
        bounds = self.update_bounds(curr_stage_partitions, p_bounds, p_idx)
        
        curr_depth = parent.get_depth() + self.depths[stage_idx]

        node = Node(parent, curr_depth, node_idx)

        if stage_idx >= len(self.depths) - 1:
            node = self.create_leaf(node, leaf_partition, bounds)
            return node

        node = self.build_children(node, stage_idx, node_idx, leaf_partition, bounds)

        return node

    def assign_leaf_ranges(self, node):
        # Base case: If the node is a leaf, its range is just its index
        if not node.left and not node.right:
            first_idx = 2**node.depth - 1
            node.leaf_ranges = (node.idx - first_idx, node.idx - first_idx)
            return node.leaf_ranges
    
        # Recursive case: Compute ranges for left and right subtrees
        left_range = self.assign_leaf_ranges(node.left) if node.left else None
        right_range = self.assign_leaf_ranges(node.right) if node.right else None
    
        # Combine the ranges from left and right children
        start = left_range[0] if left_range else right_range[0]
        end = right_range[1] if right_range else left_range[1]
        node.leaf_ranges = (start, end)
    
        return node.leaf_ranges

    def get_leaves(self):
        return self.leaves

    def get_leaf_partitions(self):
        return self.leaf_partitions

    def print_MSET(self, node):
        
        if node.left is not None:
            # print(node.idx, ' ', node.leaf_ranges)
            self.print_MSET(node.left)
            self.print_MSET(node.right)
        else:
            print(node.idx)
            print(node.value)
            print(node.leaf_partitions)
            print(node.leaf_ranges)
            print('\n\n')
        
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









