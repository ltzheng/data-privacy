import pandas as pd
import numpy as np


def load_data(config):
    data_config = config['data']
    table = pd.read_csv(data_config['path'], header=None, skipinitialspace=True)

    # drop rows containing '?'
    print('\nrow count before sanitizing:', table.shape[0])  # 32561
    table = table[~table.isin(['?']).any(axis=1)]
    print('row count sanitized:', table.shape[0])  # 30162

    # rename column names
    table.columns = data_config['columns']

    if config['samarati']:
        quasi_id = data_config['categorical_quasi_id']
    else:
        quasi_id = data_config['numerical_quasi_id']

    data = {'table': table, 'quasi_id': quasi_id, 'sensitive': data_config['sensitive']}
    # print('data:', data)
    return data


def build_categorical_hierarchy(path):
    hierarchy = pd.read_csv(path, header=None)
    hierarchy.columns = ['child', 'parent']
    children = hierarchy['child'].tolist()
    parents = hierarchy['parent'].unique().tolist()
    # build tree (map parent to children)
    tree = {k: hierarchy.loc[hierarchy['parent'] == k]['child'].tolist() for k in parents}
    # count number of leaves for all subtree (for loss metric)
    leaves_num = {k: subtree_leaves(tree, k) for k in parents}
    # get height of tree (for lattice)
    height = get_tree_height(tree)
    # build inversed tree (map children to parent)
    inversed_tree = {}
    for index, row in hierarchy.iterrows():
        inversed_tree[row['child']] = row['parent']
    return inversed_tree, height, leaves_num


# builder for hierarchies with range generalization
def build_range_hierarchy(attribute_column, ranges=[5, 10, 20]):
    height = len(ranges) + 1  # including * generalization
    column = list(attribute_column)
    inversed_tree = {}
    leaves_num = {}
    visited = []

    for i, r in enumerate(ranges):
        if i == 0:  # generalize original values
            pairs = []
            for num in column:
                if num not in visited:
                    # left closed, right open
                    pair = (r * int(num / r), r * (int(num / r) + 1))
                    pairs.append(pair)
                    inversed_tree[str(num)] = str(pair)

                    visited.append(num)
                    if str(pair) not in leaves_num.keys():
                        leaves_num[str(pair)] = 1
                    else:
                        leaves_num[str(pair)] += 1
        else:  # generalize range values
            new_pairs = []
            for pair in pairs:
                if str(pair) not in visited:
                    p = (r * int(pair[0] / r), r * (int(pair[0] / r) + 1))
                    new_pairs.append(p)
                    inversed_tree[str(pair)] = str(p)
                    
                    visited.append(str(pair))
                    if str(p) not in leaves_num.keys():
                        leaves_num[str(p)] = leaves_num[str(pair)]
                    else:
                        leaves_num[str(p)] += leaves_num[str(pair)]
            pairs = new_pairs

    for pair in pairs:  # root generalization
        if str(pair) not in visited:
            inversed_tree[str(pair)] = '*'

            visited.append(str(pair))
            if '*' not in leaves_num.keys():
                leaves_num['*'] = leaves_num[str(pair)]
            else:
                leaves_num['*'] += leaves_num[str(pair)]

    return inversed_tree, height, leaves_num


def get_tree_height(tree, root='*'):
    height = 0
    pointer = root
    while pointer in list(tree.keys()):
        height += 1
        pointer = tree[pointer][0]
    return height


# get the number of leaves for the subtree with given root
def subtree_leaves(tree, root='*'):
    if root not in tree.keys():
        return 1
    children = tree[root]
    leaves_num = sum([subtree_leaves(tree, r) for r in children])
    return leaves_num