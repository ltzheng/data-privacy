import pandas as pd
import numpy as np


def load_data(config):
    data_config = config['data']
    table = pd.read_csv(data_config['path'], header=None, skipinitialspace=True)

    # drop rows containing '?'
    print('row count before sanitizing:', table.shape[0])  # 32561
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
    tree = {k: hierarchy.loc[hierarchy['parent'] == k]['child'].tolist() for k in parents}
    # print('tree:', tree)
    height = get_tree_height(tree)
    inversed_tree = {}
    for index, row in hierarchy.iterrows():
        inversed_tree[row['child']] = row['parent']
    return inversed_tree, height


# builder for hierarchies with range generalization
def build_numerical_hierarchy(attribute_column, ranges=[5, 10, 20]):
    height = len(ranges) + 1  # including * generalization
    column = list(attribute_column)
    inversed_tree = {}

    for i, r in enumerate(ranges):
        if i == 0:
            pairs = []
            for num in column:
                pair = (r * int(num / r), r * (int(num / r) + 1))
                pairs.append(pair)
                inversed_tree[str(num)] = str(pair)
        else:
            new_pairs = []
            for pair in pairs:
                p = (r * int(pair[0] / r), r * (int(pair[0] / r) + 1))
                new_pairs.append(p)
                inversed_tree[str(pair)] = str(p)
            pairs = new_pairs
        for pair in pairs:
            inversed_tree[str(pair)] = '*'

    return inversed_tree, height


def get_tree_height(tree, root='*'):
    height = 0
    pointer = root
    while pointer in list(tree.keys()):
        height += 1
        pointer = tree[pointer][0]
    return height
