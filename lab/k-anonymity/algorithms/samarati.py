from itertools import product
import pandas as pd


class Lattice():
    def __init__(self, hierarchies, quasi_id, heights):
        self.hierarchies = hierarchies
        self.heights = heights
        self.total_height = sum(heights.values())
        self.attr_num = len(self.heights)
        self.quasi_id = quasi_id
        self.height_array = [list(range(h + 1)) for h in self.heights.values()]
        self.lattice_map = self.build_map()

    def build_map(self):
        lattice_map = {h: [] for h in range(self.total_height + 1)}
        all_combinations = [x for x in product(*self.height_array)]
        for dist in all_combinations:
            temp = sum(dist)
            if temp <= self.total_height:
                lattice_map[temp].append(dist)
        print('lattice_map:', lattice_map)
        return lattice_map
    
    def get_vectors(self, height):
        return self.lattice_map[height]

    def satisfies(self, vector, k, table, maxsup):
        # generalization
        anonymized_table = self.generalization(table, vector)        
        # suppression & validation
        valid, anonymized_table, sup = self.validation(anonymized_table, k, maxsup)
        return valid, sup, anonymized_table

    def generalization(self, table, vector):
        table = table.copy()
        for attribute, gen_level in zip(self.quasi_id, vector):
            col = [str(elem) for elem in list(table[attribute])]
            # find the ancestors for generalization
            ancestors = {k: k for k in [str(elem) for elem in list(set(col))]}
            for k in ancestors.keys():
                for _ in range(gen_level):
                    ancestors[k] = self.hierarchies[attribute][ancestors[k]]
            # replace old values
            col = [ancestors[elem] for elem in col]
            table[attribute] = col

        return table

    def validation(self, table, k, maxsup):
        sup = 0
        table = table.copy()
        anonymized_table = pd.DataFrame(columns=table.columns)
        while sup <= maxsup and not table.empty:
            first_row = table.loc[table.index[0], self.quasi_id]
            row_counts = table.shape[0]
            # Note: hard-coded quasi-identifier conditions here, need to be changed for other datasets
            conditions = (table['age'] != first_row['age']) | (table['gender'] != first_row['gender']) | \
                (table['race'] != first_row['race']) | (table['marital_status'] != first_row['marital_status'])
            # tuples same as 1st row are deleted
            residual_table = table[~conditions]
            table = table[conditions]
            new_row_counts = table.shape[0]
            delta = row_counts - new_row_counts
            if (delta < k):
                sup += delta
            else:
                anonymized_table = anonymized_table.append(residual_table)

        return sup <= maxsup, anonymized_table, sup


def samarati(table, lattice, k=10, maxsup=20):
    low = 0
    high = lattice.total_height
    satisfied_vector = lattice.get_vectors(lattice.total_height)[0]
    solution = lattice.generalization(table=table, vector=satisfied_vector)
    final_sup = None
    # print('initial solution:\n', solution)
    while low < high:
        mid = int((low + high) / 2)
        vectors = lattice.get_vectors(mid)
        reach_k = False
        for v in vectors:
            valid, sup, anonymized_table = lattice.satisfies(vector=v, k=k, table=table, maxsup=maxsup)
            if valid:
                print('====================')
                print('satisfied vector:', v)
                print('maxsup:', sup)
                satisfied_vector = v
                final_sup = sup
                solution = anonymized_table
                reach_k = True
                break
        if reach_k:
            high = mid
        else:
            low = mid + 1

    return solution, satisfied_vector, final_sup