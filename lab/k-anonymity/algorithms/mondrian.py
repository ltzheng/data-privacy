from statistics import median
import random
import pandas as pd
from utils.loss_metrics import numerical_loss_metric


class Partition:
    def __init__(self, table, quasi_id, k):
        self.table = table
        self.quasi_id = quasi_id
        self.k = k
        self.allow_split = self.set_allow_split()
        self.lhs = None
        self.rhs = None
        self.summary = pd.DataFrame()

    # map(key: a dimension -> value: whether it can be split)
    def set_allow_split(self):
        allow_split = {}
        for qi in self.quasi_id:
            if not self.table[qi].tolist():  # column is empty
                allow_split[qi] = False
                break
            split_val = median(self.table[qi].tolist())  # get median value
            # check k-anonymity
            lhs_tb = self.table[self.table[qi] <= split_val]
            rhs_tb = self.table[self.table[qi] > split_val]
            allow_split[qi] = self.check(lhs_tb, rhs_tb)
        return allow_split

    # random selection from all allow-split dimensions
    def choose_dimension(self):
        candidates = [qi for qi in self.quasi_id if self.allow_split[qi]]
        return random.choice(candidates)

    # split table with given dimension and median value
    def split(self, dim, split_val):
        lhs_tb = self.table[self.table[dim] <= split_val]
        rhs_tb = self.table[self.table[dim] > split_val]
        lhs = Partition(table=lhs_tb,
                        quasi_id=self.quasi_id,
                        k=self.k)
        rhs = Partition(table=rhs_tb,
                        quasi_id=self.quasi_id,
                        k=self.k)
        return lhs, rhs

    # check k-anonymity constraint
    def check(self, table1, table2):
        return table1.shape[0] >= self.k and table2.shape[0] >= self.k

    # recursively partition table
    def strict_anonymize(self):
        if not any(self.allow_split.values()):  # until no possible split
            for dim in self.quasi_id:  # replace original values with a new range (string format)
                max_val = max(self.table[dim].tolist())
                min_val = min(self.table[dim].tolist())
                if min_val != max_val:  # replace the column of selected dimension
                    self.table[dim] = [str(min_val) + '-' + str(max_val)] * self.table.shape[0]
            self.summary = self.table

        else:
            # choose a dimension from the attribute domain
            dim = self.choose_dimension()
            # get the median value
            split_val = median(self.table[dim].tolist())
            # split table and check k-anonymity
            lhs, rhs = self.split(dim, split_val)
            if not self.check(lhs.table, rhs.table):
                self.allow_split[dim] = False
            
            # combine lhs table and rhs table
            lhs_summary = lhs.strict_anonymize()
            rhs_summary = rhs.strict_anonymize()
            self.summary = lhs_summary.append(rhs_summary)

        return self.summary


def mondrian(table, quasi_id, k, sensitive):
    partition = Partition(table=table, quasi_id=quasi_id, k=k)
    partition.strict_anonymize()
    anonymized_table = partition.summary
    # drop the sensitive column
    anonymized_table.drop(sensitive, axis=1, inplace=True)
    loss_metric = numerical_loss_metric(anonymized_table.loc[:, quasi_id])
    print('\n====================')
    print('\nloss_metric:', loss_metric)

    return anonymized_table