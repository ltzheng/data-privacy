# K-Anonymity

**郑龙韬 PB18061352**

## Problem Statement

Given a table to be generalized, every QI-cluster should contains k or more tuples after the k-anonymization.

### Dataset

- [adult.data](https://archive.ics.uci.edu/ml/datasets/adult) 
  
  Attributes include 'age', 'work_class', 'final_weight', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', and 'class'.
- adult_*.txt: there are 2 strings in each row, separated by a comma. The left one represents the children node, and the right one represents the parent node.

## Algorithms

### Samarati (categorical)

In [samarati.py](algorithms/samarati.py), we build a class `Lattice`. The algorithm is implementd in method `samarati()`.

### Mondrian (numerical)

## User Guide

To run samarati with k = 10 and maxsup = 20:
```bash
python main.py --samarati --k 10 --maxsup 20
```

To run mondrian with k = 10:
```bash
python main.py --mondrian --k 10
```

The data path, hierarchy paths and other configurations can be easily editied in the dictionary `config` in [main.py](main.py).

Code structure:
- algorithms
  - [samarati.py](algorithms/mondrian.py)
  - [mondrian.py](algorithms/mondrian.py)
- data: dataset and generalization hierarchies
- results: where anonymized tables are saved
- utils
  - data loader
  - loss metrics
  - display table
- main.py

See more argument setting description:
```bash
python main.py --help
```

## Results & Analysis

different k

run time

loss metric

## Discussion & Conclusion


## Bonus Part

### Best result for Samarati


### Mondrian for categorical attribute

Calculate the frequency for each categorical attribute. Consider the categorical attributes as a distribution, then select the median based on it.