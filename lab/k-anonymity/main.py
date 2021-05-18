import argparse
from algorithms.samarati import samarati, Lattice
from utils.data_loader import load_data, build_categorical_hierarchy, build_numerical_hierarchy
from algorithms.mondrian import mondrian


def main(config):
    data = load_data(config=config)
    if config['samarati']:
        hierarchies = {}
        heights = {}
        for attr, path in config['data']['hierarchies'].items():
            if config['data']['generalization_type'][attr] == 'categorical':
                hierarchies[attr], heights[attr] = build_categorical_hierarchy(path)
            else:  # range generalization
                hierarchies[attr], heights[attr] = build_numerical_hierarchy(data['table'][attr])
            
        print('hierarchies:', hierarchies)
        print('heights:', heights)
        lattice = Lattice(hierarchies=hierarchies, quasi_id=data['quasi_id'], heights=heights)
        solution, vector, sup = samarati(table=data['table'], lattice=lattice, 
                                    k=config['k'], maxsup=config['maxsup'])

        print('\n====================')
        # drop the sensitive column
        # print('final table shape:', solution.shape)
        solution.drop(config['data']['sensitive'], axis=1, inplace=True)
        # print('final table shape:', solution.shape)
        print('anonymized table:\n', solution)
        print('====================')
        print('generalization vector:', vector)
        print('max suppression:', sup)
        print('====================')
        solution.to_csv('results/samarati.csv', header=None, index=None)

    elif config['mondrian']:
        mondrian(table=data['table'], quasi_id=data['quasi_id'], k=config['k'])

    else:
        raise NotImplementedError('Algorithm not chosen. Please add argument --samarati or --mondrian.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--maxsup", default=20, type=int)
    parser.add_argument("--samarati", action='store_true')
    parser.add_argument("--mondrian", action='store_true')

    config = vars(parser.parse_args())
    config['data'] = {
        'path': 'data/adult.data', 
        # QI for samarati
        'categorical_quasi_id': ['age', 'gender', 'race', 'marital_status'],
        # QI for mondrian
        'numerical_quasi_id': ['age', 'education_num'],
        'sensitive': 'occupation',
        'hierarchies': {
            'age': '',
            'gender': 'data/adult_gender.txt',
            'race': 'data/adult_race.txt',
            'marital_status': 'data/adult_marital_status.txt',
        },
        'columns': ['age', 'work_class', 'final_weight', 'education', 'education_num', 
                    'marital_status', 'occupation', 'relationship', 'race', 'gender', 
                    'capital_gain', 'capital_loss', 'hours_per_week', 
                    'native_country', 'class'],
        'generalization_type': {
            'age': 'numerical',
            'gender': 'categorical',
            'race': 'categorical',
            'marital_status': 'categorical',
        },
    }
    print('config:', config)

    
    main(config)
