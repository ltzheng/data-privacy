import argparse
from algorithms.samarati import samarati, Lattice
from algorithms.mondrian import mondrian
from utils.data_loader import load_data, build_categorical_hierarchy, build_range_hierarchy
from utils import display_table, default_data_config


def main(config):
    data = load_data(config=config)
    if config['samarati']:
        hierarchies, heights, leaves_num = {}, {}, {}
        for attr, path in config['data']['hierarchies'].items():
            if config['data']['samarati_generalization_type'][attr] == 'categorical':
                hierarchies[attr], heights[attr], leaves_num[attr] = build_categorical_hierarchy(path)
            else:  # range generalization
                hierarchies[attr], heights[attr], leaves_num[attr] = build_range_hierarchy(data['table'][attr])
            
        print('\nhierarchies:\n', hierarchies)
        print('\nhierarchy heights:\n', heights)
        # run samarati
        lattice = Lattice(hierarchies=hierarchies, quasi_id=data['quasi_id'], heights=heights)
        anonymized_table, vector, sup, loss_metric = samarati(table=data['table'], lattice=lattice, 
                                                    k=config['k'], maxsup=config['maxsup'], 
                                                    optimal=config['optimal_samarati'],
                                                    leaves_num=leaves_num,
                                                    sensitive=config['data']['sensitive'])

        # display
        print('generalization vector:', vector)
        print('max suppression:', sup)
        display_table(anonymized_table[data['quasi_id']], name='quasi identifiers in table')
        display_table(anonymized_table)
        # save to file      
        anonymized_table.to_csv('results/samarati.csv', header=None, index=None)

    elif config['mondrian']:
        table = data['table']

        # preprocessing categorical data
        encoders = {}
        from utils.data_loader import preprocess_categorical_column, recover_categorical_mondrian
        for attr in data['quasi_id']:
            if config['data']['mondrian_generalization_type'][attr] == 'categorical':
                table[attr], encoder = preprocess_categorical_column(table[attr].tolist())
                encoders[attr] = encoder
            
        # run mondrian
        anonymized_table, loss_metric = mondrian(table=table, quasi_id=data['quasi_id'], 
                                    k=config['k'], sensitive=config['data']['sensitive'])
        
        # recover from encoded data to categorical data
        for attr in data['quasi_id']:
            if config['data']['mondrian_generalization_type'][attr] == 'categorical':
                table[attr] = recover_categorical_mondrian(table[attr].tolist(), encoders[attr])

        # display
        display_table(anonymized_table[data['quasi_id']], name='quasi identifiers in table')
        display_table(anonymized_table)        
        # save to file      
        anonymized_table.to_csv('results/mondrian.csv', header=None, index=None)

    else:
        raise NotImplementedError('Algorithm not chosen. Please add argument --samarati or --mondrian.')

    return loss_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--maxsup", default=20, type=int)
    parser.add_argument("--samarati", action='store_true')
    parser.add_argument("--mondrian", action='store_true')
    parser.add_argument("--optimal-samarati", action='store_true')

    config = vars(parser.parse_args())
    config['data'] = default_data_config
    print('\nconfiguration:\n', config)

    
    main(config)
