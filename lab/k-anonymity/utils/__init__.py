def display_table(table, name='anonymized table'):
    print('\n====================')
    print(name, '\n', table)
    print('====================\n')


default_data_config = {
    'path': 'data/adult.data', 
    # QI for samarati
    'samarati_quasi_id': ['age', 'gender', 'race', 'marital_status'],
    # QI for mondrian
    'mondrian_quasi_id': ['age', 'education_num'],
    # 'mondrian_quasi_id': ['age', 'gender', 'education_num'],
    'sensitive': 'occupation',
    'columns': ['age', 'work_class', 'final_weight', 'education', 'education_num', 
                'marital_status', 'occupation', 'relationship', 'race', 'gender', 
                'capital_gain', 'capital_loss', 'hours_per_week', 
                'native_country', 'class'],
    'samarati_generalization_type': {
        'age': 'range',
        'gender': 'categorical',
        'race': 'categorical',
        'marital_status': 'categorical',
    },
    'hierarchies': {
        'age': None,  # range type generalization
        'gender': 'data/adult_gender.txt',
        'race': 'data/adult_race.txt',
        'marital_status': 'data/adult_marital_status.txt',
    },
    'mondrian_generalization_type': {
        'age': 'numerical',
        # 'gender': 'categorical',
        'education_num': 'numerical',
    },
}