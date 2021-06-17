def generate_categorical_loss_metric_map(leaves_num, hierarchies):
    loss_metric_map = {attr: {} for attr in hierarchies.keys()}
    print('\nleaves_num:\n', leaves_num)
    for attr, vals in hierarchies.items():
        loss_metric_map[attr]['*'] = 1
        for v in vals:
            if v in leaves_num[attr].keys():
                loss_metric_map[attr][v] = (leaves_num[attr][v] - 1) / (leaves_num[attr]['*'] - 1)
            else: 
                loss_metric_map[attr][v] = 0
    return loss_metric_map


def categorical_loss_metric(qi_columns, leaves_num, hierarchies, sup):
    loss_metric_map = generate_categorical_loss_metric_map(leaves_num, hierarchies)
    print('\nloss_metric_map:\n', loss_metric_map)
    loss_metric = 0

    for attr in qi_columns:
        col = qi_columns[attr].tolist()
        # the loss for an attribute is the AVERAGE of the loss for all tuples
        # the loss for the entire data set is the SUM of the losses for each attribute
        sum_attr_lm = sum([loss_metric_map[attr][str(v)] for v in col])
        loss_metric += (sum_attr_lm + sup) / (len(col) + sup)
    return loss_metric


def compute_numerical_loss_metric(column):
    loss = 0
    # initialize lowest and highest values
    if not isinstance(column[0], int):  # string value, e.g., '35-40'
        current_range = [int(i) for i in list(column[0].replace(' ', '').split('-'))]
        lowest, highest = current_range[0], current_range[1]
    else:  # integer value, e.g., 37
        lowest, highest = column[0], column[0]

    # iterate through column
    for v in column:
        if not isinstance(v, int):  # extract range from table content (string, e.g., '35-40')
            current_range = [int(i) for i in list(v.replace(' ', '').split('-'))]
            loss += current_range[1] - current_range[0]
            # update lowest & highest
            lowest = min(lowest, current_range[0])
            highest = max(highest, current_range[1])
        else:  # integer value, loss is 0 here
            lowest = min(lowest, v)
            highest = max(highest, v)
            
    max_range = highest - lowest
    return loss / (max_range * len(column))  # average


def numerical_loss_metric(qi_columns):
    loss_metric = 0
    for attr in qi_columns:
        col = qi_columns[attr].tolist()
        # the loss for the entire data set is the SUM of the losses for each attribute
        loss_metric += compute_numerical_loss_metric(col)
    return loss_metric