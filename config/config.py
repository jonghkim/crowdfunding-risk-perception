import os

def get_config():
    params = {}

    ##### Data #####
    params['data_dir'] = 'data'

    params['training_data'] = 'training_data.csv'
    params['test_data'] = 'test_data.csv'
    
    ##### Preprocessing #####
    params['user_type'] = 'all' # experienced
    params['train_test_split_ratio'] = 0.7

    ##### Models #####
    # RandomForest with TF-IDF


    # RandomForest with Correlation Filtering


    # Two Topic Model with Correlation Filtering
    params['min_df'] = 0.01 # word appears at least # % of document
    params['max_features'] = 3000 # # of words to consider 

    return params