import os

def get_config():
    params = {}

    ##### Data #####
    params['data_dir'] = 'data'

    params['training_data'] = 'training_data.csv'
    params['test_data'] = 'test_data.csv'
    
    ##### Preprocessing #####
    params['user_type'] = 'all' # all, experienced
    params['label_type'] = 'categorical_type1' # numerical, categorical_type1: [1,3]/(3,5], categorical_type2: [1,3)/[3,5]
    params['train_test_split_ratio'] = 0.7

    ##### Models #####
    ## Prediction Models for Categorical Label
    # Model1. RandomForest with TF-IDF
    params['model1_params'] = {'vectorizer': {'vectorizer_type': 'tf_idf', 'min_df': 10, 'max_features': 3000}, 
                               'predictor': {'model_type': 'random_forest',
                                             'cross_validation_grid_search': True,
                                             'param_grid': {
                                                            'bootstrap': [True],
                                                            'max_depth': [10, 50, None],
                                                            'max_features': ['log2', 'sqrt'],
                                                            'min_samples_leaf': [3, 5],
                                                            'min_samples_split': [5, 10],
                                                            'n_estimators': [100]
                                                           },
                                            'plot_feature_importance':False}
                              }

    # Model2. RandomForest with Correlation Filtering
    params['model2_params'] = {'vectorizer': {'vectorizer_type': 'correlation_filtering', 'min_df': 10, 'max_features': 3000,
                                             'alpha':0.1, # alpha: correlation filter    
                                             'kappa':0.01}, # kappa: words frequency filter
                               'predictor': {'model_type': 'random_forest',
                                             'cross_validation_grid_search': True,
                                             'param_grid': {
                                                            'bootstrap': [True],
                                                            'max_depth': [10, 50, None],
                                                            'max_features': ['log2', 'sqrt'],
                                                            'min_samples_leaf': [3, 5],
                                                            'min_samples_split': [5, 10],
                                                            'n_estimators': [100]
                                                           },
                                             'plot_feature_importance':False}
                              }

    # Modle3. Two Topic Model with Correlation Filtering
    params['model3_params'] = {'vectorizer': {'vectorizer_type': 'correlation_filtering', 'min_df': 10, 'max_features': 3000,
                                             'alpha':0.1, # alpha: correlation filter    
                                             'kappa':0.01}, # kappa: words frequency filter
                               'predictor': {'model_type': 'two_topic_model', 
                                             'lamb':5}} #lamb: beta priori penalty

    ## Prediction Models for Numerical Label
    # Model4. ElasticNet with TF-IDF
    params['model4_params'] = {}

    # Model5. ElasticNet with Correlation Filtering
    params['model5_params'] = {}

    ##### Results #####
    params['result_dir'] = 'results'

    return params