import matplotlib.pyplot as plt

from models.predictor.base_predictor import BasePredictor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from models.vectorizer.vectorizer_tfidf import VectorizerTfidf
from models.vectorizer.vectorizer_correlation_filtering import VectorizerCorrelationFiltering

from models.pipeline.label_generator.label_generator import LabelGenerator

import numpy as np
import pandas as pd

class RandomForest(BasePredictor):
    def __init__(self):
        self.vectorizer = None
        self.prediction_model = None

    def set_config(self, params):
        self.config = params

        # vectorizer
        self.vectorizer_type = params['vectorizer']['vectorizer_type']     
        self.min_df = params['vectorizer']['min_df']
        self.max_features = params['vectorizer']['max_features']

        if self.vectorizer_type == 'correlation_filtering':
            self.alpha = params['vectorizer']['alpha']     
            self.kappa = params['vectorizer']['kappa']     

        # predictor
        self.model_type = params['predictor']['model_type']
        self.label_type = params['predictor']['label_type']        
        self.param_grid = params['predictor']['param_grid']
        self.k_fold_cv = params['predictor']['k_fold_cv']

        self.plot_feature_importance = params['predictor']['plot_feature_importance']

        return self

    def fit_vectorizer(self, risk_desc_list, vectorizer_type, label=None):
        if vectorizer_type == 'tf_idf':
            tfidf_vectorizer = VectorizerTfidf()
            train_X, word_list, self.vectorizer = tfidf_vectorizer.fit_transform(risk_desc_list, min_df=self.min_df, max_features=self.max_features)
            
            print("   Size of Words",len(word_list))

        elif vectorizer_type == 'correlation_filtering':
            corr_filtering_vectorizer = VectorizerCorrelationFiltering()
            train_X, plus_word_list, minus_word_list, self.vectorizer, self.S_hat = corr_filtering_vectorizer.fit_transform(risk_desc_list, label, 
                                                                                    self.min_df, self.max_features, self.alpha, self.kappa)

            print("   Size of Positive Words: ", len(plus_word_list))
            print("   Size of Negative Words: ", len(minus_word_list))

            risk_words_df = pd.DataFrame()

            risk_words_df['high_risk_words'] = plus_word_list
            risk_words_df['low_risk_words'] = minus_word_list

            print(risk_words_df)

            word_list = plus_word_list + minus_word_list

        return train_X, word_list

    def transform_vectorizer(self, risk_desc_list, vectorizer_type):
        if vectorizer_type == 'tf_idf':
            test_X = self.vectorizer.transform(risk_desc_list)
            test_X = test_X.toarray()
            
        elif vectorizer_type == 'correlation_filtering':
            test_X = self.vectorizer.transform(risk_desc_list)
            test_X = test_X.toarray()         
            test_X = test_X[:, self.S_hat]             

        return test_X

    def get_label(self, df, label_type):
        label_generator = LabelGenerator()
        label = label_generator.get_label(df['perceived_risk'].tolist(), label_type)
        label = np.array(label)

        return label

    def fit_model(self, train_X, train_Y, param_grid, k_fold_cv):
        self.prediction_model = RandomForestClassifier() 

        grid_search = GridSearchCV(estimator = self.prediction_model, param_grid = param_grid, 
                                cv = k_fold_cv, n_jobs = -1, verbose = 2)
        grid_search.fit(train_X, train_Y)

        print("#### Best Grid Search Output ####")
        print(grid_search.best_params_)

        self.prediction_model = grid_search.best_estimator_

        return self

    def feature_importance_analysis(self, word_list, vectorizer_type):
        # Get numerical feature importances
        importances = list(self.prediction_model.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(word_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances 
        for pair in feature_importances[:20]:
            print('{}: {}'.format(pair[0], pair[1]))

        #### text - importance ####
        # Set the style
        plt.figure()

        plt.style.use('fivethirtyeight')
        # list of x locations for plotting
        x_values = list(range(len(importances)))
        # Make a bar chart
        plt.bar(x_values, importances, orientation = 'vertical')
        # Tick labels for x axis
        plt.xticks(x_values, word_list, rotation='vertical')
        # Axis labels and title
        plt.ylabel('Importance')
        plt.xlabel('Variable')
        plt.title('Variable Importances')
        plt.savefig('results/rf_{}.jpg'.format(vectorizer_type))

        #### text - cumulative importance ####
        plt.figure()        
        # List of features sorted from most to least important
        sorted_importances = [importance[1] for importance in feature_importances]
        sorted_features = [importance[0] for importance in feature_importances]
        # Cumulative importances
        cumulative_importances = np.cumsum(sorted_importances)
        # Make a line graph
        plt.plot(x_values, cumulative_importances, 'g-')
        # Draw line at 95% of importance retained
        plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
        # Format x ticks and labels
        plt.xticks(x_values, sorted_features, rotation = 'vertical')
        # Axis labels and title
        plt.xlabel('Variable')
        plt.ylabel('Cumulative Importance')
        plt.title('Cumulative Importances')
        plt.savefig('results/rf_cumulative_{}.jpg'.format(vectorizer_type))

        return self

    def predict_model(self, test_X):
        prediction = self.prediction_model.predict(test_X)
        
        return prediction

    def evaluation(self, prediction, test_Y):
        print("### Model - Random Forest ###")
        print("#### Setting: ", self.config)

        self.get_confusion_matrix(prediction, test_Y)
        self.get_classification_report(prediction, test_Y)
        self.get_accuracy_score(prediction, test_Y)
        
        return self

    def run(self, train_df, validation_df, test_df, params):
        
        # Set Config
        self.set_config(params)

        # Get Label
        train_Y = self.get_label(train_df, self.label_type)
        validation_Y = self.get_label(validation_df, self.label_type)
        
        # Get Features
        if self.vectorizer_type == 'tf_idf':
            train_X, word_list = self.fit_vectorizer(train_df['risk_desc'].tolist(), self.vectorizer_type)
            validation_X = self.transform_vectorizer(validation_df['risk_desc'].tolist(), self.vectorizer_type)
        elif self.vectorizer_type == 'correlation_filtering':
            train_X, word_list = self.fit_vectorizer(train_df['risk_desc'].tolist(), self.vectorizer_type, train_Y)
            validation_X = self.transform_vectorizer(validation_df['risk_desc'].tolist(), self.vectorizer_type)

        # Fit Model
        self.fit_model(train_X, train_Y, self.param_grid, self.k_fold_cv)

        if self.plot_feature_importance == True:
            self.feature_importance_analysis(word_list, self.vectorizer_type)

        # Prediction
        prediction = self.predict_model(validation_X)

        # Evaluation
        self.evaluation(prediction, validation_Y)

        return self
