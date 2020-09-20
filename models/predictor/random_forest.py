import matplotlib.pyplot as plt

from models.predictor.base_predictor import BasePredictor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

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

        if self.vectorizer_type == 'corr_filter':
            self.alpha = params['vectorizer']['alpha']     
            self.kappa = params['vectorizer']['kappa']     

        # predictor
        self.model_type = params['predictor']['model_type']
        self.prediction_label = params['predictor']['prediction_label']
        self.user_type = params['predictor']['user_type']
        self.label_type = params['predictor']['label_type']        
        self.hyperparams = params['predictor']['hyperparams']
        self.k_fold_cv = params['predictor']['k_fold_cv']

        self.plot_feature_importance = params['predictor']['plot_feature_importance']

        return self

    def fit_vectorizer(self, risk_desc_list, vectorizer_type, label=None):
        if vectorizer_type == 'tf_idf':
            tfidf_vectorizer = VectorizerTfidf()
            train_X, word_list, self.vectorizer = tfidf_vectorizer.fit_transform(risk_desc_list, min_df=self.min_df, max_features=self.max_features)
            
            print("   Size of Words",len(word_list))

        elif vectorizer_type == 'corr_filter':
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
            
        elif vectorizer_type == 'corr_filter':
            test_X = self.vectorizer.transform(risk_desc_list)
            test_X = test_X.toarray()         
            test_X = test_X[:, self.S_hat]             

        return test_X

    def get_label(self, df, label_type):
        label_generator = LabelGenerator()
        label = label_generator.get_label(df['perceived_risk'].tolist(), label_type)
        label = np.array(label)

        return label

    def evaluate_model(self, X, Y, hyperparams, k_fold_cv):
        evaluation_model = RandomForestClassifier(**hyperparams) 

        scoring = {'acc': 'accuracy',
                   'prec': 'precision_macro',
                   'rec': 'recall_macro'}
                
        scores = cross_validate(evaluation_model, X, Y, scoring=scoring)

        # Test Set Score
        print("Train Set - Mean Acc: ", np.mean(scores['train_acc']))
        print("Train Set - Mean Precision: ", np.mean(scores['train_prec']))
        print("Train Set - Mean Recall: ", np.mean(scores['train_rec']))
        
        print("Test Set - Mean Acc: ", np.mean(scores['test_acc']))
        print("Test Set - Mean Precision: ", np.mean(scores['test_prec']))
        print("Test Set - Mean Recall: ", np.mean(scores['test_rec']))

        return np.mean(scores['test_acc'])

    def fit_model(self, X, Y, hyperparams):
        self.prediction_model = RandomForestClassifier(**hyperparams)
        self.prediction_model.fit(X, Y)

        return self

    def feature_importance_analysis(self, word_list, vectorizer_type, acc):
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
        plt.savefig('results/rf_{}_{}_usr_type_{}_acc_{:.2f}.jpg'.format(vectorizer_type, self.label_type, self.user_type, acc))

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
        plt.savefig('results/rf_cumulative_{}_{}_usr_type_{}_acc_{:.2f}.jpg'.format(self.vectorizer_type, self.label_type, self.user_type, acc))

        return self

    def predict_model(self, test_X):
        prediction = self.prediction_model.predict(test_X)
        
        return prediction

    def run(self, perceived_risk_df, prediction_df, params):
        
        # Set Config
        self.set_config(params)

        # Get Label
        Y = self.get_label(perceived_risk_df, self.label_type)
        
        # Get Features
        if self.vectorizer_type == 'tf_idf':
            X, word_list = self.fit_vectorizer(perceived_risk_df['risk_desc'].tolist(), self.vectorizer_type)
            
            if self.prediction_label == 'desc_combined':
                prediction_df['desc_combined'] =  prediction_df["desc_total"] + " " + prediction_df["risk_desc"]

            prediction_X = self.transform_vectorizer(prediction_df[self.prediction_label].tolist(), self.vectorizer_type)

        elif self.vectorizer_type == 'corr_filter':
            X, word_list = self.fit_vectorizer(perceived_risk_df['risk_desc'].tolist(), self.vectorizer_type, Y)

            if self.prediction_label == 'desc_combined':
                prediction_df['desc_combined'] =  prediction_df["desc_total"] + " " + prediction_df["risk_desc"]

            prediction_X = self.transform_vectorizer(prediction_df[self.prediction_label].tolist(), self.vectorizer_type)
        
        # Evaluation Model
        acc = self.evaluate_model(X, Y, self.hyperparams, self.k_fold_cv)

        # Train Final Model
        self.fit_model(X, Y, self.hyperparams)

        if self.plot_feature_importance == True:
            self.feature_importance_analysis(word_list, self.vectorizer_type, acc)
        
        # Prediction
        prediction_Y_hat = self.predict_model(prediction_X)
        prediction_df['prediction'] = prediction_Y_hat

        if self.vectorizer_type == 'tf_idf':
            prediction_df.to_csv('results/{}_wv_size_{}_{}_usr_type_{}_acc_{:.2f}.csv'.format('rf_{}'.format(self.vectorizer_type), len(word_list), self.label_type, self.user_type, acc))
        elif self.vectorizer_type == 'corr_filter':
            prediction_df.to_csv('results/{}_wv_size_{}_alpha_{}_kappa_{}_{}_usr_type_{}_acc_{:.2f}.csv'.format('rf_{}'.format(self.vectorizer_type), len(word_list), self.alpha, self.kappa, self.label_type, self.user_type, acc))
        
        return self
