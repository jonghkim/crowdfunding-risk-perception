from models.predictor.base_predictor import BasePredictor
from sklearn.ensemble import RandomForestClassifier
from models.vectorizer.vectorizer_tfidf import VectorizerTfidf
from models.vectorizer.vectorizer_correlation_filtering import VectorizerCorrelationFiltering

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
        self.n_estimators = params['predictor']['n_estimators']

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

        return train_X

    def transform_vectorizer(self, risk_desc_list, vectorizer_type):
        if vectorizer_type == 'tf_idf':
            test_X = self.vectorizer.transform(risk_desc_list)
            test_X = test_X.toarray()
            
        elif vectorizer_type == 'correlation_filtering':
            test_X = self.vectorizer.transform(risk_desc_list)
            test_X = test_X.toarray()         
            test_X = test_X[:, self.S_hat]             

        return test_X

    def get_label(self, df):
        label = np.array(df['label'].tolist())

        return label

    def fit_model(self, train_X, train_Y, n_estimators):
        self.prediction_model = RandomForestClassifier(n_estimators=n_estimators)  
        self.prediction_model.fit(train_X, train_Y)        

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

    def run(self, train_df, test_df, params):
        
        # Set Config
        self.set_config(params)

        # Get Label
        train_Y = self.get_label(train_df)
        test_Y = self.get_label(test_df)
        
        # Get Features
        if self.vectorizer_type == 'tf_idf':
            train_X = self.fit_vectorizer(train_df['risk_desc'].tolist(), self.vectorizer_type)
            test_X = self.transform_vectorizer(test_df['risk_desc'].tolist(), self.vectorizer_type)
        elif self.vectorizer_type == 'correlation_filtering':
            train_X = self.fit_vectorizer(train_df['risk_desc'].tolist(), self.vectorizer_type, train_Y)
            test_X = self.transform_vectorizer(test_df['risk_desc'].tolist(), self.vectorizer_type)

        # Fit Model
        self.fit_model(train_X, train_Y, self.n_estimators)

        # Prediction
        prediction = self.predict_model(test_X)

        # Evaluation
        self.evaluation(prediction, test_Y)

        return self
