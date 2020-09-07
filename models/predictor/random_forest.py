from models.predictor.base_predictor import BasePredictor
from sklearn.ensemble import RandomForestClassifier
import numpy as np

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

        # predictor
        self.model_type = params['predictor']['model_type']
        self.n_estimators = params['predictor']['n_estimators']

        return self

    def fit_vectorizer(self, risk_desc_list, vectorizer_type):
        if vectorizer_type == 'tf_idf':
            train_X, word_list, self.vectorizer = self.tfidf_vectorizer(risk_desc_list, min_df=self.min_df, max_features=self.max_features)
            
        elif vectorizer_type == 'correlation_filtering':
            pass

        return train_X, word_list

    def transform_vectorizer(self, risk_desc_list):
        test_X = self.vectorizer.transform(risk_desc_list)
        test_X = test_X.toarray()

        return test_X

    def get_label(self, df):
        label = np.array(df['label'].tolist())

        return label

    def fit_model(self, train_X, train_Y, n_estimators):
        self.text_classifier = RandomForestClassifier(n_estimators=n_estimators)  
        self.text_classifier.fit(train_X, train_Y)        

        return self

    def predict_model(self, test_X):
        prediction = self.text_classifier.predict(test_X)
        
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
        
        # Get Features
        train_X, word_list = self.fit_vectorizer(train_df['risk_desc'].tolist(), self.vectorizer_type)
        test_X = self.transform_vectorizer(test_df['risk_desc'].tolist())

        # Get Label
        train_Y = self.get_label(train_df)
        test_Y = self.get_label(test_df)

        # Fit Model
        self.fit_model(train_X, train_Y, self.n_estimators)

        # Prediction
        prediction = self.predict_model(test_X)

        # Evaluation
        self.evaluation(prediction, test_Y)
        
        return self
