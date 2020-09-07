from models.predictor.base_predictor import BasePredictor
from sklearn.ensemble import RandomForestClassifier

class RandomForest(BasePredictor):

    def set_config(self, params):
        pass

    def get_vectorizer(self, ):
        pass

    def get_label(self):
        pass

    def fit_transform(self):
        model1_text_classifier = RandomForestClassifier(n_estimators=3, random_state=0)  

        model1_text_classifier.fit(train_X, train_category_type1_Y)        
        pass

    def transform(self):
        model1_prediction = model1_text_classifier.predict(test_X)        
        
        pass

    def run(self, train_df, test_df, params):
        pass    