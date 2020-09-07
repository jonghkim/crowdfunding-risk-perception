from abc import ABCMeta, abstractmethod
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

class BasePredictor:

    __metaclass__ = ABCMeta

    @abstractmethod
    def set_config(self):
        pass

    @abstractmethod
    def fit_vectorizer(self):
        pass

    @abstractmethod
    def transform_vectorizer(self):
        pass

    @abstractmethod
    def get_label(self):
        pass

    @abstractmethod
    def fit_model(self):
        pass

    @abstractmethod
    def predict_model(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @classmethod
    def tfidf_vectorizer(self, risk_desc_list, min_df=30, max_features=3000):        
        print("    Number of Documents: {}".format(len(risk_desc_list)))
        vectorizer = TfidfVectorizer(min_df=min_df, max_features=max_features)

        train_X = vectorizer.fit_transform(risk_desc_list)
        train_X = train_X.toarray()

        word_list = vectorizer.get_feature_names()       

        return train_X, word_list, vectorizer

    @classmethod
    def get_confusion_matrix(self, prediction, true_label):
        print("#### Confusion Matrix ####")
        print(confusion_matrix(prediction, true_label))  
        return self
    
    @classmethod
    def get_classification_report(self, prediction, true_label):
        print("#### Classification Report ####")
        print(classification_report(prediction, true_label))  
        return self

    @classmethod
    def get_accuracy_score(self, prediction, true_label):
        print("#### Accuracy Score ####")
        print(accuracy_score(prediction, true_label))
        return self
