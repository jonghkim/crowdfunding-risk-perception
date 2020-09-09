from abc import ABCMeta, abstractmethod
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error

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
    def mean_absolute_error(self, prediction, true_label):
        print("#### Mean Absoulte Error ####")
        print(mean_absolute_error(true_label, prediction))  
        return self
    
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
