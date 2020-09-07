from abc import ABCMeta, abstractmethod
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class BasePredictor:

    __metaclass__ = ABCMeta

    @abstractmethod
    def fit_transform(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @classmethod
    def get_confusion_matrix(self, prediction, true_label):
        print("===== Confusion Matrix =====")
        print(confusion_matrix(prediction, true_label))  
        return self
    
    @classmethod
    def get_classification_report(self, prediction, true_label):
        print("===== Classification Report =====")        
        print(classification_report(prediction, true_label))  
        return self

    @classmethod
    def get_accuracy_score(self, prediction, true_label):
        print("===== Accuracy Score =====")        
        print(accuracy_score(prediction, true_label))
        return self
