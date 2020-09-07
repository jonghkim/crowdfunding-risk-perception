from abc import ABCMeta, abstractmethod
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

class BasePredictor:

    __metaclass__ = ABCMeta

    @abstractmethod
    def set_data(self):
        pass

    @abstractmethod
    def get_vectorizer(self):
        pass

    @abstractmethod
    def get_label(self):
        pass

    @abstractmethod
    def fit_transform(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @classmethod
    def tfidf_vectorizer(self, train_df, test_df, min_df=30, max_features=3000):
        train_df.dropna(subset=['risk_desc'],inplace=True)
        
        print("    Number of Documents: {}".format(train_df.shape[0]))
        vectorizer = TfidfVectorizer(min_df=min_df, max_features=max_features)

        train_X = vectorizer.fit_transform(train_df['risk_desc'].tolist())
        train_X = train_X.toarray()

        word_list = vectorizer.get_feature_names()       
        
        test_X = vectorizer.transform(test_df['risk_desc'].tolist())
        test_X = test_X.toarray()

        return train_X, test_X, word_list

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
