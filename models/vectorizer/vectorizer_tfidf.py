from sklearn.feature_extraction.text import TfidfVectorizer

class VectorizerTfidf:
    def fit_transform(self, risk_desc_list, min_df=30, max_features=3000):        
        print("    Number of Documents: {}".format(len(risk_desc_list)))
        vectorizer = TfidfVectorizer(min_df=min_df, max_features=max_features)

        train_X = vectorizer.fit_transform(risk_desc_list)
        train_X = train_X.toarray()

        word_list = vectorizer.get_feature_names()       

        return train_X, word_list, vectorizer