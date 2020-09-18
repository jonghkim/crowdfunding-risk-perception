from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class VectorizerCorrelationFiltering:
    def count_vectorizer(self, risk_desc_list, min_df=10, max_features=1000):
        
        print("    Number of Section Text: {}".format(len(risk_desc_list)))

        print("    Minimum Document Frequency: {}".format(min_df))
        print("    Maximum Number of Features: {}".format(max_features))

        vectorizer = CountVectorizer(min_df=min_df, max_features=max_features)

        train_X = vectorizer.fit_transform(risk_desc_list)
        train_X = train_X.toarray()

        word_list = vectorizer.get_feature_names()       
        
        return train_X, word_list, vectorizer
        
    # Correlation Filtering
    def corr_filter(self, X, y, alpha, kappa):
        X_binary = np.where(X > 0, 1, 0)
        y_binary = np.where(y > 0, 1, 0).reshape(-1,1)    

        k = np.sum(X_binary, axis=0)
        f = np.sum(X_binary * y_binary, axis=0) / k

        kappa = np.partition(k.flatten(), int(len(k)*kappa))[int(len(k)*kappa)]
        index_kappa = np.where(k>=kappa)

        print("    kappa: {}".format(kappa))
        print("    # of words after kappa filter: {}".format(len(index_kappa[0])))

        f_order_index = np.argsort(f)
        f_order_index = f_order_index[np.isin(f_order_index, index_kappa)]

        alpha_plus_score = f[f_order_index[-int(alpha*len(f_order_index))]]
        alpha_minus_score = f[f_order_index[int(alpha*len(f_order_index))]]

        print("    alpha_plus: {}".format(alpha_plus_score))
        print("    alpha_minus: {}".format(alpha_minus_score))

        S_plus = f_order_index[-int(alpha*len(f_order_index)):]
        S_minus = f_order_index[:int(alpha*len(f_order_index))]

        print("    # of Positive Words: {}".format(len(S_plus)))
        print("    # of Negative Words: {}".format(len(S_minus)))

        S_hat = np.append(S_plus, S_minus)

        return S_hat, S_plus, S_minus

    def fit_transform(self, risk_desc_list, label, min_df=10, max_features=1000, alpha=0.1, kappa=0.01):
        train_X, word_list, vectorizer = self.count_vectorizer(risk_desc_list, min_df, max_features)
        S_hat, S_plus, S_minus = self.corr_filter(train_X, label, alpha, kappa)

        train_D_S = train_X[:, S_hat]
        plus_word_list = [word_list[index] for index in S_plus]
        minus_word_list = [word_list[index] for index in S_minus]

        return train_D_S, plus_word_list, minus_word_list, vectorizer, S_hat
