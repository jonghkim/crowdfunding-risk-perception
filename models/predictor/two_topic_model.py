import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, precision_score, recall_score, accuracy_score

from models.predictor.base_predictor import BasePredictor
from models.pipeline.label_generator.label_generator import LabelGenerator

from scipy.optimize import minimize

class TwoTopicModel(BasePredictor):
    def __init__(self):
        self.vectorizer = None
        self.S_hat = None
        self.S_plus = None
        self.S_minus = None
        
        self.word_list = None
        self.O_hat = None

    def set_config(self, params):
        self.config = params

        # vectorizer
        self.min_df = params['vectorizer']['min_df']
        self.max_features = params['vectorizer']['max_features']

        # predictor
        self.model_type = params['predictor']['model_type']
        self.prediction_label = params['predictor']['prediction_label']
        self.user_type = params['predictor']['user_type']        
        self.label_type = params['predictor']['label_type']        
        self.k_fold_cv = params['predictor']['k_fold_cv']        

        self.alpha_plus = params['predictor']['hyperparams']['alpha_plus']
        self.alpha_minus = params['predictor']['hyperparams']['alpha_minus']
        self.kappa = params['predictor']['hyperparams']['kappa']
        self.lamb = params['predictor']['hyperparams']['lamb']

        return self        

    def get_label(self, df):
        label_generator = LabelGenerator()
        label = label_generator.get_label(df['perceived_risk'].tolist(), 'numerical_type1')
        label = np.array(label)
        
        return label

    def fit_vectorizer(self, train_df, min_df=10, max_features=1000):
        #min_df = int(sentiment_df.shape[0]*min_df)
        train_df.dropna(subset=['risk_desc'],inplace=True)
        print("    Number of Training Text: {}".format(train_df.shape[0]))

        print("    Minimum Document Frequency: {}".format(min_df))
        print("    Maximum Number of Features: {}".format(max_features))

        vectorizer = CountVectorizer(min_df=min_df, max_features=max_features)

        X = vectorizer.fit_transform(train_df['risk_desc'].tolist())
        X = X.toarray()
        
        word_list = vectorizer.get_feature_names()       
        self.vectorizer = vectorizer
        self.word_list = word_list

        return X
    
    def transform_vectorizer(self, text_list):
        X = self.vectorizer.transform(text_list)
        X = X.toarray()
        
        return X 

    def sreen_sentiment_charged_words(self, X, y, alpha_plus, alpha_minus, kappa, label_type):
        X_binary = np.where(X > 0, 1, 0)

        if label_type == 'categorical_type1':
            y_binary = np.where(y > 0, 1, 0).reshape(-1,1)    
        elif label_type == 'categorical_type2':
            y_binary = np.where(y >= 0, 1, 0).reshape(-1,1)    

        k = np.sum(X_binary, axis=0)
        f = np.sum(X_binary * y_binary, axis=0) / k

        kappa = np.partition(k.flatten(), int(len(k)*kappa))[int(len(k)*kappa)]
        index_kappa = np.where(k>=kappa)

        print("    kappa: {}".format(kappa))
        print("    # of words after kappa filter: {}".format(len(index_kappa[0])))

        f_order_index = np.argsort(f)
        f_order_index = f_order_index[np.isin(f_order_index, index_kappa)]
        
        # count based
        alpha_plus_score = f[f_order_index[-int(alpha_plus*len(f_order_index))]]
        alpha_minus_score = f[f_order_index[int(alpha_minus*len(f_order_index))]]
        
        print("    alpha_plus: {}".format(alpha_plus_score))
        print("    alpha_minus: {}".format(alpha_minus_score))

        S_plus = f_order_index[-int(alpha_plus*len(f_order_index)):]
        S_minus = f_order_index[:int(alpha_minus*len(f_order_index))]

        print("    # of Positive Words: {}".format(len(S_plus)))
        print("    # of Negative Words: {}".format(len(S_minus)))

        S_hat = np.append(S_plus, S_minus)

        return S_hat, S_plus, S_minus

    def estimate_two_topic(self, y, X, S_hat):
        # Estimation of p-hat based on Return Rank
        y_order_index = y.argsort()
        rank = y_order_index.argsort()+1
        
        p_hat = rank/len(y)
        
        # Estimation of Word Vector
        D_S = X[:, S_hat]
        S = np.sum(D_S, axis=1)
        S = S.reshape(-1,1)
        
        zero_mask = np.all(np.equal(S, 0), axis=1)
        
        S = S[~zero_mask]
        D_S = D_S[~zero_mask]
        p_hat = p_hat[~zero_mask]
        
        D_S_tilder = (D_S / S).T
        
        W = np.stack((p_hat, 1-p_hat))
        
        O_hat = D_S_tilder @ W.T @ np.linalg.inv(W @ W.T) # linear regression
        O_hat = O_hat.clip(min=0)
        O_hat = O_hat/np.linalg.norm(O_hat, ord=1, axis=0)
        
        return O_hat

    def penalized_log_likelihood(self, p, s, d_S, O_hat, lamb):
        p = p.reshape(-1,1)
        l_l = (1/s)*(d_S*np.log(p*O_hat[:,0].reshape((1,-1)) + (1-p)*O_hat[:,1].reshape((1,-1)))).sum() + lamb*np.log(p*(1-p))

        return l_l

    def maximum_likelihood_estimate(self, d, S_hat, S_plus, S_minus, O_hat, lamb):
        # Estimation of Word Vector
        d_S = d[S_hat]
        s = np.sum(d_S)
        
        s_plus = np.sum(d[S_plus])
        s_minus = np.sum(d[S_minus])
        
        if s_plus <= 15 or s_minus <=15:
            return None
            
        p_hat_zero = 1/2
                                        
        bnds = [(0,1)]
        p_hat = minimize(fun=self.penalized_log_likelihood,
                        x0=p_hat_zero, 
                        args=(s, d_S, O_hat, lamb),
                        method='L-BFGS-B', #SLSQP
                        bounds = bnds)    

        return p_hat['x'][0]

    def evaluate_model(self, X, Y, k_fold_cv):
        # k fold validation
        kf = KFold(n_splits=k_fold_cv)

        is_scores_list = []
        oos_scores_list = []

        for train_index, test_index in kf.split(X):
            train_X, test_X = X[train_index], X[test_index]
            train_Y, test_Y = Y[train_index], Y[test_index]

            self.fit_model(train_X, train_Y, self.alpha_plus, self.alpha_minus, self.kappa, self.label_type)

            train_Y_hat = self.predict_model(train_X) 
            is_scores = self.evaluation(train_Y_hat, train_Y)           
            is_scores_list.append(is_scores)


            test_Y_hat = self.predict_model(test_X)
            oos_scores = self.evaluation(test_Y_hat, test_Y)           
            oos_scores_list.append(oos_scores)

        is_scores_df = pd.DataFrame(is_scores_list)
        oos_scores_df = pd.DataFrame(oos_scores_list)

        print("In-Sample Average Scores: ", is_scores_df.mean())
        print("Out-of-Sample Average Scores: ", oos_scores_df.mean())

        return oos_scores_df.mean()

    def fit_model(self, X, y, alpha_plus=0.25, alpha_minus=0.25, kappa=0.01, label_type='categorical_type1'):
        
        S_hat, S_plus, S_minus = self.sreen_sentiment_charged_words(X, y, alpha_plus, alpha_minus, kappa, label_type)
        O_hat = self.estimate_two_topic(y, X, S_hat)
        
        self.S_hat, self.O_hat = S_hat, O_hat
        return self
    
    def predict(self, d, S_hat, S_plus, S_minus, O_hat, lamb=5):
        sentiment = self.maximum_likelihood_estimate(d, S_hat, S_plus, S_minus, O_hat, lamb)

        return sentiment    

    def predict_model(self, test_X):
    
        prediction_list = []

        for x in test_X:
            prediction_list.append(self.predict(x, self.S_hat, self.S_plus, self.S_minus, self.O_hat, self.lamb))

        return prediction_list

    def get_topic_df(self):

        word_list = self.word_list
        word_list = [word_list[index] for index in self.S_hat]

        topic_df = pd.DataFrame(self.O_hat, columns=['high_risk_topic_importance', 'low_risk_topic_importance'])
        topic_df['word_list'] = word_list
        topic_df['topic_importance_diff'] = topic_df['high_risk_topic_importance'] - topic_df['low_risk_topic_importance']
        topic_df = topic_df[['word_list','high_risk_topic_importance','low_risk_topic_importance', 'topic_importance_diff']]

        return topic_df

    def evaluation(self, prediction_list, test_Y):

        scores = {}
        
        print("### Model - Two Topic Model ###")
        print("#### Setting: ", self.config)
    
        prediction_np = np.array(prediction_list)
        prediction_np[prediction_np==None]=0.5
        
        # Continuous Value Evaluation
        ## MAE Evaluation
        scores['mae'] = self.mean_absolute_error(prediction_np*4+1, test_Y*2+3)
        ## R^2 Evaluation
        scores['r_2'] = r2_score(test_Y*2+3, prediction_np*4+1)

        # Categorical Evaluation
        if self.label_type == 'categorical_type1':
            prediction_category = np.array([1 if score > 0.5 else 0 for score in prediction_np])
            test_Y_category = np.array([1 if score > 0 else 0 for score in test_Y])
            
        elif self.label_type == 'categorical_type2':
            prediction_category = np.array([1 if score >= 0.5 else 0 for score in prediction_np])
            test_Y_category = np.array([1 if score >= 0 else 0 for score in test_Y])

        scores['acc'] = accuracy_score(prediction_category, test_Y_category)
        scores['precision'] = precision_score(test_Y_category, prediction_category, average='micro')
        scores['recall'] = recall_score(test_Y_category, prediction_category, average='micro')

        return scores        

    def run(self, perceived_risk_df, prediction_df, params):
        print("# Two Topic Model")
        
        # Set Config
        self.set_config(params)

        # Get Label
        Y = self.get_label(perceived_risk_df)
        # Train Model
        X = self.fit_vectorizer(perceived_risk_df)

        # evaluate_model
        scores = self.evaluate_model(X, Y, self.k_fold_cv)

        # Train Final Model
        self.fit_model(X, Y, self.alpha_plus, self.alpha_minus, self.kappa, self.label_type)

        # Two Topic Save
        word_df = self.get_topic_df()
        word_df.to_csv('results/two_topic_score_{}_usr_type_{}_wv_size_{}_alpha_{}_kappa_{}_mae_{:.2f}_acc_{:.2f}.csv'.format(self.label_type, self.user_type, word_df.shape[0],self.alpha_plus, self.kappa, scores['mae'], scores['acc']))

        # Vectorizer
        if self.prediction_label == 'desc_combined':
            prediction_df['desc_combined'] =  prediction_df["desc_total"] + " " + prediction_df["risk_desc"]

        prediction_X = self.transform_vectorizer(prediction_df[self.prediction_label].tolist(), self.vectorizer_type)
        
        # Prediction
        prediction_Y_hat = self.predict_model(prediction_X)
        
        prediction_df['prediction'] = [val*4+1 if val != None else None for val in prediction_Y_hat]
        prediction_df.to_csv('results/{}_{}_usr_type_{}_wv_size_{}_alpha_{}_kappa_{}_mae_{:.2f}_acc_{:.2f}.csv'.format('two_topic_model', self.label_type, self.user_type, word_df.shape[0], self.alpha_plus, self.kappa, scores['mae'], scores['acc']))

        return self