import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import os

from config.config import get_config

from models.pipeline.text_normalizer.text_normalizer import TextNormalizer

from sklearn.model_selection import train_test_split

from models.predictor.random_forest import RandomForest
from models.predictor.two_topic_model import TwoTopicModel
from models.predictor.svm import SVM
from models.predictor.elastic_net import ElasticNet

class RiskPerception:
    def __init__(self):
        self.config = get_config()

        ##### Data #####
        self.data_dir = self.config['data_dir']
        self.labeled_data = self.config['labeled_data']
        self.prediction_data = self.config['prediction_data']

        ##### Preprocessing #####
        self.user_type = self.config['user_type']
    
    def get_data(self):        

        if 'normalized_' not in self.labeled_data:
            labeled_data_df = pd.read_csv(os.path.join(self.data_dir, self.labeled_data), engine='python', index_col=0)
            labeled_data_df.columns = ['project_id','project_title','project_short_desc','project_url','risk_desc',
                                        'perceived_risk', 'experience', 'age', 'gender', 'project_address'] 
        else:
            labeled_data_df = pd.read_csv(os.path.join(self.data_dir, self.labeled_data), encoding='utf-8', index_col=0)

        print("##### Training Data #####")
        print(labeled_data_df.head())

        prediction_df = pd.read_csv(os.path.join(self.data_dir, self.prediction_data), encoding='utf-8', index_col=0)

        print("##### Prediction Data #####")
        print(prediction_df.head())

        return labeled_data_df, prediction_df

    def get_project_level_perceived_risk(self, labeled_data_df, user_type):

        if user_type == 'all':
            selected_df = labeled_data_df
        elif user_type == 'experienced':
            selected_df = labeled_data_df[labeled_data_df['experience']=='yes']

        perceived_risk_list = []

        for pid, sub_df in selected_df.groupby('project_id'):
            item_dict = {}
            item_dict['project_id'] = pid
            item_dict['risk_desc'] = sub_df['risk_desc'].iloc[0]
            item_dict['project_address'] = sub_df['project_address'].iloc[0]
            item_dict['perceived_risk'] = sub_df['perceived_risk'].mean()
            perceived_risk_list.append(item_dict)
        
        perceived_risk_df = pd.DataFrame(perceived_risk_list)

        return perceived_risk_df

    def normalizing_risk_description(self, df, cols):
        
        print("## Before NA Text Drop: ", df.shape[0])
        df.dropna(subset=cols, inplace=True)
        print("## After NA Text Drop: ", df.shape[0])

        for col in cols:
            df[col] = df[col].apply(str)

            print("## Normalizing Risk Description")
            text_normalizer = TextNormalizer()

            # Step 1. Standardize
            # - Lower Case
            print("Upper Case to Lower Case")
            df[col] = text_normalizer.lower_case(df[col])
            # - Remove Space
            print("Remove Extra Space")
            df[col] = text_normalizer.remove_extra_spaces(df[col])
            # - Expand Abbrviated Text
            print("Expand Abbreviated Words")
            df[col] = text_normalizer.expand_abbreviation(df[col])
            df[col] = df[col].apply(str)
            # - Remove Non English Special Characters
            print("Remove Non-English")
            df[col] = text_normalizer.remove_non_english(df[col])
            df[col] = df[col].apply(str)
            # Step 2. Remove Stop Words & Lemmatization
            print("Normalize Text with Lemmatization and Stop Words Removal")
            df[col] = text_normalizer.normalize_text(df[col])
            df[col] = df[col].apply(str)
            print("\n")
                
        return df

    def merge_training_with_prediction_df(self, perceived_risk_df, prediction_df):
        prediction_df = prediction_df.merge(perceived_risk_df[['project_address', 'perceived_risk']], how='left', on='project_address')

        return prediction_df

    def fit_transform_models(self, perceived_risk_df, prediction_df):
      
        # Prediction Models for Categorical Label 
        ## Model1. TF-IDF + RandomForest
        model1_params = self.config['model1_params']
        model1_predictor = RandomForest()
        model1_predictor.run(perceived_risk_df, prediction_df, model1_params)

        ## Model2. Correlation Filtering + RandomForest 
        model2_params = self.config['model2_params']
        model2_predictor = RandomForest()
        model2_predictor.run(perceived_risk_df, prediction_df, model2_params)
        
        ## Model3. Correlation Filtering + Two Topic Model
        model3_params = self.config['model3_params']
        model3_predictor = TwoTopicModel()
        model3_predictor.run(perceived_risk_df, prediction_df, model3_params)
        
        # Prediction Models for Numerical Label
        ## Model4. TF-IDF + ElasticNet
        model4_params = self.config['model4_params']
        model4_predictor = SVM()
        model4_predictor.run(perceived_risk_df, prediction_df, model4_params)

        ## Model5. Correlation Filtering + ElasticNet        
        model5_params = self.config['model5_params']
        model5_predictor = SVM()
        model5_predictor.run(perceived_risk_df, prediction_df, model5_params)
        
        ## Model6. TF-IDF + ElasticNet
        model6_params = self.config['model6_params']
        model6_predictor = ElasticNet()
        model6_predictor.run(perceived_risk_df, prediction_df, model6_params)
    
        ## Model7. Correlation Filtering + ElasticNet        
        model7_params = self.config['model7_params']
        model7_predictor = ElasticNet()
        model7_predictor.run(perceived_risk_df, prediction_df, model7_params)

        return self

    def run(self):
        # Load Data
        labeled_data_df, prediction_df = self.get_data()        
        
        if 'normalized_' not in self.labeled_data:
            # Aggregate into Project-Level Data
            perceived_risk_df = self.get_project_level_perceived_risk(labeled_data_df, self.user_type)
        
            # Normalize Risk Description
            perceived_risk_df = self.normalizing_risk_description(perceived_risk_df, ['risk_desc'])
        else:
            perceived_risk_df = labeled_data_df

        if 'normalized_' not in self.prediction_data:
            prediction_df = self.normalizing_risk_description(prediction_df, ['risk_desc', 'desc_total'])
            # Merge Label Info
            prediction_df = self.merge_training_with_prediction_df(perceived_risk_df, prediction_df)
        
        # Training        
        self.fit_transform_models(perceived_risk_df, prediction_df)

        return self

if __name__ == '__main__':

    risk_perception = RiskPerception()
    risk_perception.run()