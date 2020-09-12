import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import os

from config.config import get_config

from models.pipeline.text_normalizer.text_normalizer import TextNormalizer

from sklearn.model_selection import train_test_split

from models.predictor.random_forest import RandomForest
from models.predictor.two_topic_model import TwoTopicModel

class RiskPerception:
    def __init__(self):
        self.config = get_config()

        ##### Data #####
        self.data_dir = self.config['data_dir']
        self.labeled_data = self.config['labeled_data']
        self.prediction_data = self.config['prediction_data']

        ##### Preprocessing #####
        self.user_type = self.config['user_type']
        self.train_test_split_ratio = self.config['train_test_split_ratio']
    
    def get_data(self):        
        labeled_data_df = pd.read_csv(os.path.join(self.data_dir, self.labeled_data), engine='python')
        labeled_data_df.columns = ['project_id','project_title','project_short_desc','project_url','risk_desc',
                          'perceived_risk', 'experience', 'age', 'gender'] 

        print("##### Raw Data Load Start #####")
        print(labeled_data_df.head())

        prediction_df = pd.read_csv(os.path.join(self.data_dir, self.prediction_data), engine='python')

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
            item_dict['perceived_risk'] = sub_df['perceived_risk'].mean()
            perceived_risk_list.append(item_dict)
        
        perceived_risk_df = pd.DataFrame(perceived_risk_list)

        return perceived_risk_df

    def normalizing_risk_description(self, perceived_risk_df):
        print("## Normalizing Risk Description")

        text_normalizer = TextNormalizer()

        # Step 1. Standardize
        # - Lower Case
        print("Upper Case to Lower Case")
        perceived_risk_df = text_normalizer.lower_case(perceived_risk_df)
        # - Remove Space
        print("Remove Extra Space")
        perceived_risk_df = text_normalizer.remove_extra_spaces(perceived_risk_df)
        # - Expand Abbrviated Text
        print("Expand Abbreviated Words")
        perceived_risk_df = text_normalizer.expand_abbreviation(perceived_risk_df)
        # - Remove Non English Special Characters
        print("Remove Non-English")
        perceived_risk_df = text_normalizer.remove_non_english(perceived_risk_df)
        # Step 2. Remove Stop Words & Lemmatization
        print("Normalize Text with Lemmatization and Stop Words Removal")
        perceived_risk_df = text_normalizer.normalize_text(perceived_risk_df)

        return perceived_risk_df

    def preprocessing(self, training_raw_df, user_type):
        print("# Preprocessing")
        # Aggregate into Project-Level Data
        perceived_risk_df = self.get_project_level_perceived_risk(training_raw_df, user_type)

        # Normalize Risk Description
        perceived_risk_df = self.normalizing_risk_description(perceived_risk_df)

        print(perceived_risk_df.head())

        return perceived_risk_df

    def split_data(self, perceived_risk_df, train_test_split_ratio):
        train_df, test_df = train_test_split(perceived_risk_df, train_size=train_test_split_ratio)

        return train_df, test_df

    def fit_transform_models(self, train_df, test_df, prediction_df):
        """
        # Prediction Models for Categorical Label 
        ## Model1. TF-IDF + RandomForest
        model1_params = self.config['model1_params']

        model1_predictor = RandomForest()
        model1_predictor.run(train_df, test_df, prediction_df, model1_params)

        ## Model2. Correlation Filtering + RandomForest 
        model2_params = self.config['model2_params']

        model2_predictor = RandomForest()
        model2_predictor.run(train_df, test_df, prediction_df, model2_params)
        """
        ## Model3. Correlation Filtering + Two Topic Model
        model3_params = self.config['model3_params']
        model3_predictor = TwoTopicModel()
        model3_predictor.run(train_df, test_df, prediction_df, model3_params)

        # Prediction Models for Numerical Label
        ## Model4. TF-IDF + ElasticNet
        model4_params = self.config['model4_params']

        ## Model5. Correlation Filtering + ElasticNet        
        model5_params = self.config['model5_params']

        pass

    def run(self):
        labeled_data_df, prediction_df = self.get_data()

        perceived_risk_df = self.preprocessing(labeled_data_df, self.user_type)
        train_df, test_df = self.split_data(perceived_risk_df, self.train_test_split_ratio)

        self.fit_transform_models(train_df, test_df, prediction_df)
        pass

if __name__ == '__main__':

    risk_perception = RiskPerception()
    risk_perception.run()