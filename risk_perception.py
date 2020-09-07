import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import os

from config.config import get_config

from models.pipeline.text_normalizer.text_normalizer import TextNormalizer
from models.pipeline.label_generator.label_generator import LabelGenerator

class RiskPerception:
    def __init__(self):
        config = get_config()

        ##### Data #####
        self.data_dir = config['data_dir']
        self.training_data = config['training_data']
        self.test_data = config['test_data']

        ##### Preprocessing #####
        self.user_type = config['user_type']
        self.label_type = config['label_type']
        self.train_test_split_ratio = config['train_test_split_ratio']
    
    def get_data(self):        
        raw_df = pd.read_csv(os.path.join(self.data_dir, self.training_data), engine='python')
        raw_df.columns = ['project_id','project_title','project_short_desc','project_url','risk_desc',
                          'perceived_risk', 'experience', 'age', 'gender'] 

        print("##### Raw Data Load Start #####")
        print(raw_df.head())

        return raw_df

    def get_project_level_perceived_risk(self, raw_df, user_type):

        if user_type == 'all':
            selected_raw_df = raw_df
        elif user_type == 'experienced':
            selected_raw_df = raw_df[raw_df['experience']=='yes']

        perceived_risk_list = []

        for pid, sub_df in selected_raw_df.groupby('project_id'):
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

    def label_generation(self, perceived_risk_df, label_type):

        label_generator = LabelGenerator()
        label = label_generator.get_label(perceived_risk_df['perceived_risk'].tolist(), label_type)

        perceived_risk_df['label'] = label

        return perceived_risk_df

    def preproceesing(self, raw_df, user_type):
        # Aggregate into Project-Level Data
        perceived_risk_df = self.get_project_level_perceived_risk(raw_df, user_type)

        # Get Label
        perceived_risk_df = self.label_generation(perceived_risk_df, self.label_type)

        # Normalize Risk Description
        perceived_risk_df = self.normalizing_risk_description(perceived_risk_df)

        return perceived_risk_df

    def train(self):
        pass

    def predict(self):
        pass

    def run(self):
        pass

if __name__ == '__main__':

    risk_perception = RiskPerception()
    risk_perception.run()