import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import os

from config.config import get_config

class RiskPerception:
    def __init__(self):
        config = get_config()

        ##### Risk Perception Raw Data #####
        self.data_dir = config['data_dir']

    def get_data(self):
        pass

    def preproceesing(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def run(self):
        pass

if __name__ == '__main__':

    risk_perception = RiskPerception()
    risk_perception.run()