import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import os

from config.config import get_config

class Preprocessing:
    def __init__(self):
        config = get_config()

        ##### Market Data #####
        self.data_dir = config['data_dir']
