import os
import sys
import unittest

import pandas as pd

from src.modeling import sarimaModel, lstmModel
from src.processing import dataProc
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

DATASET_PATH = module_path + "/notebook/dt_agg1hour.h5"


class TestSarimaModel(unittest.TestCase):

    def setUp(self):
        self.sarmodel = sarimaModel()

        self.start_date = '2015-11-10'
        self.prediction_start_date = '2015-11-24'
        self.end_date = '2015-11-25'

        dataset = pd.read_pickle(
            DATASET_PATH)
        self.dataset = dataset[self.start_date:self.end_date]

    def test_train(self):
        self.sarmodel.train(dataset=self.dataset, evaluate= True, timedelta=12)
        self.sarmodel.perform_prediction(start_date=self.prediction_start_date, visualize=True)

    def test_bad_date(self):
        self.sarmodel.train(dataset=self.dataset, evaluate=True, timedelta=12)
        with self.assertRaises(Exception): self.sarmodel.perform_prediction(start_date='2018-11-25', visualize=False)


class TestLstmModel(unittest.TestCase):

    def setUp(self):
        self.lstmodel = lstmModel()

        self.start_date = '2015-11-10'
        self.prediction_start_date = '2015-11-25'
        self.end_date = '2015-11-25'

        dataset = pd.read_pickle(
            DATASET_PATH)
        self.dataset = dataset[self.start_date:self.end_date]
        self.dataset = dataProc.create_features(dataset=dataset)

    def test_train(self):

        self.lstmodel.train(dataset=self.dataset, evaluate=True)

    def test_predict(self):
        y_pred, y = self.lstmodel.generate_prediction(input_data=self.dataset)
