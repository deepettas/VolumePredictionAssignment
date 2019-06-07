import os
import sys
import unittest
from pathlib import Path

import pandas as pd

from src.processing import dataProc

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

BAD_CSV_PATH = Path(module_path + '/data/raw/routes.csv')
CSV_PATH = Path(module_path + '/data/raw/routes.csv')


class Testdataproc(unittest.TestCase):
    def setUp(self):
        self.dataset = pd.read_csv(
            CSV_PATH, sep='	', low_memory=False)
        self.dataset['request_date'] = pd.to_datetime(self.dataset['request_date'], format='%Y-%m-%d %H:%M:%S')

        self.outlier_detector = dataProc(self.dataset)

    def test_aggregations(self):
        quart_sum = dataProc.aggregate_by_mins(
            dataset=self.dataset, datetime_column='request_date', minutes=15)

        hour_sum = dataProc.aggregate_by_hours(
            dataset=self.dataset, datetime_column='request_date', hours=1)

    def test_create_features(self):
        dataset = dataProc.create_features(self.dataset)

    def test_generate_time_aggregations_good(self):
        self.outlier_detector.generate_time_aggregations(aggregation_column='request_date')

    def test_generate_time_aggregations_bad_name(self):
        with self.assertRaises(Exception): self.outlier_detector.generate_time_aggregations(
            aggregation_column='id')

    def test_generate_time_aggregations_bad_type_col(self):
        with self.assertRaises(Exception): self.outlier_detector.generate_time_aggregations(
            aggregation_column='passenger_id')

    def test_z_detect_outliers(self):
        quart_sum, hour_sum, day_sum, week_sum = self.outlier_detector.generate_time_aggregations(
            aggregation_column='request_date')
        self.outlier_detector.z_detect_outliers(hour_sum, 3)
