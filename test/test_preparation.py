import unittest
from pathlib import Path
import pandas as pd
from src.preparation import csvInterface

BAD_CSV_PATH = Path('/Users/noresources/Pycharm_projects/justbeatit/data/routes.csv')
CSV_PATH = Path('/Users/noresources/Pycharm_projects/justbeatit/data/routes.csv')


class TestcsvInterface(unittest.TestCase):
    def setUp(self):
        self.Interface = csvInterface()

    def test_update_model(self):
        new_model = {
            'passenger_id': '[0-9]+',
            'source_latitude': '-[0-9]+.[0-9]+',
            'source_longitude': '-[0-9]+.[0-9]+',
            'source_address': '\s*',
            'destination_latitude': '-[0-9]+.[0-9]+',
            'destination_longitude': '-[0-9]+.[0-9]+',
            'destination_address': '\s*',
            'request_date': '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}'
        }
        self.Interface.update_model(model=new_model)

    def test_evaluate_column_wrong_column_name(self):
        with self.assertRaises(Exception): self.Interface.evaluate_column(model=self.Interface.model,
                                                                          dataset=pd.DataFrame(), column='a_col_name')

    # Each test method starts with the keyword test_
    def test_csv_to_dataframe_bad_dataset_with_evaluate(self):
        with self.assertRaises(Exception): self.Interface.csv_to_dataframe(source=BAD_CSV_PATH, separator='	',
                                                                           evaluate=True)

    def test_evaluate_row_good(self):
        dataset = self.Interface.csv_to_dataframe(source=BAD_CSV_PATH, separator='	',
                                                  evaluate=False)
        self.Interface.evaluate_row(self.Interface.model, row=list(dataset.iloc[2]), row_idx=2)

    def test_evaluate_row_bad(self):
        dataset = self.Interface.csv_to_dataframe(source=BAD_CSV_PATH, separator='	',
                                                  evaluate=False)
        bad_model = {
            'passenger_id': '',
            'source_latitude': '',
            'source_longitude': '',
            'source_address': '\s*',
            'destination_latitude': '',
            'destination_longitude': '',
            'destination_address': '',
            'request_date': ''
        }

        with self.assertRaises(Exception): self.Interface.evaluate_row(bad_model, row=list(dataset.iloc[2]), row_idx=2)

    def test_evaluate_headers_bad(self):
        bad_dataset = pd.DataFrame()
        bad_dataset['foreign column'] = [0, 0, 0]
        with self.assertRaises(Exception): self.Interface.evaluate_headers(dataset=bad_dataset)


