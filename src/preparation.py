import re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


class csvInterface():

    def __init__(self):

        self.model = {
            'passenger_id': '[0-9]+',
            'source_latitude': '-[0-9]+.[0-9]+',
            'source_longitude': '-[0-9]+.[0-9]+',
            'source_address': '\s*',
            'destination_latitude': '-[0-9]+.[0-9]+',
            'destination_longitude': '-[0-9]+.[0-9]+',
            'destination_address': '\s*',
            'request_date': '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}'
        }
        self.model.update((col_name, re.compile(pat)) for col_name, pat in self.model.items())

        self.header = list(self.model.keys())

    def update_model(self, model: dict):
        """
        Updates the model of the csv interface
        :param model: dictionary with Key: column_name | Value: Regular Expression Pattern
        :return:
        """
        self.model = model
        self.model.update((col_name, re.compile(pat)) for col_name, pat in self.model.items())
        self.header = list(self.model.keys())

    def evaluate_headers(self, dataset: pd.DataFrame):
        """
        Checks if we have the correct headers loaded
        :param dataset: Target Dataset
        :return:
        """

        for header in dataset.columns:
            if header not in self.header:
                raise Exception('Column name {0} not in regex model.'.format(header))
        return

    def evaluate_dataframe(self, dataset: pd.DataFrame):
        """
        Evaluates a given regex model to a target dataset
        :param dataset: Target Dataset
        :return:
        """

        self.evaluate_headers(dataset=dataset)

        # Per column evaluation
        for idx, col in enumerate(dataset):
            self.evaluate_column(model=self.model, dataset=dataset, column=col)

        return dataset

    @staticmethod
    def evaluate_row(model: dict, row: list, row_idx: int):
        """
        Evaluates a single row with a given regex model
        :param model:
        :param row:
        :param row_idx:
        :return:
        """

        header = list(model.keys())

        for idx, item in enumerate(row):
            pattern = model[header[idx]]
            if not pattern.match(str(item)):
                raise Exception('Pattern mismatch at row: {0}, column: {1}'.format(row_idx, header[idx]))

    @staticmethod
    def evaluate_column(model: dict, dataset: pd.DataFrame, column: str):
        """
        Evaluates a dataset's with a given model
        :param model: Target model for evaluation
        :param dataset: Target Dataset
        :param column: Target column name
        :return:
        """
        try:
            pattern = model[column]
        except Exception:
            raise Exception('Column name {0} not in regex model.'.format(column))

        for idx, item in enumerate(dataset[column]):
            if not pattern.match(str(item)):
                raise Exception('Pattern mismatch at row: {0}, column: {1}'.format(idx, column))

    def csv_to_dataframe(self, source: Path, separator: str, evaluate: bool):
        """
        Compiles a given csv into a Pandas Dataframe
        :return: Pandas Dataframe Object
        """
        dataset = pd.read_csv(
            source, sep=separator, low_memory=False)

        if evaluate:
            self.evaluate_dataframe(dataset)
        return dataset





# cI = csvInterface()
#
# cI.csv_to_dataframe(source=Path('/Users/noresources/Pycharm_projects/justbeatit/data/routes.csv'), separator='	',
#                     evaluate=True)
