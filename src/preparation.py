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
        '''
        Checks if we have the correct headers loaded
        :param dataset: Target Dataset
        :return:
        '''
        for header in dataset.columns:
            if header not in self.header:
                raise Exception('Column name {0} not in regex model.'.format(header))
        return

    def evaluate_dataframe(self, dataset: pd.DataFrame):
        '''
        Evaluates a given regex model to a target dataset
        :param dataset: Target Dataset
        :return:
        '''
        self.evaluate_headers(dataset=dataset)

        # Per column evaluation
        for idx, col in enumerate(dataset):
            self.evaluate_column(model=self.model, dataset=dataset, column=col)

        return dataset

    # def evaluate_row(self, model: dict, row: list, row_idx: int):
    #     '''
    #     Evaluates a single row with a given regex model
    #     :param model:
    #     :param row:
    #     :param row_idx:
    #     :return:
    #     '''
    #
    #     header = list(model.keys())
    #
    #     for idx, item in enumerate(row):
    #         pattern = model[header[idx]]
    #         if not pattern.match(str(item)):
    #             raise Exception('Pattern mismatch at row: {0}, column: {1}'.format(row_idx, header[idx]))

    def evaluate_column(self, model: dict, dataset: pd.DataFrame, column: str):
        '''
        Evaluates a dataset's with a given model
        :param dataset: Target Dataset
        :param column: Target column name
        :return:
        '''
        try:
            pattern = model[column]
        except Exception as ex:
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
            print('Dataset Evaluation Passed')

        return dataset


class outlierDetection():

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def generate_time_aggregations(self, aggregation_column: str):
        '''
        Generate the lists of sums for quarter, hour, day and week
        :return:
        '''
        if aggregation_column not in self.dataset.columns:
            raise Exception('Aggregation column name not in loaded dataset')

        if type(self.dataset[aggregation_column]) != pd.DatetimeIndex:
            raise Exception('Dataset DateTime column must be a DatetimeIndex type.')

        quart_sum = [0]
        hour_sum = [0]
        day_sum = [0]
        week_sum = [0]

        start = self.dataset['request_date'][0]
        quart, hour, day, week = (start.minute // 15, start.hour, start.weekday(), start.day // 7)

        for date in self.dataset['request_date']:

            # Quarts
            if date.minute // 15 == quart:
                quart_sum[-1] += 1
            else:
                quart_sum.append(1)
                quart = date.minute // 15

            # Hours
            if date.hour == hour:
                hour_sum[-1] += 1
            else:
                hour_sum.append(1)
                hour = date.hour
            # Days
            if date.day == day:
                day_sum[-1] += 1
            else:
                day_sum.append(1)
                day = date.day

            # Weeks
            if date.day // 7 == week:
                week_sum[-1] += 1

            else:
                week_sum.append(1)
                week = day // 7

        return quart_sum, hour_sum, day_sum, week_sum

    def z_detect_outliers(self, values: list, threshold: int):
        '''
        Generates the positions of the values that have a z index higher than a given threshold
        :param threshold: Allowed Threshold
        :return: outlier positions
        '''
        z = np.abs(stats.zscore(values))
        return np.where(z > threshold)




# cI = csvInterface()
#
# cI.csv_to_dataframe(source=Path('/Users/noresources/Pycharm_projects/justbeatit/data/routes.csv'), separator='	',
#                     evaluate=True)


