# import osmnx as ox
# G = ox.graph_from_place('Manhattan Island, New York City, New York, USA', network_type='drive')
# ox.plot_graph(G)


from pathlib import Path
import pandas as pd
from src.preparation import csvInterface
import numpy as np
from scipy import stats


class dataProc():

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def generate_time_aggregations(self, aggregation_column: str):
        """
        Generate the lists of sums for quarter, hour, day and week
        :return:
        """
        if aggregation_column not in self.dataset.columns:
            raise Exception('Aggregation column name not in loaded dataset')

        if type(self.dataset[aggregation_column][0]) != pd._libs.tslibs.timestamps.Timestamp:
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

    @staticmethod
    def time_to_timestamp(dataset: pd.DataFrame, datetime_column: str):
        """
        Converts a dataset's datetime string into a pandas timestamp
        :param dataset: input dataset
        :param datetime_column: name of target column
        :return: converted dataset
        """
        if datetime_column not in dataset.columns:
            raise Exception('Aggregation column name not in loaded dataset')

        dataset[datetime_column] = pd.to_datetime(dataset[datetime_column], format='%Y-%m-%d %H:%M:%S')
        return dataset

    @staticmethod
    def aggregate_by_hours(dataset: pd.DataFrame, datetime_column: str, hours: int):
        """
        Aggregates the input dataset by hour
        :param dataset: Input dataset
        :param datetime_column: Timestamp column name
        :return:
        """
        if hours <= 0:
            raise Exception('A positive integer of hours must be chosen')
        if datetime_column not in dataset.columns:
            raise Exception('Aggregation column name not in loaded dataset')

        return pd.DataFrame(dataset.set_index(datetime_column).
                            resample('{0}h'.format(hours)).apply(np.sum))

    @staticmethod
    def aggregate_by_mins(dataset: pd.DataFrame, datetime_column: str, minutes: int):
        """
        Aggregates the input dataset by 15 minutes
        :param dataset: Input dataset
        :param datetime_column: Timestamp column name
        :return:
        """
        if minutes <= 0:
            raise Exception('A positive integer of minutes must be chosen')
        print(minutes * 60)
        if datetime_column not in dataset.columns:
            raise Exception('Aggregation column name not in loaded dataset')

        return pd.DataFrame(dataset.set_index(datetime_column).
                            resample('{0}s'.format(minutes * 60)).apply(np.sum))

    @staticmethod
    def z_detect_outliers(values: list, threshold: int):
        """
        Generates the positions of the values that have a z index higher than a given threshold
        :param threshold: Allowed Threshold
        :return: outlier positions
        """
        z = np.abs(stats.zscore(values))
        return np.where(z > threshold)
