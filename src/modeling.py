import collections
import math
import os
import sys

import numpy as np
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pylab as plt
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential

sns.set_context("notebook", font_scale=1.)
sns.set_style("whitegrid")
import itertools
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import pandas as pd

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# class volumePredictor():
#
#     def __init__(self):
#         self.model = None
#         self.dataset = None
#
#     def fetch_dataset(self, filepath: str):
#         """
#         Fetches a compiled dataset in pickle form.
#         :param filepath: Relative Path to the dataset.
#         :return:
#         """
#         self.dataset = pd.read_pickle(module_path + filepath)
#


class sarimaModel():

    def __init__(self, ):

        self.model = None
        self.timestamp_range = None
        self.dataset = None

    def train(self, dataset, timedelta):
        """
        Trains the model based on the dataset
        :param dataset: target dataset that is loaded in the model
        :return:
        """
        self.timestamp_range = (pd.Timestamp(dataset.index.values[0]), pd.Timestamp(dataset.index.values[-1]))

        self.dataset = dataset

        opt_AIC = self.optimize_AIC(timedelta=timedelta)

        mod = sm.tsa.statespace.SARIMAX(self.dataset,
                                        order=(opt_AIC[0][0], opt_AIC[0][1], opt_AIC[0][2]),
                                        seasonal_order=(opt_AIC[0][0], opt_AIC[1][1], opt_AIC[1][2], timedelta),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        self.model = mod.fit()

        self.evaluate_results(visualize=True)

    def optimize_AIC(self, timedelta):
        """
        Optimizes the Seasonality, trend and noise parameters for the SARIMA model
        :param dataset: target dataset
        :return: Optimal AIC parameters
        """
        if type(self.dataset) != pd.DataFrame:
            raise Exception('Dataset not loaded. Please run train() first.')

        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], timedelta) for x in list(itertools.product(p, d, q))]
        opt_AIC = []
        min_AIC = sys.maxsize
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(self.dataset,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)

                    results = mod.fit(disp=0)
                    if results.aic < min_AIC:
                        min_AIC = results.aic
                        opt_AIC = (param, param_seasonal, results.aic)
                        # print('ARIMA{}x{}12 - AIC:{}'.format(opt_AIC[0], opt_AIC[1], opt_AIC[2]))

                except:
                    continue

        print('SARIMA{}x{}12 - AIC:{}'.format(opt_AIC[0], opt_AIC[1], opt_AIC[2]))
        return opt_AIC

    def evaluate_results(self, visualize: bool):
        """
        Evaluates the prediction results of the model
        :param visualize: Boolean visualization option
        :return:
        """
        print(self.model.summary().tables[1])

        if visualize:
            self.model.plot_diagnostics(figsize=(16, 8))
            plt.show()

    def perform_prediction(self, start_date: str, visualize: bool):
        """
        Performs the prediction of given a starting date that is within the range of the trained model.
        :param start_date: Starting date of the prediction
        :param visualize: Boolean visualization option
        :return:
        """

        if self.model == None:
            raise Exception('Model: {0} not trained, please perform a .train() first.'.format(self.__name__))

        if pd.Timestamp(start_date) < self.timestamp_range[0] or pd.Timestamp(start_date) >= self.timestamp_range[1]:
            raise Exception('Given start date is not within the range of the trained model.\n'
                            'Please give a range between {0} and {1}, or retrain'.format(*self.timestamp_range))

        pred = self.model.get_prediction(start=pd.to_datetime(start_date), dynamic=False)

        y_forecasted = pred.predicted_mean
        y_truth = self.dataset['ride_volume'][start_date:]
        mse = ((y_forecasted - y_truth) ** 2).mean()
        print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

        print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

        if visualize:
            self.visualize_prediction(pred)

    def visualize_prediction(self, prediction):
        """
        Visualizes the prediction of the model
        :param prediction:
        :return:
        """

        pred_ci = prediction.conf_int()
        ax = self.dataset['2015-10-01':].plot(label='observed')
        prediction.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume of Calls')
        plt.legend()
        plt.show()


class lstmModel():

    def __init__(self, n_hours=1):

        self.model = None
        self.timestamp_range = None
        self.dataset = None
        self.n_features = None
        self.n_hours = n_hours

    def train(self, dataset: pd.DataFrame, evaluate: bool, epochs=100, batch_size=16):
        """
        Trains the model based on the dataset
        :param dataset: target dataset that is loaded in the model
        :return:
        """
        self.timestamp_range = (pd.Timestamp(dataset.index.values[0]), pd.Timestamp(dataset.index.values[-1]))
        self.dataset = dataset
        self.n_features = len(dataset.columns)
        # Extract the correct month to train from
        reframed = self.prep_dataset_many_to_one(dataset)

        train_X, train_y, test_X, test_y = self.split_test_train(reframed=reframed, train_percentage=0.9)

        self.model = self.create_network(train_X_shape=train_X.shape)

        # fit network
        history = self.model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
                                 validation_data=(test_X, test_y),
                                 verbose=2,
                                 shuffle=False)
        if evaluate:
            self.evaluate_results(test_X, test_y)

    def generate_prediction(self, input_data: pd.DataFrame):
        """
        Generates a prediction on a given dataset
        :param input_data:
        :return:
        """
        if self.n_features == None:
            Warning('The dataset given does not fit the model\'s input shape. Trying to retrain.')
            self.train(dataset=input_data, evaluate=True)

        if self.n_features != len(input_data.columns):
            Warning('The dataset given does not fit the model\'s input shape. Trying to retrain.')
            self.train(dataset=input_data, evaluate=True)

        reframed = self.prep_dataset_many_to_one(input_data)

        test_X, test_y = self.reshape_to_3D(reframed.values)
        return self.evaluate_results(test_X, test_y)

    def evaluate_results(self, test_X, test_y):
        """
        Evaluates the results on a given test dataset
        :param test_X: Test input
        :param test_y: Expected output
        :return: Predictions, Ground_Truth
        """
        # # make a prediction
        yhat = self.model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], self.n_hours * self.n_features))
        # invert scaling for forecast
        print(test_X.shape, test_X)
        inv_yhat = np.concatenate((yhat, test_X[:, -(self.n_features - 1):]), axis=1)
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 0]
        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = np.concatenate((test_y, test_X[:, -(self.n_features - 1):]), axis=1)
        inv_y = self.scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 0]
        # calculate RMSE
        rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)
        return inv_yhat, inv_y

    def create_network(self, train_X_shape):
        """
        Create a network given the input's shape
        :param train_X_shape: The shape of the given input
        :return: the model's structure
        """

        # design network
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X_shape[1], train_X_shape[2])))

        model.add(Dense(1, activation='relu'))
        model.compile(loss='mae', optimizer='adam')
        return model

    def split_test_train(self, reframed, train_percentage=0.8):
        """
        Splits the reframed dataset into the test and train section with a given percentage
        :param reframed: The reshaped 3D dataset
        :param percentage: The percentage cut for train and test
        :return:
        """
        # split into train and test sets
        values = reframed.values
        n_train_hours = math.floor(train_percentage * len(values))
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]

        train_X, train_y = self.reshape_to_3D(train)
        test_X, test_y = self.reshape_to_3D(test)

        return train_X, train_y, test_X, test_y

    def reshape_to_3D(self, data):
        """
        Reshapes the data that are given serialised into a 3D shape in order to be able to be fed into the
        model. 1st dimention is the iterations of the data, 2nd is the hour window. 3rd are the features.
        :param data: serialised data
        :return: reshaped data
        """
        # split into input and outputs
        n_obs = self.n_hours * self.n_features
        data_X, data_y = data[:, :n_obs], data[:, - self.n_features]
        # reshape input to be 3D [samples, timesteps, features]
        data_X = data_X.reshape((data_X.shape[0], self.n_hours, self.n_features))
        return data_X, data_y

    def prep_dataset_many_to_one(self, dataset):
        """
        Transforms the dataset in a many-to-one shape in order to be fed to the LSTM.
        :param dataset: Target Dataset
        :parm lag_input: Number of lag observations as input (X).
        :param label_column_index: Index of the label column
        :return: reframed dataset in the form: var1(t-1)	var2(t-1)	var3(t-1)	var4(t-1)	var1(t)
        """
        if type(dataset) != pd.DataFrame:
            raise Exception('Dataset not loaded. Please run train() first.')

        # load dataset
        values = dataset.values

        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        self.scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaled = self.scaler.fit_transform(values)
        # frame as supervised learning
        reframed = self.series_to_supervised(scaled, lag_input=self.n_hours, n_out=1)

        return reframed

    def series_to_supervised(self, data, lag_input=1, n_out=1, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
        :parm lag_input: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(lag_input, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg


class visualizationModule():

    @staticmethod
    def plot_forecast(x, y,
                      forecast_mean, forecast_scale, forecast_samples,
                      title, x_locator=None, x_formatter=None):
        """Plot a forecast distribution against the 'true' time series."""
        colors = sns.color_palette()
        c1, c2 = colors[0], colors[1]
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)

        num_steps = len(y)
        num_steps_forecast = forecast_mean.shape[-1]
        num_steps_train = num_steps - num_steps_forecast

        ax.plot(x, y, lw=2, color=c1, label='ground truth')

        forecast_steps = np.arange(
            x[num_steps_train],
            x[num_steps_train] + num_steps_forecast,
            dtype=x.dtype)

        ax.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)

        ax.plot(forecast_steps, forecast_mean, lw=2, ls='--', color=c2,
                label='forecast')
        ax.fill_between(forecast_steps,
                        forecast_mean - 2 * forecast_scale,
                        forecast_mean + 2 * forecast_scale, color=c2, alpha=0.2)

        ymin, ymax = min(np.min(forecast_samples), np.min(y)), max(np.max(forecast_samples), np.max(y))
        yrange = ymax - ymin
        ax.set_ylim([ymin - yrange * 0.1, ymax + yrange * 0.1])
        ax.set_title("{}".format(title))
        ax.legend()

        if x_locator is not None:
            ax.xaxis.set_major_locator(x_locator)
            ax.xaxis.set_major_formatter(x_formatter)
            fig.autofmt_xdate()

        return fig, ax

    @staticmethod
    def plot_components(dates,
                        component_means_dict,
                        component_stddevs_dict,
                        x_locator=None,
                        x_formatter=None):
        """Plot the contributions of posterior components in a single figure."""
        colors = sns.color_palette()
        c1, c2 = colors[0], colors[1]

        axes_dict = collections.OrderedDict()
        num_components = len(component_means_dict)
        fig = plt.figure(figsize=(12, 2.5 * num_components))
        for i, component_name in enumerate(component_means_dict.keys()):
            component_mean = component_means_dict[component_name]
            component_stddev = component_stddevs_dict[component_name]

            ax = fig.add_subplot(num_components, 1, 1 + i)
            ax.plot(dates, component_mean, lw=2)
            ax.fill_between(dates,
                            component_mean - 2 * component_stddev,
                            component_mean + 2 * component_stddev,
                            color=c2, alpha=0.5)
            ax.set_title(component_name)
            if x_locator is not None:
                ax.xaxis.set_major_locator(x_locator)
                ax.xaxis.set_major_formatter(x_formatter)
            axes_dict[component_name] = ax
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig, axes_dict

    @staticmethod
    def plot_one_step_predictive(dates, observed_time_series,
                                 one_step_mean, one_step_scale,
                                 x_locator=None, x_formatter=None):
        """Plot a time series against a model's one-step predictions."""

        colors = sns.color_palette()
        c1, c2 = colors[0], colors[1]

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
        num_timesteps = one_step_mean.shape[-1]
        ax.plot(dates, observed_time_series, label="observed time series", color=c1)
        ax.plot(dates, one_step_mean, label="one-step prediction", color=c2)
        ax.fill_between(dates,
                        one_step_mean - one_step_scale,
                        one_step_mean + one_step_scale,
                        alpha=0.1, color=c2)
        ax.legend()

        if x_locator is not None:
            ax.xaxis.set_major_locator(x_locator)
            ax.xaxis.set_major_formatter(x_formatter)
            fig.autofmt_xdate()
        fig.tight_layout()
        return fig, ax
