from src.preparation import csvInterface
from src.processing import dataProc

import matplotlib as mpl
from matplotlib import pylab as plt
import matplotlib.dates as mdates
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
import collections
from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts
import numpy as np
import matplotlib as mpl
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm

import tensorflow as tf
import tensorflow_probability as tfp
import os,sys
sns.set_context("notebook", font_scale=1.)
sns.set_style("whitegrid")
import itertools

import pandas as pd
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


class volumePredictor():

    def __init__(self):

        self.model = None
        self.dataset = None

    def fetch_dataset(self, filepath:str):
        """
        Fetches a compiled dataset in pickle form.
        :param filepath: Relative Path to the dataset.
        :return:
        """
        self.dataset =  pd.read_pickle( module_path + filepath)


        pass


    def update_model(self):
        pass

    def visualize_results(self):
        pass



class sarimaModel():

    def __init__(self, timedelta = 24):

        self.model = None
        self.timestamp_range = None
        self.dataset = None
        self.timedelta = timedelta


    def train(self, dataset):
        """
        Trains the model based on the dataset
        :param dataset: target dataset that is loaded in the model
        :return:
        """
        self.timestamp_range = (pd.Timestamp(dataset.index.values[0]), pd.Timestamp(dataset.index.values[-1]))

        self.dataset = dataset

        opt_AIC = self.optimize_AIC()

        mod = sm.tsa.statespace.SARIMAX(self.dataset,
                                        order=(opt_AIC[0][0], opt_AIC[0][1], opt_AIC[0][2]),
                                        seasonal_order=(opt_AIC[0][0], opt_AIC[1][1], opt_AIC[1][2], self.timedelta),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        self.model = mod.fit()

        self.evaluate_results(visualize= True)



    def optimize_AIC(self):
        """
        Optimizes the Seasonality, trend and noise parameters for the SARIMA model
        :param dataset: target dataset
        :return: Optimal AIC parameters
        """
        if type(self.dataset) != pd.DataFrame:
            raise Exception('Dataset not loaded. Please run train() first.')




        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], self.timedelta) for x in list(itertools.product(p, d, q))]
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

    def evaluate_results(self, visualize:bool):
        """
        Evaluates the prediction results of the model
        :param visualize: Boolean visualization option
        :return:
        """
        print(self.model.summary().tables[1])

        if visualize:
            self.model.plot_diagnostics(figsize=(16, 8))
            plt.show()




    def perform_prediction(self, start_date:str, visualize:bool):
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



