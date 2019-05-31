import re
from pathlib import Path

class csvInterface():

    def __init__(self):

        self.model = {
            'passenger_id': '[0-9]+	',
            'source_latitude': '-[0-9]+.[0-9]+	',
            'source_longitude' : '-[0-9]+.[0-9]+	',
            'source_address': '',
            'destination_latitude': '-[0-9]+.[0-9]+	',
            'destination_longitude': '-[0-9]+.[0-9]+	',
            'destination_address': '',
            'request_date': '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}'
        }

        self.header = list(self.model.keys())


    def update_model(self, model: dict):
        """
        Updates the model of the csv interface
        :param model: dictionary with Key: column_name | Value: Regular Expression
        :return:
        """
        self.model = model
        self.header = list(self.model.keys())

    def compile_csv(self, source:Path, destination:Path, delimiter:str):
        """
        Compiles a given csv into an accepted format
        :return:
        """







