import os
import sys
from datetime import datetime

from dateutil.parser import parse
from dotenv import dotenv_values


class CommonUtils:

    def get_time(self):
        """
        :return current time:
        """
        return datetime.now().strftime("%H:%M:%S").__str__()

    def get_date(self):
        """
        :return current date:
        """
        return datetime.now().date().__str__()

    def get_difference_in_second(self, future_date_time: str, past_date_time: str):
        """
        :param future_date_time:
        :param past_date_time:
        :return difference in second:
        """
        future_date = parse(future_date_time)
        past_date = parse(past_date_time)
        difference = future_date - past_date
        total_seconds = difference.total_seconds()
        return total_seconds

    def get_difference_in_milisecond(self, future_date_time: str, past_date_time: str):
        """
        :param future_date_time:
        :param past_date_time:
        :return difference in milisecond:
        """
        total_seconds = self.get_difference_in_second(future_date_time, past_date_time)
        total_milisecond = total_seconds * 1000
        return total_milisecond

    def get_environment_variable(self, variable_name: str):
        """
        :param variable_name:
        :return environment variable:
        """
        if os.environ.get(variable_name) is None:
            enironment_variable = dotenv_values(".env")
            return enironment_variable[variable_name]
        else:
            return os.environ.get(variable_name)