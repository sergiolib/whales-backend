import pandas as pd


class Datafile:
    def load_data(self, filename, formatter, formatter_type=pd.Series):
        raise NotImplementedError
